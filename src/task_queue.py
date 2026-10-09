#!/usr/bin/env python3
"""
src/task_queue.py
-----------------
Thread-Safe FIFO Render Queue & VRAM Guardian.
Manages concurrent render jobs with hardware VRAM memory reclamation:
1. Thread-safe FIFO task scheduling backed by queue.Queue.
2. VRAM Guardian: calls torch.cuda.empty_cache() and gc.collect() between jobs
   to prevent CUDA Out-Of-Memory (OOM) during heavy batch processing.
3. Provides context manager and programmatic memory telemetry.
"""

import os
import sys
import gc
import time
import queue
import logging
import threading
import sqlite3
import json
import ctypes
from typing import Callable, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger("task_queue")

# Monkey-patch numpy ufunc to add __qualname__ dynamically if missing (NumPy < 2.0 on Python 3.10+)
def _patch_ufunc_qualname():
    try:
        import numpy as np
        if not hasattr(np.ufunc, '__qualname__'):
            class _MappingProxyStruct(ctypes.Structure):
                _fields_ = [('ob_refcnt', ctypes.c_ssize_t), ('ob_type', ctypes.c_void_p), ('mapping', ctypes.py_object)]
            proxy = np.ufunc.__dict__
            proxy_obj = _MappingProxyStruct.from_address(id(proxy))
            ctypes.pythonapi.PyDict_SetItem(
                ctypes.py_object(proxy_obj.mapping),
                ctypes.py_object('__qualname__'),
                ctypes.py_object(property(lambda self: getattr(self, '__name__', str(self))))
            )
    except Exception as _e:
        logger.debug(f"[task_queue] ufunc patch notice: {_e}")

_patch_ufunc_qualname()

def _safe_json_dumps(obj: Any) -> str:
    """Safe json serializer that never crashes on unhandled callable/ufunc/object types."""
    def _default_serializer(o):
        if hasattr(o, '__name__'):
            return o.__name__
        if hasattr(o, '__dict__'):
            return o.__dict__
        return str(o)
    return json.dumps(obj, default=_default_serializer)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
DB_PATH = os.path.join(_PROJECT_ROOT, "pipeline.db")


@dataclass
class RenderJob:
    job_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return round(self.completed_at - self.started_at, 3)
        return 0.0


class RenderQueueManager:
    """
    FIFO queue manager and VRAM memory guardian for heavy render jobs.
    """

    _instance: Optional["RenderQueueManager"] = None
    _lock = threading.RLock()

    def __init__(self, maxsize: int = 0):
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = RenderQueueManager._lock
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute('''CREATE TABLE IF NOT EXISTS render_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT CHECK(status IN ('queued', 'running', 'completed', 'failed')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                payload_json TEXT NOT NULL,
                error_message TEXT
            )''')
            
    def _update_job_status(self, job_id: str, status: str, payload: dict = None, error_message: str = None):
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            if payload is not None:
                conn.execute(
                    "UPDATE render_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP, payload_json = ? WHERE job_id = ?",
                    (status, _safe_json_dumps(payload), job_id)
                )
            elif error_message is not None:
                conn.execute(
                    "UPDATE render_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP, error_message = ? WHERE job_id = ?",
                    (status, error_message, job_id)
                )
            else:
                conn.execute(
                    "UPDATE render_jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                    (status, job_id)
                )

    @classmethod
    def get_instance(cls) -> "RenderQueueManager":
        """Singleton accessor for global queue manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def is_gpu_available() -> bool:
        """Checks if PyTorch with CUDA acceleration is active."""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    @staticmethod
    def cleanup_vram() -> Dict[str, Any]:
        """
        Forces Python garbage collection and clears PyTorch CUDA memory cache.
        Safe to call on CPU-only machines.
        """
        # 1. Host RAM garbage collection
        gc.collect()

        stats = {"gpu_available": False, "freed": True}
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu_available"] = True
                before_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                before_res = torch.cuda.memory_reserved() / (1024 * 1024)

                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

                after_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                after_res = torch.cuda.memory_reserved() / (1024 * 1024)

                stats["before_allocated_mb"] = round(before_alloc, 2)
                stats["after_allocated_mb"] = round(after_alloc, 2)
                stats["before_reserved_mb"] = round(before_res, 2)
                stats["after_reserved_mb"] = round(after_res, 2)
                logger.debug(
                    f"[VRAM Guardian] Cache purged: reserved {before_res:.1f}MB -> {after_res:.1f}MB"
                )
        except Exception as e:
            stats["error"] = str(e)

        return stats

    @staticmethod
    def get_vram_stats() -> Dict[str, Any]:
        """Returns current VRAM usage telemetry."""
        stats = {
            "gpu_available": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "device_name": "CPU",
        }
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu_available"] = True
                stats["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                stats["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
                stats["max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                stats["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return stats

    @contextmanager
    def vram_guard(self, label: str = "task"):
        """
        Context manager that automatically cleans up GPU memory before and after
        a compute block to avoid VRAM fragmentation.
        """
        self.cleanup_vram()
        start_t = time.time()
        try:
            yield
        finally:
            self.cleanup_vram()
            elapsed = time.time() - start_t
            logger.debug(f"[VRAM Guard] '{label}' completed in {elapsed:.2f}s, VRAM reclaimed.")

    def submit_job(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        **kwargs
    ) -> RenderJob:
        """Enqueues a task for FIFO execution with separated internal control metadata."""
        jid = job_id or f"job_{int(time.time() * 1000)}_{self.queue.qsize() + 1}"
        # Separate framework control kwargs (e.g. _cleanup_callback) from function kwargs
        internal_keys = {"_cleanup_callback"}
        meta = {k: kwargs[k] for k in internal_keys if k in kwargs}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in internal_keys}

        job = RenderJob(job_id=jid, func=func, args=args, kwargs=clean_kwargs, metadata=meta)
        self.queue.put(job)
        logger.info(f"[task_queue] Enqueued job: {jid} (Queue depth: {self.queue.qsize()})")
        
        payload_data = {"kwargs": clean_kwargs, "metadata": {k: (v.__name__ if callable(v) else str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v) for k, v in meta.items()}}
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO render_jobs (job_id, status, payload_json, updated_at) VALUES (?, 'queued', ?, CURRENT_TIMESTAMP)",
                (jid, _safe_json_dumps(payload_data))
            )
            
        return job

    def execute_job(self, job: RenderJob) -> RenderJob:
        """
        Executes a single job synchronously with VRAM isolation and bulletproof error boundaries.
        Guarantees status is updated to COMPLETED or FAILED and evicted from running_jobs under all conditions.
        """
        job.status = "RUNNING"
        job.started_at = time.time()
        with self._lock:
            try:
                with self.queue.mutex:
                    self.queue.queue = type(self.queue.queue)([j for j in self.queue.queue if getattr(j, 'job_id', None) != job.job_id])
            except Exception:
                pass
        self._update_job_status(job.job_id, "running")
        print(f"[task_queue] >>> [Heartbeat] Starting render job {job.job_id}...", flush=True)

        try:
            with self.vram_guard(label=job.job_id):
                try:
                    job.result = job.func(*job.args, **job.kwargs)
                    job.status = "COMPLETED"
                except Exception as e:
                    job.status = "FAILED"
                    err_msg = getattr(e, "stderr", None)
                    if err_msg and str(err_msg).strip():
                        job.error = f"{e}\nDetails: {str(err_msg).strip()}"
                    else:
                        job.error = str(e)
                    logger.error(f"[task_queue] Job {job.job_id} failed during execution: {job.error}", exc_info=True)
        except Exception as outer_e:
            # Handles failure inside vram_guard __enter__ or __exit__
            job.status = "FAILED"
            job.error = str(outer_e)
            logger.error(f"[task_queue] Job {job.job_id} failed in VRAM guardian / execution envelope: {outer_e}", exc_info=True)
        finally:
            job.completed_at = time.time()
            dur = round(job.completed_at - job.started_at, 1)
            print(f"[task_queue] <<< [Heartbeat] Job {job.job_id} {job.status.lower()} in {dur}s.", flush=True)
            if job.status == "COMPLETED":
                payload_data = {"kwargs": job.kwargs, "result": job.result}
                self._update_job_status(job.job_id, "completed", payload=payload_data)
            else:
                self._update_job_status(job.job_id, "failed", error_message=job.error)

        return job

    def process_all_jobs(self) -> List[RenderJob]:
        """Processes all currently enqueued jobs sequentially."""
        processed = []
        while not self.queue.empty():
            try:
                job = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                self.execute_job(job)
                processed.append(job)
            finally:
                self.queue.task_done()
        return processed

    def start_worker(self):
        """Starts a background daemon thread to process jobs continuously (thread-safe and idempotent)."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            
            def _worker():
                while not self._stop_event.is_set():
                    try:
                        job = self.queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"[task_queue] Queue get error: {e}", exc_info=True)
                        continue

                    try:
                        self.execute_job(job)
                    except Exception as e:
                        logger.error(f"[task_queue] Unhandled worker error processing {getattr(job, 'job_id', 'unknown')}: {e}", exc_info=True)
                    finally:
                        try:
                            self.queue.task_done()
                        except ValueError:
                            pass
                        
            self._worker_thread = threading.Thread(target=_worker, daemon=True, name="RenderQueueWorker")
            self._worker_thread.start()
            logger.info("[task_queue] Background render worker thread started.")

    def stop_worker(self, timeout: Optional[float] = 5.0):
        """Stops the background worker thread gracefully."""
        with self._lock:
            self._stop_event.set()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
                self._worker_thread = None
                logger.info("[task_queue] Background render worker thread stopped.")

    def get_job(self, job_id: str) -> Optional[RenderJob]:
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, payload_json, error_message, created_at, updated_at FROM render_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                status, payload_json, error_message, created_at, updated_at = row
                try:
                    payload = json.loads(payload_json)
                except:
                    payload = {}
                
                # Map SQLite status to RenderJob status
                status_map = {
                    "queued": "PENDING",
                    "running": "RUNNING",
                    "completed": "COMPLETED",
                    "failed": "FAILED"
                }
                
                job = RenderJob(job_id=job_id, func=None)
                job.status = status_map.get(status, "PENDING")
                job.error = error_message
                job.result = payload.get("result")
                return job
                
        # Also check pending jobs in queue as fallback
        with self._lock:
            try:
                with self.queue.mutex:
                    queue_items = list(self.queue.queue)
                for job in queue_items:
                    if getattr(job, "job_id", None) == job_id and getattr(job, "status", None) == "PENDING":
                        return job
            except Exception:
                pass
        return None

    def get_job_position(self, job_id: str) -> int:
        """Returns the 1-based queue position of a pending job in a thread-safe manner."""
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM render_jobs WHERE job_id = ? AND status = 'queued'", (job_id,))
            target_row = cursor.fetchone()
            if target_row:
                cursor.execute('''
                    SELECT COUNT(*) FROM render_jobs 
                    WHERE status = 'queued' 
                    AND rowid <= ?
                ''', (target_row[0],))
                row = cursor.fetchone()
                if row and row[0] > 0:
                    return row[0]
                
        # Fallback to queue iteration
        with self._lock:
            try:
                with self.queue.mutex:
                    queue_items = list(self.queue.queue)
                for idx, job in enumerate(queue_items):
                    if getattr(job, "job_id", None) == job_id:
                        return idx + 1
            except Exception:
                pass
        return 1

    def cleanup_old_jobs(self, max_age_seconds: float = 86400):
        """
        Removes jobs older than max_age_seconds from database (both completed and failed).
        """
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT job_id, payload_json FROM render_jobs 
                WHERE status IN ('completed', 'failed') AND strftime('%s', 'now') - strftime('%s', updated_at) > ?
            ''', (max_age_seconds,))
            rows = cursor.fetchall()
            
            for job_id, payload_json in rows:
                try:
                    payload = json.loads(payload_json)
                    result = payload.get("result", {})
                    out_path = result.get("output_path") if isinstance(result, dict) else None
                    if out_path and os.path.exists(out_path):
                        os.remove(out_path)
                        logger.info(f"[task_queue] Cleaned up expired render artifact: {out_path}")
                except Exception as e:
                    logger.error(f"[task_queue] Cleanup artifact failed for {job_id}: {e}", exc_info=True)
            
            conn.execute('''
                DELETE FROM render_jobs 
                WHERE strftime('%s', 'now') - strftime('%s', updated_at) > ?
            ''', (max_age_seconds,))

        with self._lock:
            try:
                cleaned_ids = {r[0] for r in rows}
                with self.queue.mutex:
                    self.queue.queue = type(self.queue.queue)([j for j in self.queue.queue if getattr(j, 'job_id', None) not in cleaned_ids])
            except Exception:
                pass

# Module-level convenience functions
def cleanup_vram() -> Dict[str, Any]:
    return RenderQueueManager.cleanup_vram()

def get_vram_stats() -> Dict[str, Any]:
    return RenderQueueManager.get_vram_stats()

@contextmanager
def vram_guard(label: str = "task"):
    manager = RenderQueueManager.get_instance()
    with manager.vram_guard(label=label):
        yield

if __name__ == "__main__":
    print(f"VRAM Stats: {get_vram_stats()}")
    cleanup_res = cleanup_vram()
    print(f"VRAM Cleanup Result: {cleanup_res}")
    
    # Test FIFO queue
    mgr = RenderQueueManager()
    mgr.submit_job(lambda x, y: x + y, 10, 20, job_id="test_add")
    mgr.submit_job(lambda s: s.upper(), "hello world", job_id="test_upper")
    results = mgr.process_all_jobs()
    for r in results:
        print(f"Job {r.job_id}: status={r.status}, result={r.result}, duration={r.duration}s")
