import os
import sys
import time
import threading
import unittest
import sqlite3
import tempfile

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from task_queue import RenderQueueManager, RenderJob, DB_PATH
from server import _run_render_job


class TestPhase2Audit(unittest.TestCase):
    def setUp(self):
        # Clear the database for clean tests
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("DELETE FROM render_jobs")
            
    def test_start_worker_concurrency_and_idempotency(self):
        mgr = RenderQueueManager()
        
        # Simulate concurrent start_worker calls from 10 threads
        threads = [threading.Thread(target=mgr.start_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify worker thread is alive
        self.assertIsNotNone(mgr._worker_thread)
        self.assertTrue(mgr._worker_thread.is_alive())

        # Gracefully stop worker
        mgr.stop_worker(timeout=2.0)
        self.assertTrue(mgr._worker_thread is None or not mgr._worker_thread.is_alive())

    def test_execute_job_error_envelope_and_status_transition(self):
        mgr = RenderQueueManager()

        def faulty_job():
            raise ValueError("Simulated catastrophic compute error")

        job = mgr.submit_job(faulty_job, job_id="faulty_test_job")
        fetched_job = mgr.get_job("faulty_test_job")
        self.assertEqual(fetched_job.status, "PENDING")

        executed_job = mgr.execute_job(job)
        self.assertEqual(executed_job.status, "FAILED")
        self.assertIn("Simulated catastrophic compute error", executed_job.error)
        
        fetched_job_after = mgr.get_job("faulty_test_job")
        self.assertEqual(fetched_job_after.status, "FAILED")
        self.assertIn("Simulated catastrophic compute error", fetched_job_after.error)

    def test_kwarg_filtering_prevents_signature_mismatch(self):
        mgr = RenderQueueManager()

        def mock_cleanup(job):
            pass

        # A function that strictly takes only 1 argument without **kwargs
        def strict_receiver(payload):
            return f"received:{payload['key']}"

        job = mgr.submit_job(
            strict_receiver,
            {"key": "test_val"},
            job_id="strict_job",
            _cleanup_callback=mock_cleanup,
        )

        # Verify _cleanup_callback was filtered out of kwargs into metadata
        self.assertNotIn("_cleanup_callback", job.kwargs)
        self.assertEqual(job.metadata["_cleanup_callback"], mock_cleanup)

        executed_job = mgr.execute_job(job)
        self.assertEqual(executed_job.status, "COMPLETED")
        self.assertEqual(executed_job.result, "received:test_val")

    def test_cleanup_old_jobs_reentrancy_no_deadlock(self):
        mgr = RenderQueueManager()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            out_path = f.name
            
        # Write some data so it actually exists
        with open(out_path, "w") as f:
            f.write("test")

        job = mgr.submit_job(
            lambda: {"output_path": out_path},
            job_id="aged_job"
        )
        mgr.execute_job(job)
        
        # Verify job is completed
        fetched = mgr.get_job("aged_job")
        self.assertEqual(fetched.status, "COMPLETED")

        # Force job completion timestamp into past
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("UPDATE render_jobs SET updated_at = datetime('now', '-100000 seconds') WHERE job_id = 'aged_job'")

        # Run cleanup with 0 max_age to force eviction
        mgr.cleanup_old_jobs(max_age_seconds=10)

        self.assertFalse(os.path.exists(out_path))
        self.assertIsNone(mgr.get_job("aged_job"))

    def test_get_job_position_thread_safe(self):
        mgr = RenderQueueManager()
        job1 = mgr.submit_job(lambda: 1, job_id="pos_job_1")
        time.sleep(0.01) # ensure order
        job2 = mgr.submit_job(lambda: 2, job_id="pos_job_2")
        time.sleep(0.01)
        job3 = mgr.submit_job(lambda: 3, job_id="pos_job_3")

        self.assertEqual(mgr.get_job_position("pos_job_1"), 1)
        self.assertEqual(mgr.get_job_position("pos_job_2"), 2)
        self.assertEqual(mgr.get_job_position("pos_job_3"), 3)
        self.assertEqual(mgr.get_job_position("non_existent"), 1)

    def test_run_render_job_output_verification(self):
        import server
        dummy_input = os.path.join(server.INPUTS_DIR, "dummy_test_vid.mp4")
        os.makedirs(server.INPUTS_DIR, exist_ok=True)
        with open(dummy_input, "w") as f:
            f.write("dummy content")

        orig_render_clip = server.render_clip
        try:
            server.render_clip = lambda **kwargs: None
            with self.assertRaises(RuntimeError) as cm:
                _run_render_job({
                    "video": "dummy_test_vid.mp4",
                })
            self.assertIn("was not created or is 0 bytes", str(cm.exception))
        finally:
            server.render_clip = orig_render_clip
            if os.path.exists(dummy_input):
                os.remove(dummy_input)


if __name__ == "__main__":
    unittest.main()
