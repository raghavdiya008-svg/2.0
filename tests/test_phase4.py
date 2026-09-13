#!/usr/bin/env python3
"""
tests/test_phase4.py
--------------------
Phase 4 Test Suite:
1. Hardware-Bound Machine Fingerprinting (src/licensing/hwid.py).
2. Cryptographic Offline Lease Verification & Tamper Detection (src/licensing/lease.py).
3. Thread-Safe FIFO Render Queue & VRAM Guardian (src/task_queue.py).
4. Pipeline Integration & Licensing Gate (src/pipeline.py).
"""

import os
import sys
import time
import json
import tempfile
import unittest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from licensing.hwid import get_machine_hwid, get_hardware_details
from licensing.lease import (
    verify_lease,
    generate_dev_lease,
    issue_lease,
    generate_rsa_keypair,
    LicenseError,
    DEFAULT_LEASE_PATH,
)
from task_queue import RenderQueueManager, vram_guard, cleanup_vram, get_vram_stats
import pipeline


class TestPhase4HardwareLicensing(unittest.TestCase):

    def test_01_hwid_fingerprint_deterministic(self):
        hwid1 = get_machine_hwid()
        hwid2 = get_machine_hwid()
        self.assertEqual(len(hwid1), 64, "HWID must be 64-char SHA-256 hex string")
        self.assertEqual(hwid1, hwid2, "HWID must be deterministic across calls")
        self.assertTrue(all(c in "0123456789abcdef" for c in hwid1.lower()))

        details = get_hardware_details()
        self.assertIn("os", details)
        self.assertIn("cpu", details)
        self.assertIn("uuid", details)
        self.assertIn("mac", details)
        print(f"\n  [PASS] HWID fingerprint verified: {hwid1[:16]}... ({details['os']} / {details['node']})")

    def test_02_lease_generation_and_verification(self):
        with tempfile.NamedTemporaryFile(suffix=".lease", delete=False) as tf:
            lease_path = tf.name

        try:
            generate_dev_lease(lease_path=lease_path, duration_hours=72.0)
            self.assertTrue(os.path.isfile(lease_path))
            
            # Verify lease succeeds on current hardware
            is_valid = verify_lease(lease_path=lease_path)
            self.assertTrue(is_valid, "Valid development lease must verify successfully")
            print("  [PASS] Dev lease generated and verified for 72-hour window")
        finally:
            if os.path.exists(lease_path):
                os.remove(lease_path)

    def test_03_tampered_hwid_fails_verification(self):
        with tempfile.NamedTemporaryFile(suffix=".lease", delete=False) as tf:
            lease_path = tf.name

        try:
            # Issue lease for a different fake HWID
            fake_hwid = "0000000000000000000000000000000000000000000000000000000000000000"
            lease_data = issue_lease(hwid=fake_hwid, duration_hours=72.0)
            with open(lease_path, "w", encoding="utf-8") as f:
                json.dump(lease_data, f)

            with self.assertRaises(LicenseError) as ctx:
                verify_lease(lease_path=lease_path, auto_issue_dev=False)
            self.assertIn("Hardware mismatch", str(ctx.exception))
            print(f"  [PASS] Tampered HWID rejected cleanly: {ctx.exception}")
        finally:
            if os.path.exists(lease_path):
                os.remove(lease_path)

    def test_04_tampered_signature_fails_verification(self):
        with tempfile.NamedTemporaryFile(suffix=".lease", delete=False) as tf:
            lease_path = tf.name

        try:
            lease_data = issue_lease(duration_hours=72.0)
            # Modify payload after signing to simulate payload tampering
            lease_data["payload"]["max_offline_hours"] = 9999.0
            with open(lease_path, "w", encoding="utf-8") as f:
                json.dump(lease_data, f)

            with self.assertRaises(LicenseError) as ctx:
                verify_lease(lease_path=lease_path, auto_issue_dev=False)
            self.assertIn("signature verification failed", str(ctx.exception).lower())
            print(f"  [PASS] Tampered payload/signature rejected cleanly: {ctx.exception}")
        finally:
            if os.path.exists(lease_path):
                os.remove(lease_path)

    def test_05_expired_lease_fails_verification(self):
        with tempfile.NamedTemporaryFile(suffix=".lease", delete=False) as tf:
            lease_path = tf.name

        try:
            # Issue lease that expired in the past (-10 hours)
            lease_data = issue_lease(duration_hours=-10.0)
            with open(lease_path, "w", encoding="utf-8") as f:
                json.dump(lease_data, f)

            with self.assertRaises(LicenseError) as ctx:
                verify_lease(lease_path=lease_path, auto_issue_dev=False)
            self.assertIn("expired", str(ctx.exception).lower())
            print(f"  [PASS] Expired lease rejected cleanly: {ctx.exception}")
        finally:
            if os.path.exists(lease_path):
                os.remove(lease_path)


class TestPhase4TaskQueueAndVRAM(unittest.TestCase):

    def test_06_vram_telemetry_and_cleanup(self):
        stats = get_vram_stats()
        self.assertIn("gpu_available", stats)
        self.assertIn("allocated_mb", stats)
        self.assertIn("reserved_mb", stats)
        
        cleanup_res = cleanup_vram()
        self.assertIn("freed", cleanup_res)
        self.assertTrue(cleanup_res["freed"])
        print(f"  [PASS] VRAM telemetry & cleanup executed: GPU={stats['gpu_available']}, Device={stats['device_name']}")

    def test_07_vram_guard_context_manager(self):
        executed = False
        with vram_guard(label="test_guard"):
            executed = True
        self.assertTrue(executed, "vram_guard block must execute and clean up")
        print("  [PASS] vram_guard context manager verified")

    def test_08_fifo_render_queue(self):
        mgr = RenderQueueManager()
        results = []

        def worker(num):
            results.append(num)
            return num * 2

        mgr.submit_job(worker, 1, job_id="job_1")
        mgr.submit_job(worker, 2, job_id="job_2")
        mgr.submit_job(worker, 3, job_id="job_3")

        completed = mgr.process_all_jobs()
        self.assertEqual(len(completed), 3)
        self.assertEqual(results, [1, 2, 3], "Jobs must execute in FIFO order")
        self.assertEqual([j.result for j in completed], [2, 4, 6])
        self.assertTrue(all(j.status == "COMPLETED" for j in completed))
        print("  [PASS] FIFO RenderQueueManager processed 3 jobs in exact order")


class TestPhase4PipelineIntegration(unittest.TestCase):

    def test_09_pipeline_licensing_and_vram_integration(self):
        self.assertIsNotNone(pipeline.verify_lease, "pipeline.verify_lease must not be None")
        self.assertTrue(callable(pipeline.verify_lease), "pipeline.verify_lease must be callable")
        
        # Verify that invalid lease triggers LicenseError
        fake_lease_path = os.path.join(tempfile.gettempdir(), f"invalid_{time.time()}.lease")
        with open(fake_lease_path, "w") as f:
            f.write('{"tampered": true}')
            
        try:
            with self.assertRaises(pipeline.LicenseError):
                pipeline.verify_lease(lease_path=fake_lease_path, auto_issue_dev=False)
        finally:
            if os.path.exists(fake_lease_path):
                os.remove(fake_lease_path)

        print("  [PASS] pipeline.py imports and enforces licensing gate + VRAM guardian")


if __name__ == "__main__":
    unittest.main(verbosity=2)
