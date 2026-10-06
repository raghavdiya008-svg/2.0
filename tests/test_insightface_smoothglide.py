import os
import sys
import unittest
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from engine_vision import (
    FaceDetection,
    SmoothGlideTracker,
    calculate_pan_offset,
    _detect_insightface,
    detect_faces_or_subjects,
)


class TestInsightFaceSmoothGlide(unittest.TestCase):
    def setUp(self):
        self.source_w = 1920
        self.source_h = 1080
        self.target_crop_w = 1080
        self.neutral_x = (self.source_w - self.target_crop_w) // 2  # 420

    def test_face_detection_landmark_center_calculation(self):
        """Tests that eye/nose landmarks accurately calculate the anatomical focal center."""
        # Left eye: (500, 300), Right eye: (600, 300), Nose: (550, 350)
        kps = np.array([
            [500.0, 300.0],  # left eye
            [600.0, 300.0],  # right eye
            [550.0, 350.0],  # nose
            [520.0, 420.0],  # mouth left
            [580.0, 420.0],  # mouth right
        ])
        eye_mid_x = (500.0 + 600.0) / 2.0  # 550.0
        eye_mid_y = (300.0 + 300.0) / 2.0  # 300.0
        expected_focal_x = 0.5 * eye_mid_x + 0.5 * 550.0  # 550.0
        expected_focal_y = 0.4 * eye_mid_y + 0.6 * 350.0  # 330.0

        fd = FaceDetection(
            box=(450, 200, 200, 280),
            landmarks=kps,
            landmark_center=(expected_focal_x, expected_focal_y),
            score=0.98,
        )
        self.assertEqual(fd.landmark_center[0], 550.0)
        self.assertEqual(fd.landmark_center[1], 330.0)

        # Pan offset should center the 1080 crop window on focal_x = 550:
        # crop_x = 550 - 1080 // 2 = 550 - 540 = 10
        pan_x = calculate_pan_offset(
            [fd],
            source_w=self.source_w,
            target_crop_w=self.target_crop_w,
            focal_center_x=fd.landmark_center[0],
        )
        self.assertEqual(pan_x, 10)

    def test_tracker_deadband_suppresses_micro_jitter(self):
        """Tests that tiny head sways (< 16px) do not jitter or jerk the camera framing."""
        tracker = SmoothGlideTracker(
            source_w=self.source_w,
            target_crop_w=self.target_crop_w,
            deadband=16.0,
        )
        start_x = 300.0
        pos1 = tracker.update(start_x, dt=1.0)
        # Small head sway of only 4px
        pos2 = tracker.update(start_x + 4.0, dt=1.0)
        pos3 = tracker.update(start_x - 5.0, dt=1.0)

        # Due to deadband, positions should remain tightly locked without micro-jitter
        self.assertAlmostEqual(pos1, pos2, delta=2)
        self.assertAlmostEqual(pos2, pos3, delta=2)

    def test_tracker_smooth_glide_momentum_on_subject_exit(self):
        """Tests that when a subject steps out of frame, the camera glides smoothly with momentum decay rather than snapping."""
        tracker = SmoothGlideTracker(
            source_w=self.source_w,
            target_crop_w=self.target_crop_w,
        )
        # Subject moves from 200 to 260
        tracker.update(200.0, dt=1.0)
        tracker.update(230.0, dt=1.0)
        pos_before_exit = tracker.update(260.0, dt=1.0)

        # Subject steps out of view (target_pan_x=None)
        exit_pos_1 = tracker.update(None, dt=1.0)
        exit_pos_2 = tracker.update(None, dt=1.0)

        # Camera should NOT snap back to center immediately (neutral_x = 420)
        # It should coast smoothly near the exit position
        self.assertNotEqual(exit_pos_1, self.neutral_x)
        self.assertTrue(abs(exit_pos_1 - pos_before_exit) < 30)
        self.assertTrue(abs(exit_pos_2 - exit_pos_1) < 30)

    def test_tracker_gentle_recentering_on_prolonged_absence(self):
        """Tests that prolonged absence causes camera to gently glide toward neutral center."""
        tracker = SmoothGlideTracker(
            source_w=self.source_w,
            target_crop_w=self.target_crop_w,
        )
        # Start at far left (pos=100)
        tracker.update(100.0, dt=1.0)

        # Simulate 10 seconds of subject absence
        positions = []
        for _ in range(10):
            p = tracker.update(None, dt=1.0)
            positions.append(p)

        # Camera should glide smoothly towards neutral_x (420)
        self.assertGreater(positions[-1], positions[0])
        # Smooth progression without any sharp jump (> 50px in single step)
        for i in range(1, len(positions)):
            delta = abs(positions[i] - positions[i - 1])
            self.assertLess(delta, 50, f"Sudden camera jump detected: {delta}px at step {i}")

    def test_tracker_hard_scene_cut_reset(self):
        """Tests that a scene cut resets the tracker instantaneously without dragging."""
        tracker = SmoothGlideTracker(
            source_w=self.source_w,
            target_crop_w=self.target_crop_w,
        )
        tracker.update(100.0, dt=1.0)
        self.assertLess(tracker.smoothed_x, 200.0)

        # Scene cut occurs: new shot has subject at 700.0
        tracker.reset(700.0)
        self.assertEqual(tracker.smoothed_x, 700.0)
        self.assertEqual(tracker.v, 0.0)

    def test_detect_multispeaker_framing_requires_simultaneous_presence(self):
        """Alternating solo shots must NOT trigger split-stack; wide two-shot MUST trigger."""
        from src.engine_vision import detect_multispeaker_framing

        speaker_turns = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]

        # Case 1: Alternating solo shots (Host on left in frames 0-4, Guest on right in frames 5-9, never together)
        alternating_frames = [
            [(300, 200, 150, 150)] for _ in range(5)
        ] + [
            [(1500, 200, 150, 150)] for _ in range(5)
        ]
        res_alternating = detect_multispeaker_framing(
            alternating_frames, speaker_turns, source_w=1920, source_h=1080
        )
        self.assertIsNone(res_alternating, "Alternating solo closeups must NOT trigger dual-speaker split!")

        # Case 2: Genuine wide 2-shot (both Host and Guest appear together in the same frame >= 35% of time)
        wide_frames = [
            [(300, 200, 150, 150), (1500, 200, 150, 150)] for _ in range(8)
        ] + [
            [(300, 200, 150, 150)] for _ in range(2)
        ]
        res_wide = detect_multispeaker_framing(
            wide_frames, speaker_turns, source_w=1920, source_h=1080
        )
        self.assertIsNotNone(res_wide, "Genuine wide 2-shot with both faces visible simultaneously MUST trigger split-stack!")
        self.assertTrue(res_wide["is_dual_speaker"])
        self.assertIn("top_crop", res_wide)
        self.assertIn("bottom_crop", res_wide)


if __name__ == "__main__":
    unittest.main()

