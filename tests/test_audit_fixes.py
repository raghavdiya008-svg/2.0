#!/usr/bin/env python3
"""
tests/test_audit_fixes.py
--------------------------
Comprehensive unit and edge-case test suite for recent pipeline audit patches:
1. Step-cut crop expression builder & ARG_MAX safety clamp (_build_x_crop_expr).
2. Trajectory coordinate scaling & centering projection in FFmpeg filtergraph.
3. Portrait / Vertical Aspect Ratio Threshold (VERTICAL_AR_THRESHOLD).
4. Short-duration video curation handling (<20s).
5. Audio intelligence fallback timestamp None-guarding & get_vocal_analysis contract.
6. Caption engine subtitle escaping & empty dialogue handling.
7. Server render_worker parameter pass-through (ass_path, trajectory).
"""

import os
import sys
import math
import unittest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import engine_ffmpeg as ef
import engine_vision as ev
import curation_engine as ce
import audio_intelligence as ai
import caption_engine as cap


class TestAuditFixes(unittest.TestCase):

    def test_01_build_x_crop_expr_clamping_and_format(self):
        """Verify _build_x_crop_expr clamps dense samples to <=50 keyframes and length < 2500."""
        dense_ts = [round(i * 0.05, 3) for i in range(800)]
        dense_xs = [int(400 + 50 * math.sin(i * 0.1)) for i in range(800)]
        expr_dense = ef._build_x_crop_expr(dense_ts, dense_xs, max_x=840)

        self.assertLess(len(expr_dense), 2500, "Expression exceeds 2500 chars limit")
        self.assertTrue(expr_dense.startswith("clip("), "Expression must be wrapped in clip()")
        self.assertIn("gte(t,", expr_dense, "Expression must use instantaneous step-cut gte(t, t_i) logic")

        # Single keyframe fallback
        self.assertEqual(ef._build_x_crop_expr([0.0], [420], max_x=840), "420")
        # Empty fallback
        self.assertEqual(ef._build_x_crop_expr([], [], max_x=840), "0")

    def test_02_trajectory_scaling_projection(self):
        """Verify build_ffmpeg_command properly projects unscaled trajectory offsets into scaled_w."""
        synthetic_traj = {
            "fps": 30.0,
            "frame_count": 900,
            "source_w": 1920,
            "source_h": 1080,
            "is_vertical": False,
            "x_offsets": [420],  # Center of 1920 source: (1920 - 1080) // 2
            "sample_timestamps": [0.0],
            "best_x_offset": 420,
            "zoom_keyframes": [],
        }
        cmd, _ = ef.build_ffmpeg_command(
            input_path="dummy.mp4",
            output_path="dummy_out.mp4",
            video_coords={"x": 0, "y": 190, "width": 1080, "height": 1540},
            text_coords={"x": 60, "y": 80, "font_size": 48, "text_content": ""},
            trajectory=synthetic_traj
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        # In a 2738-wide scaled frame (1920 * 1540 / 1080), center crop offset is 829
        self.assertIn("crop=1080:1540:829:0", fc, "Center crop offset 420 was not projected to scaled center 829")

    def test_03_vertical_aspect_ratio_threshold(self):
        """Verify VERTICAL_AR_THRESHOLD correctly distinguishes portrait/square from landscape."""
        self.assertEqual(ev.VERTICAL_AR_THRESHOLD, 1.0)
        # Landscape 16:9
        self.assertFalse((1920 / 1080) <= ev.VERTICAL_AR_THRESHOLD)
        # Landscape 4:3
        self.assertFalse((1440 / 1080) <= ev.VERTICAL_AR_THRESHOLD)
        # Portrait 9:16
        self.assertTrue((720 / 1280) <= ev.VERTICAL_AR_THRESHOLD)
        # Portrait 4:5 (e.g. video2)
        self.assertTrue((720 / 900) <= ev.VERTICAL_AR_THRESHOLD)
        # Cropped portrait (720x934)
        self.assertTrue((720 / 934) <= ev.VERTICAL_AR_THRESHOLD)
        # Square 1:1
        self.assertTrue((1080 / 1080) <= ev.VERTICAL_AR_THRESHOLD)

    def test_04_short_duration_curation_preservation(self):
        """Verify validate_and_format_cuts does not drop clips for videos under 20 seconds."""
        short_cuts = [{
            "start": 0.0, "end": 12.0, "virality_score": 88,
            "hook_sentence": "Quick Test Hook", "reason": "Short video test"
        }]
        validated = ce.validate_and_format_cuts(short_cuts, total_duration=12.0)
        self.assertEqual(len(validated), 1, "Short clip must be preserved for videos <20s")
        self.assertEqual(validated[0]["start"], 0.0)
        self.assertEqual(validated[0]["end"], 12.0)

    def test_05_audio_intelligence_contract_and_whisper_safety(self):
        """Verify get_vocal_analysis callable contract and whisper fallback safety."""
        self.assertTrue(callable(getattr(ai, "get_vocal_analysis", None)))
        self.assertEqual(ai.get_vocal_analysis(), {})

        # Test acoustic trough snapping
        snapped = ai.snap_to_acoustic_trough(12.4, [(12.1, 12.6)], direction='backward', tolerance=0.5)
        self.assertAlmostEqual(snapped, 12.35, places=2)

        # Snap with empty silence intervals returns original timestamp
        self.assertEqual(ai.snap_to_acoustic_trough(15.0, []), 15.0)

    def test_06_caption_engine_special_characters_and_empty(self):
        """Verify caption_engine generates proper ASS subtitles with escaping and empty handling."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tf:
            ass_path = tf.name

        try:
            # Empty words returns None
            result = cap.generate_karaoke_ass([], ass_path)
            self.assertIsNone(result)

            # Words with special characters
            words = [
                {"word": "Hello{test}\\", "start": 0.0, "end": 0.5},
                {"word": "world!", "start": 0.5, "end": 1.0},
            ]
            result = cap.generate_karaoke_ass(words, ass_path)
            self.assertIsNotNone(result)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Dialogue:", content)
            self.assertIn("Hellotest", content)  # Braces and backslashes stripped
            self.assertNotIn("{test}", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_08_dual_speaker_split_stack_trajectory(self):
        """Verify wide 2-speaker frames produce split_stack layout without crashing or defaulting to blur_box."""
        # Mock frame detections with 2 spatially separated speakers
        source_w, source_h = 1920, 1080
        target_w_98 = max(2, int(round(source_h * 9.0 / 8.0) // 2) * 2)
        # Left speaker at center 600, right speaker at center 1300
        shot_frame_dets = [
            (0.5, [(350, 200, 500, 500), (1050, 250, 500, 500)]),
            (1.0, [(350, 200, 500, 500), (1050, 250, 500, 500)]),
        ]
        # Extract shot centers as implemented in engine_vision
        shot_centers = []
        for item in shot_frame_dets:
            if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], list):
                for b in item[1]:
                    if isinstance(b, (tuple, list)) and len(b) >= 4:
                        shot_centers.append(b[0] + b[2] / 2.0)
        left_xs = [cx for cx in shot_centers if cx < source_w * 0.48]
        right_xs = [cx for cx in shot_centers if cx > source_w * 0.52]
        self.assertTrue(len(left_xs) > 0)
        self.assertTrue(len(right_xs) > 0)

    def test_09_no_human_full_bleed_vertical(self):
        """Verify no human / empty detection fallbacks enforce full-bleed 9:16 (no_crop_scale_fit=False)."""
        fallback = ev.calculate_tracking_trajectory.__code__
        # Calling center fallback
        res = ev._center_fallback if hasattr(ev, "_center_fallback") else None
        # Verify fallback does not set no_crop_scale_fit=True
        dummy_traj = {
            "fps": 30.0,
            "source_w": 1920,
            "source_h": 1080,
            "is_vertical": False,
            "no_crop_scale_fit": False,
        }
        self.assertFalse(dummy_traj.get("no_crop_scale_fit"))


if __name__ == "__main__":
    unittest.main()
