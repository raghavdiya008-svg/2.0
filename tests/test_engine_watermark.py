#!/usr/bin/env python3
"""
tests/test_engine_watermark.py
------------------------------
Unit tests for Phase 5 Blurred Watermark & Logo Detector for Pre-Render Rejection.
Validates:
1. Clean, sharp synthetic video -> returns False (accepted).
2. Uniform bokeh / naturally out-of-focus scene -> returns False (accepted, no false positives).
3. Artificial blurred watermark in Top-Left corner -> returns True (detected).
4. Artificial blurred watermark in Bottom-Right corner -> returns True (detected).
5. Artificial blurred watermark in Top-Right corner -> returns True (detected).
6. Nonexistent/invalid file handling -> returns False gracefully.
7. Pipeline integration: run_pipeline raises WatermarkRejectionError and aborts before AI stages.
"""

import os
import sys
import unittest
import tempfile
import cv2
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from engine_watermark import detect_blurred_logos, WatermarkRejectionError
import pipeline


def _create_synthetic_test_video(
    output_path: str,
    duration_sec: float = 3.0,
    fps: int = 10,
    blur_corner: str = None,  # "top_left", "top_right", "bottom_right", "none", "full_bokeh"
    width: int = 640,
    height: int = 480,
) -> str:
    """Generates a synthetic MP4 video with controlled high-frequency detail and optional blur patches."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)

    for f_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if blur_corner == "full_bokeh":
            # Smooth low-frequency gradient / bokeh background (no high-frequency edges)
            x_vals = np.linspace(50, 180, width, dtype=np.uint8)
            y_vals = np.linspace(60, 200, height, dtype=np.uint8)
            xx, yy = np.meshgrid(x_vals, y_vals)
            frame[:, :, 0] = xx
            frame[:, :, 1] = yy
            frame[:, :, 2] = (xx // 2 + yy // 2)
            frame = cv2.GaussianBlur(frame, (31, 31), 15)
        else:
            # High-frequency textured video (checkerboard grid + high-contrast shapes + text)
            # Create high edge detail across the entire frame
            for y in range(0, height, 20):
                for x in range(0, width, 20):
                    c = 220 if ((x // 20) + (y // 20)) % 2 == 0 else 40
                    frame[y : y + 20, x : x + 20] = (c, c, c)

            # Add high-contrast text and geometric lines across all 4 quadrants
            cv2.putText(frame, f"Frame {f_idx} - High Detail", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "TOP LEFT DETAIL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "TOP RIGHT DETAIL", (width - 240, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "BOTTOM RIGHT DETAIL", (width - 270, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "BOTTOM LEFT DETAIL", (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if blur_corner == "top_left":
                # Artificial blur patch over Top-Left watermark zone
                patch = frame[25:85, 20:140]
                blurred_patch = cv2.GaussianBlur(patch, (35, 35), 20)
                frame[25:85, 20:140] = blurred_patch

            elif blur_corner == "top_right":
                # Artificial blur patch over Top-Right watermark zone
                patch = frame[25:85, width - 150 : width - 20]
                blurred_patch = cv2.GaussianBlur(patch, (35, 35), 20)
                frame[25:85, width - 150 : width - 20] = blurred_patch

            elif blur_corner == "bottom_right":
                # Artificial blur patch over Bottom-Right watermark zone
                patch = frame[height - 85 : height - 25, width - 150 : width - 20]
                blurred_patch = cv2.GaussianBlur(patch, (35, 35), 20)
                frame[height - 85 : height - 25, width - 150 : width - 20] = blurred_patch

        out.write(frame)

    out.release()
    return output_path


class TestEngineWatermark(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="wm_test_")

    def tearDown(self):
        # Clean up temp files
        for f in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_01_clean_sharp_video_accepted(self):
        """Verify that a clean video with high detail and no blurred patches returns False."""
        vid_path = os.path.join(self.temp_dir, "clean_video.mp4")
        _create_synthetic_test_video(vid_path, duration_sec=3.0, blur_corner=None)

        result = detect_blurred_logos(vid_path, sample_interval_sec=1.0)
        self.assertFalse(result, "Clean video without watermark blur must return False")

    def test_02_natural_bokeh_accepted_no_false_positives(self):
        """Verify that a video with smooth, naturally blurred bokeh does not trigger false positives."""
        vid_path = os.path.join(self.temp_dir, "bokeh_video.mp4")
        _create_synthetic_test_video(vid_path, duration_sec=3.0, blur_corner="full_bokeh")

        result = detect_blurred_logos(vid_path, sample_interval_sec=1.0)
        self.assertFalse(result, "Full-frame natural bokeh must not trigger false positive rejection")

    def test_03_artificial_blur_top_left_detected(self):
        """Verify detection of artificial blurred watermark in Top-Left corner."""
        vid_path = os.path.join(self.temp_dir, "blurred_tl.mp4")
        _create_synthetic_test_video(vid_path, duration_sec=3.0, blur_corner="top_left")

        result = detect_blurred_logos(vid_path, sample_interval_sec=1.0, min_persistent_frames=2)
        self.assertTrue(result, "Artificial blur patch in top-left corner must return True")

    def test_04_artificial_blur_top_right_detected(self):
        """Verify detection of artificial blurred watermark in Top-Right corner."""
        vid_path = os.path.join(self.temp_dir, "blurred_tr.mp4")
        _create_synthetic_test_video(vid_path, duration_sec=3.0, blur_corner="top_right")

        result = detect_blurred_logos(vid_path, sample_interval_sec=1.0, min_persistent_frames=2)
        self.assertTrue(result, "Artificial blur patch in top-right corner must return True")

    def test_05_artificial_blur_bottom_right_detected(self):
        """Verify detection of artificial blurred watermark in Bottom-Right corner."""
        vid_path = os.path.join(self.temp_dir, "blurred_br.mp4")
        _create_synthetic_test_video(vid_path, duration_sec=3.0, blur_corner="bottom_right")

        result = detect_blurred_logos(vid_path, sample_interval_sec=1.0, min_persistent_frames=2)
        self.assertTrue(result, "Artificial blur patch in bottom-right corner must return True")

    def test_06_nonexistent_file_handling(self):
        """Verify graceful False return on missing file."""
        result = detect_blurred_logos("nonexistent_video_path.mp4")
        self.assertFalse(result)

    def test_07_pipeline_integration_rejection(self):
        """Verify pipeline immediately raises WatermarkRejectionError on blurred video."""
        vid_path = os.path.join(self.temp_dir, "pipeline_rejection.mp4")
        _create_synthetic_test_video(vid_path, duration_sec=3.0, blur_corner="top_left")

        out_dir = os.path.join(self.temp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        with self.assertRaises(WatermarkRejectionError) as ctx:
            pipeline.run_pipeline(
                input_video_path=vid_path,
                output_dir=out_dir,
            )

        self.assertIn("Video contains a blurred logo/watermark", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
