#!/usr/bin/env python3
"""
tests/test_phase5_ffmpeg.py
----------------------------
Unit tests for Phase 5 Anti-Detection Engine, Title Hooks, and Dual-Logo Routing:
1. Micro-speed retiming (video setpts=PTS/{speed_factor} + audio atempo={speed_factor}).
2. Subtle color grading (eq=contrast=1.04:brightness=0.01:saturation=1.08:gamma=1.02).
3. Hook headline banner (drawtext centered at y=150 with box styling and character escaping).
4. Dual-logo branding stack (brand logo at W-w-40:40, 50% opacity watermark at W-w-30:H-h-30).
5. Filtergraph pad routing integrity across all optional permutation combinations.
"""

import os
import sys
import unittest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from engine_ffmpeg import (
    build_ffmpeg_command,
    escape_ffmpeg_drawtext,
    RenderOptions,
    SPEED_FACTOR,
)


class TestPhase5FFmpegEngine(unittest.TestCase):

    def setUp(self):
        self.dummy_in = "input_test.mp4"
        self.dummy_out = "output_test.mp4"
        self.video_coords = {"x": 0, "y": 190, "width": 1080, "height": 1540, "scale_x": 1.0, "scale_y": 1.0}
        self.text_coords = {"x": 60, "y": 80, "font_size": 48, "text_content": ""}

    def _get_filter_complex(self, cmd):
        idx = cmd.index("-filter_complex")
        return cmd[idx + 1]

    def _get_audio_filter(self, cmd):
        idx = cmd.index("-filter:a")
        return cmd[idx + 1]

    def test_01_default_speed_and_color_grading(self):
        """Verify default speed_factor=1.12 and anti-detection eq color grading."""
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
        )
        fc = self._get_filter_complex(cmd)
        af = self._get_audio_filter(cmd)

        self.assertIn("setpts=PTS/1.12", fc)
        self.assertIn("atempo=1.12", af)
        self.assertIn("eq=contrast=1.04:brightness=0.01:saturation=1.08:gamma=1.02", fc)
        self.assertIn("fps=30,setpts=PTS-STARTPTS,setpts=PTS/1.12", fc)

    def test_02_custom_speed_factor(self):
        """Verify custom speed_factor propagates to both video and audio filters."""
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            speed_factor=1.18,
        )
        fc = self._get_filter_complex(cmd)
        af = self._get_audio_filter(cmd)

        self.assertIn("setpts=PTS/1.18", fc)
        self.assertIn("atempo=1.18", af)

    def test_03_speed_factor_via_render_options(self):
        """Verify speed_factor configured in RenderOptions."""
        opts = RenderOptions(speed_factor=1.25)
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            options=opts,
        )
        fc = self._get_filter_complex(cmd)
        af = self._get_audio_filter(cmd)

        self.assertIn("setpts=PTS/1.25", fc)
        self.assertIn("atempo=1.25", af)

    def test_04_headline_text_drawtext_injection_and_styling(self):
        """Verify headline_text injects styled drawtext filter."""
        headline = "5 SECRETS TO VIRAL CLIPS"
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            headline_text=headline,
        )
        fc = self._get_filter_complex(cmd)

        self.assertIn(f"drawtext=text='{headline}'", fc)
        self.assertIn("x=(w-text_w)/2:y=150", fc)
        self.assertIn("fontsize=64:fontcolor=white", fc)
        self.assertIn("box=1:boxcolor=black@0.6:boxborderw=15", fc)
        self.assertIn("[comp_title]", fc)

    def test_05_headline_text_escaping(self):
        """Verify characters like quotes, colons, brackets, and semicolons are escaped."""
        raw_text = "Wait: it's [crazy]; don't miss this!"
        escaped = escape_ffmpeg_drawtext(raw_text)

        self.assertNotIn(":", escaped.replace("\\:", ""))
        self.assertNotIn("[", escaped.replace("\\[", ""))
        self.assertNotIn("]", escaped.replace("\\]", ""))
        self.assertNotIn(";", escaped.replace("\\;", ""))
        self.assertIn("'\\''", escaped)

        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            headline_text=raw_text,
        )
        fc = self._get_filter_complex(cmd)
        self.assertIn(f"drawtext=text='{escaped}'", fc)

    def test_06_brand_logo_overlay(self):
        """Verify primary brand logo is scaled and overlaid at top-right with 40px padding."""
        brand_path = "assets/brand.png"
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            brand_logo_path=brand_path,
        )
        fc = self._get_filter_complex(cmd)

        self.assertIn("-i", cmd)
        self.assertIn(brand_path, cmd)
        self.assertIn("scale='min(200,iw)':-1,format=rgba[brand_logo]", fc)
        self.assertIn("overlay=W-w-40:40[comp_brand]", fc)

    def test_07_watermark_logo_overlay(self):
        """Verify watermark logo has 50% opacity and overlays at bottom-right with 30px padding."""
        wm_path = "assets/watermark.png"
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            watermark_logo_path=wm_path,
        )
        fc = self._get_filter_complex(cmd)

        self.assertIn(wm_path, cmd)
        self.assertIn("scale='min(250,iw)':-1,format=rgba,colorchannelmixer=aa=0.5[wm_semi]", fc)
        self.assertIn("overlay=W-w-30:H-h-30[comp_wm]", fc)

    def test_08_dual_logo_and_headline_complete_stack(self):
        """Verify full stack: Brand Logo -> Watermark -> Headline -> Output."""
        brand_path = "assets/brand.png"
        wm_path = "assets/watermark.png"
        headline = "TOP HOOK"

        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            brand_logo_path=brand_path,
            watermark_logo_path=wm_path,
            headline_text=headline,
            speed_factor=1.15,
        )
        fc = self._get_filter_complex(cmd)

        self.assertIn("[comp0][brand_logo]overlay=W-w-40:40[comp_brand]", fc)
        self.assertIn("[comp_brand][wm_semi]overlay=W-w-30:H-h-30[comp_wm]", fc)
        self.assertIn(f"[comp_wm]drawtext=text='{headline}':x=(w-text_w)/2:y=150:fontsize=64:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=15[comp_title]", fc)
        self.assertIn("[comp_title]null,format=yuv420p[vout]", fc)

    def test_09_permutations_zero_orphaned_pads(self):
        """Verify no orphaned stream labels across all permutation combinations."""
        test_cases = [
            (None, None, None),
            ("brand.png", None, None),
            (None, "wm.png", None),
            (None, None, "HOOK"),
            ("brand.png", "wm.png", None),
            ("brand.png", None, "HOOK"),
            (None, "wm.png", "HOOK"),
            ("brand.png", "wm.png", "HOOK"),
        ]

        for brand, wm, headline in test_cases:
            with self.subTest(brand=brand, wm=wm, headline=headline):
                cmd, _ = build_ffmpeg_command(
                    input_path=self.dummy_in,
                    output_path=self.dummy_out,
                    video_coords=self.video_coords,
                    text_coords=self.text_coords,
                    brand_logo_path=brand,
                    watermark_logo_path=wm,
                    headline_text=headline,
                )
                fc = self._get_filter_complex(cmd)
                self.assertTrue(fc.endswith("[vout]"), f"Filter complex did not end in [vout]: {fc}")
                self.assertIn("-map", cmd)
                self.assertIn("[vout]", cmd)

    def test_10_dual_speaker_layout_with_phase5_features(self):
        """Verify Phase 5 speed and color grading integrate into dual speaker podcast layout."""
        trajectory = {
            "layout": "dual_speaker_split",
            "is_dual_speaker": True,
            "dual_speaker_layout": {
                "top_crop": {"x": 100, "y": 0, "w": 1215, "h": 1080},
                "bottom_crop": {"x": 600, "y": 0, "w": 1215, "h": 1080},
            },
        }
        cmd, _ = build_ffmpeg_command(
            input_path=self.dummy_in,
            output_path=self.dummy_out,
            video_coords=self.video_coords,
            text_coords=self.text_coords,
            trajectory=trajectory,
            speed_factor=1.14,
            headline_text="DUAL PODCAST EPISODE",
        )
        fc = self._get_filter_complex(cmd)

        self.assertIn("eq=contrast=1.04:brightness=0.01:saturation=1.08:gamma=1.02", fc)
        self.assertIn("setpts=PTS/1.14", fc)
        self.assertIn("drawbox=x=0:y=958:w=1080:h=4:color=black@0.6:t=fill,format=yuv420p[comp0]", fc)
        self.assertIn("drawtext=text='DUAL PODCAST EPISODE'", fc)
        self.assertIn("[vout]", fc)


if __name__ == "__main__":
    unittest.main()
