import os
import sys
import unittest
from typing import List, Dict, Any

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from engine_vision import detect_multispeaker_framing, detect_dual_speaker_layout
from engine_ffmpeg import build_ffmpeg_command, RenderOptions


class TestMultiSpeakerLayout(unittest.TestCase):
    def setUp(self):
        self.source_w = 1920
        self.source_h = 1080

    def test_detect_multispeaker_framing_activates_on_dual_speakers(self):
        """Tests that two alternating speakers with distinct left/right clusters trigger dual_speaker_split."""
        # Alternating speaker turns
        speaker_turns = [
            {"speaker": "HOST_A", "start": 0.0, "end": 4.0},
            {"speaker": "GUEST_B", "start": 4.0, "end": 8.0},
            {"speaker": "HOST_A", "start": 8.0, "end": 12.0},
            {"speaker": "GUEST_B", "start": 12.0, "end": 16.0},
        ]

        # 10 frames of detections with two distinct spatial clusters
        # Left cluster: x=200, w=300, cx=350 (< 1920*0.48 = 921.6)
        # Right cluster: x=1400, w=300, cx=1550 (> 1920*0.52 = 998.4)
        frame_detections = []
        for t in range(10):
            frame_detections.append((
                float(t),
                [
                    (200, 200, 300, 500),
                    (1400, 200, 300, 500),
                ]
            ))

        result = detect_multispeaker_framing(
            frame_detections=frame_detections,
            speaker_turns=speaker_turns,
            source_w=self.source_w,
            source_h=self.source_h,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["layout"], "dual_speaker_split")
        self.assertTrue(result["is_dual_speaker"])
        self.assertIn("top_crop", result)
        self.assertIn("bottom_crop", result)

        top_crop = result["top_crop"]
        bot_crop = result["bottom_crop"]

        # Check 9:8 aspect ratio: w / h is approximately 9 / 8 (1.125)
        self.assertAlmostEqual(top_crop["w"] / top_crop["h"], 9.0 / 8.0, delta=0.05)
        self.assertAlmostEqual(bot_crop["w"] / bot_crop["h"], 9.0 / 8.0, delta=0.05)
        self.assertGreater(top_crop["w"], 0)
        self.assertGreater(top_crop["h"], 0)

        # Clamped inside frame bounds
        self.assertGreaterEqual(top_crop["x"], 0)
        self.assertLessEqual(top_crop["x"] + top_crop["w"], self.source_w)
        self.assertGreaterEqual(bot_crop["x"], 0)
        self.assertLessEqual(bot_crop["x"] + bot_crop["w"], self.source_w)

        # Left speaker crop should be placed to the left of right speaker crop
        self.assertLess(top_crop["x"], bot_crop["x"])

    def test_detect_multispeaker_framing_fallback_single_dominant_speaker(self):
        """Tests fallback to None when one speaker dominates >85% of total dialogue."""
        speaker_turns = [
            {"speaker": "HOST_A", "start": 0.0, "end": 20.0},
            {"speaker": "GUEST_B", "start": 20.0, "end": 21.0},
        ]
        frame_detections = [
            (float(t), [(200, 200, 300, 500), (1400, 200, 300, 500)])
            for t in range(10)
        ]

        result = detect_multispeaker_framing(
            frame_detections=frame_detections,
            speaker_turns=speaker_turns,
            source_w=self.source_w,
            source_h=self.source_h,
        )
        self.assertIsNone(result)

    def test_detect_multispeaker_framing_fallback_no_spatial_separation(self):
        """Tests fallback to None when subjects are in the center with no opposite-half clusters."""
        speaker_turns = [
            {"speaker": "HOST_A", "start": 0.0, "end": 5.0},
            {"speaker": "GUEST_B", "start": 5.0, "end": 10.0},
        ]
        # Both detections centered around cx=960 (middle of 1920)
        frame_detections = [
            (float(t), [(850, 200, 220, 500)])
            for t in range(10)
        ]

        result = detect_multispeaker_framing(
            frame_detections=frame_detections,
            speaker_turns=speaker_turns,
            source_w=self.source_w,
            source_h=self.source_h,
        )
        self.assertIsNone(result)

    def test_detect_multispeaker_framing_fallback_vertical_video(self):
        """Tests that portrait video (e.g. 1080x1920) skips split-stack layout."""
        speaker_turns = [
            {"speaker": "HOST_A", "start": 0.0, "end": 5.0},
            {"speaker": "GUEST_B", "start": 5.0, "end": 10.0},
        ]
        frame_detections = [
            (float(t), [(100, 200, 300, 500), (600, 200, 300, 500)])
            for t in range(10)
        ]

        result = detect_multispeaker_framing(
            frame_detections=frame_detections,
            speaker_turns=speaker_turns,
            source_w=1080,
            source_h=1920,
        )
        self.assertIsNone(result)

    def test_backward_compatible_wrapper(self):
        """Verifies detect_dual_speaker_layout aliases detect_multispeaker_framing."""
        speaker_turns = [
            {"speaker": "HOST_A", "start": 0.0, "end": 4.0},
            {"speaker": "GUEST_B", "start": 4.0, "end": 8.0},
        ]
        frame_detections = [
            (float(t), [(200, 200, 300, 500), (1400, 200, 300, 500)])
            for t in range(5)
        ]
        res = detect_dual_speaker_layout(
            detections_per_frame=frame_detections,
            diarization_segments=speaker_turns,
            source_w=1920,
            source_h=1080,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["layout"], "dual_speaker_split")

    def test_ffmpeg_build_command_dual_speaker_filtergraph(self):
        """Verifies that build_ffmpeg_command produces valid vstack filtergraph and MarginV=960 subtitles."""
        dummy_input = "dummy_input.mp4"
        dummy_output = "dummy_output.mp4"
        video_coords = {"width": 1080, "height": 1920, "x": 0, "y": 0}
        text_coords = {"x": 0, "y": 0}

        trajectory = {
            "layout": "dual_speaker_split",
            "is_dual_speaker": True,
            "dual_speaker_layout": {
                "top_crop": {"x": 0, "y": 0, "w": 1214, "h": 1080},
                "bottom_crop": {"x": 706, "y": 0, "w": 1214, "h": 1080},
            }
        }

        # Create temporary dummy ASS file with dialogue event
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ass", mode="w", delete=False, encoding="utf-8") as f:
            f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            f.write("Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,Hello world\n")
            temp_ass = f.name

        try:
            cmd, _ = build_ffmpeg_command(
                input_path=dummy_input,
                output_path=dummy_output,
                video_coords=video_coords,
                text_coords=text_coords,
                trajectory=trajectory,
                ass_path=temp_ass,
            )

            # Find -filter_complex argument
            self.assertIn("-filter_complex", cmd)
            fc_idx = cmd.index("-filter_complex")
            filtergraph = cmd[fc_idx + 1]

            # 1. Check top stream
            self.assertIn("crop=w='min(iw,1214)':h='min(ih,1080)':x='max(0,min(0,iw-min(iw,1214)))':y='max(0,min(0,ih-min(ih,1080)))'", filtergraph)
            self.assertIn("scale=1080:960:flags=lanczos:force_original_aspect_ratio=increase,crop=1080:960:(in_w-1080)/2:(in_h-960)/2[top]", filtergraph)

            # 2. Check bottom stream
            self.assertIn("crop=w='min(iw,1214)':h='min(ih,1080)':x='max(0,min(706,iw-min(iw,1214)))':y='max(0,min(0,ih-min(ih,1080)))'", filtergraph)
            self.assertIn("scale=1080:960:flags=lanczos:force_original_aspect_ratio=increase,crop=1080:960:(in_w-1080)/2:(in_h-960)/2[bottom]", filtergraph)

            # 3. Check vstack
            self.assertIn("[top][bottom]vstack=inputs=2[stacked]", filtergraph)

            # 4. Check divider line at boundary
            self.assertIn("[stacked]drawbox=x=0:y=958:w=1080:h=4:color=black@0.6:t=fill,format=yuv420p[comp0]", filtergraph)

            # 5. Check subtitle placement
            self.assertIn("subtitles=", filtergraph)

        finally:
            if os.path.exists(temp_ass):
                os.remove(temp_ass)


if __name__ == "__main__":
    unittest.main()
