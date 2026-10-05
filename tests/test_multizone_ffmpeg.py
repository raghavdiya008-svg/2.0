import os
import sys
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import engine_ffmpeg

def test_multizone_shot_timeline_filtergraph(tmp_path):
    input_path = "temp/input.mp4"
    output_path = "temp/output.mp4"
    video_coords = {"width": 1080, "height": 1540, "x": 0, "y": 190}
    text_coords = {"x": 0, "y": 0, "font_size": 48, "text_content": ""}

    ass_file = tmp_path / "test_dialogue.ass"
    ass_file.write_text("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello world\n")

    shot_timeline = [
        {
            "start": 0.0,
            "end": 5.0,
            "type": "single",
            "camera": "zone_host",
            "zone": {"x": 0, "y": 0, "width": 960, "height": 1080}
        },
        {
            "start": 5.0,
            "end": 10.0,
            "type": "split_stack",
            "camera": "split",
            "top_zone": {"x": 0, "y": 0, "width": 960, "height": 1080},
            "bot_zone": {"x": 960, "y": 0, "width": 960, "height": 1080}
        }
    ]

    trajectory = {
        "layout": "director_multizone",
        "shot_timeline": shot_timeline
    }

    cmd, _ = engine_ffmpeg.build_ffmpeg_command(
        input_path=input_path,
        output_path=output_path,
        video_coords=video_coords,
        text_coords=text_coords,
        trajectory=trajectory,
        ass_path=str(ass_file)
    )

    fc = cmd[cmd.index("-filter_complex") + 1]

    # Verify split for 2 shots
    assert "split=2[raw_s0][raw_s1]" in fc
    # Verify trim filters
    assert "trim=start=0.00:end=5.00" in fc
    assert "trim=start=5.00:end=10.00" in fc
    # Verify split stack 9:8 vertical stack and divider line
    assert "vstack=inputs=2" in fc
    assert "drawbox=x=0:y=958:w=1080:h=4:color=black@0.6" in fc
    # Verify concatenation of both shots
    assert "concat=n=2:v=1:a=0" in fc
    # Verify subtitle placement at seam (MarginV=960)
    assert "MarginV=960" in fc

def test_camera_zone_single_layout():
    input_path = "temp/input.mp4"
    output_path = "temp/output.mp4"
    video_coords = {"width": 1080, "height": 1540, "x": 0, "y": 190}
    text_coords = {"x": 0, "y": 0, "font_size": 48, "text_content": ""}

    trajectory = {
        "layout": "camera_zone",
        "zone": {"x": 100, "y": 50, "width": 800, "height": 1000}
    }

    cmd, _ = engine_ffmpeg.build_ffmpeg_command(
        input_path=input_path,
        output_path=output_path,
        video_coords=video_coords,
        text_coords=text_coords,
        trajectory=trajectory
    )

    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" in fc
    assert "crop=w='min(iw,800)':h='min(ih,1000)':x='max(0,min(100,iw-min(iw,800)))':y='max(0,min(50,ih-min(ih,1000)))'" in fc
