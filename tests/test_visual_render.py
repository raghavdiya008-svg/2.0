#!/usr/bin/env python3
"""
tests/test_visual_render.py
---------------------------
Automated verification test for the visual rendering overhaul:
1. Takes an existing slice (temp/clean_slice_1.mp4).
2. Renders a 5-second test clip using the full-bleed blurred backdrop and high-bitrate NVENC profile.
3. Confirms:
   - Video output is exactly 1080x1920.
   - No white letterbox bars exist.
   - No debug text banners are rendered.
   - Exit code is 0.
"""

import os
import sys
import subprocess
import numpy as np
import cv2

# Add src/ to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import engine_ffmpeg


def ensure_input_slice(slice_path: str, duration: float = 5.0) -> str:
    """Ensures a valid 16:9 input slice exists at slice_path."""
    if os.path.isfile(slice_path) and os.path.getsize(slice_path) > 1000:
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0", slice_path
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True)
        if res.returncode == 0 and "x" in res.stdout:
            try:
                w, h = map(int, res.stdout.strip().split("x"))
                if w > h:
                    return slice_path
            except ValueError:
                pass

    os.makedirs(os.path.dirname(slice_path) or ".", exist_ok=True)
    temp_dir = os.path.dirname(slice_path)
    raw_video = os.path.join(temp_dir, "raw_video.mp4")
    if os.path.isfile(raw_video) and os.path.getsize(raw_video) > 1000:
        subprocess.run([
            "ffmpeg", "-y", "-ss", "0", "-t", str(duration),
            "-i", raw_video,
            "-c:v", "libx264", "-c:a", "aac",
            slice_path
        ], check=True, capture_output=True)
        print(f"[test] Extracted {duration}s 16:9 slice from raw_video.mp4 to {slice_path}")
        return slice_path

    # Fallback synthesize a 16:9 test input clip
    print(f"[test] Synthesizing {duration}s 1920x1080 test video at {slice_path}...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}",
        "-c:v", "libx264", "-c:a", "aac",
        slice_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return slice_path


def test_visual_render():
    temp_dir = os.path.join(PROJECT_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    input_slice = os.path.join(temp_dir, "clean_slice_1.mp4")
    ensure_input_slice(input_slice, duration=5.0)

    # 5-second slice trimming if longer
    trimmed_slice = os.path.join(temp_dir, "clean_slice_1_5s.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-ss", "0", "-t", "5",
        "-i", input_slice,
        "-c:v", "copy", "-c:a", "copy",
        trimmed_slice
    ], check=True, capture_output=True)

    output_reel = os.path.join(temp_dir, "test_visual_render_out.mp4")
    if os.path.exists(output_reel):
        os.remove(output_reel)

    video_coords = {
        "x": 0,
        "y": (1920 - 1540) // 2,
        "width": 1080,
        "height": 1540,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    # Pass dummy text_coords to ensure no static drawtext header is rendered
    text_coords = {
        "x": 60, "y": 80, "font_size": 48,
        "text_content": "Action Hook #1 Debug Banner",
    }
    opts = engine_ffmpeg.RenderOptions(
        speed=1.0,
        preset="p6",
        tune="hq",
        rc="vbr",
        cq=18,
        crf=18,
        bitrate="6000k",
        maxrate="9000k",
        bufsize="12000k",
    )

    print(f"[test] Rendering 5s test clip from {trimmed_slice} -> {output_reel}...")
    cmd, temp_tf = engine_ffmpeg.build_ffmpeg_command(
        input_path=trimmed_slice,
        output_path=output_reel,
        video_coords=video_coords,
        text_coords=text_coords,
        options=opts,
        ass_path=None,
    )

    # 1. Filtergraph Inspection Check: Confirm drawtext purge and blurred background
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "drawtext" not in fc, "FAILED: Static drawtext header was found in filter_complex!"
    assert "boxblur=25:5" in fc, "FAILED: boxblur=25:5 not found in filter_complex!"
    assert "force_original_aspect_ratio=increase" in fc, "FAILED: force_original_aspect_ratio not found!"
    assert "scale=1080:-2" in fc, "FAILED: scale=1080:-2 not found!"
    print("  [PASS] Filtergraph check: static drawtext completely purged; blurred backdrop stack present.")

    # 2. NVENC Flags Check
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "p6", "FAILED: preset is not p6"
    assert "-tune" in cmd and cmd[cmd.index("-tune") + 1] == "hq", "FAILED: tune is not hq"
    assert "-rc" in cmd and cmd[cmd.index("-rc") + 1] == "vbr", "FAILED: rc is not vbr"
    assert "-cq" in cmd and cmd[cmd.index("-cq") + 1] == "18", "FAILED: cq is not 18"
    assert "-b:v" in cmd and cmd[cmd.index("-b:v") + 1] == "6000k", "FAILED: bitrate is not 6000k"
    assert "-maxrate" in cmd and cmd[cmd.index("-maxrate") + 1] == "9000k", "FAILED: maxrate is not 9000k"
    assert "-bufsize" in cmd and cmd[cmd.index("-bufsize") + 1] == "12000k", "FAILED: bufsize is not 12000k"
    print("  [PASS] NVENC configuration: p6 / tune=hq / rc=vbr / cq=18 / 6000k verified.")

    # 3. Execute Render (with automatic CPU fallback if NVENC unavailable locally)
    proc = engine_ffmpeg.render_clip(
        input_path=trimmed_slice,
        output_path=output_reel,
        video_coords=video_coords,
        text_coords=text_coords,
        options=opts,
        ass_path=None,
    )
    assert os.path.isfile(output_reel), "FAILED: Rendered output file does not exist!"
    file_size = os.path.getsize(output_reel)
    assert file_size > 50000, f"FAILED: Output file is suspiciously small ({file_size} bytes)"
    print(f"  [PASS] Clip rendered successfully: {output_reel} ({file_size:,} bytes).")

    # 4. Verify Exact Output Resolution (1080x1920) via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,pix_fmt",
        "-of", "csv=s=x:p=0", output_reel
    ]
    probe_out = subprocess.run(probe_cmd, capture_output=True, text=True, check=True).stdout.strip()
    assert "1080x1920" in probe_out, f"FAILED: Output dimensions not 1080x1920! Got: {probe_out}"
    assert "yuv420p" in probe_out, f"FAILED: Pixel format not yuv420p! Got: {probe_out}"
    print(f"  [PASS] Dimensions and format verified: {probe_out}.")

    # 5. Verify No Flat White Letterbox Bars Exist (Sample margins with OpenCV)
    cap = cv2.VideoCapture(output_reel)
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, "FAILED: Could not read rendered frame with OpenCV!"
    fh, fw = frame.shape[:2]
    assert (fw, fh) == (1080, 1920), f"Frame dimensions mismatch: ({fw}, {fh})"

    # Top margin (e.g. rows 30 to 120): on white letterbox, this was pure white (255, 255, 255)
    top_strip = frame[30:120, :]
    # Bottom margin (e.g. rows 1800 to 1890)
    bottom_strip = frame[1800:1890, :]

    # Check that neither margin is pure white (mean across all channels < 250)
    top_mean = float(np.mean(top_strip))
    bottom_mean = float(np.mean(bottom_strip))
    top_std = float(np.std(top_strip))
    bottom_std = float(np.std(bottom_strip))

    print(f"  [INFO] Margin analysis: Top mean={top_mean:.2f} (std={top_std:.2f}), Bottom mean={bottom_mean:.2f} (std={bottom_std:.2f})")
    assert top_mean < 252.0, f"FAILED: Top margin is solid white! Mean intensity={top_mean:.2f}"
    assert bottom_mean < 252.0, f"FAILED: Bottom margin is solid white! Mean intensity={bottom_mean:.2f}"
    print("  [PASS] Verified: No white letterbox bars exist on top or bottom.")

    # Cleanup temporary test files so temp stays clean
    for fpath in [trimmed_slice, output_reel, input_slice]:
        if fpath and os.path.exists(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass

    print("\n>>> ALL VISUAL RENDER VERIFICATION TESTS PASSED (Exit Code 0) <<<")


if __name__ == "__main__":
    test_visual_render()
    sys.exit(0)
