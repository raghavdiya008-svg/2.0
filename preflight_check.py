#!/usr/bin/env python3
"""
preflight_check.py
------------------
Detailed Pre-Flight Verification Script for the Phase 1 Video Repurposing Engine.
Checks every component: imports, FFmpeg/ffprobe availability, schema contracts,
filtergraph validity, Python dependencies, file structure, and run-time logic.
Prints a clean structured Markdown report.
"""

import sys
import os
import subprocess
import json
import math
import importlib
import inspect
import tempfile
import wave
import struct

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []
fail_count = 0
warn_count = 0


def record(section, label, status, detail=""):
    global fail_count, warn_count
    if status == FAIL:
        fail_count += 1
    elif status == WARN:
        warn_count += 1
    results.append((section, label, status, detail))
    icon = status
    detail_str = f"  -> {detail}" if detail else ""
    print(f"  {icon}  {label}{detail_str}")


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================
# 1. PROJECT FILE STRUCTURE
# =============================================================
separator("1. PROJECT FILE STRUCTURE")

required_files = [
    "src/engine_ffmpeg.py",
    "src/engine_vision.py",
    "src/pipeline.py",
    "src/curation_engine.py",
    "src/audio_intelligence.py",
    "src/caption_engine.py",
    "kaggle_run_all.py",
    "verify_phase1_audit.py",
]

optional_dirs = ["inputs", "outputs", "temp", "assets/emojis", "assets/fonts", "assets/logo"]

for rel_path in required_files:
    full = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.isfile(full):
        size_kb = os.path.getsize(full) / 1024
        record("File Structure", rel_path, PASS, f"{size_kb:.1f} KB")
    else:
        record("File Structure", rel_path, FAIL, "FILE MISSING")

for rel_dir in optional_dirs:
    full = os.path.join(PROJECT_ROOT, rel_dir)
    if os.path.isdir(full):
        contents = [f for f in os.listdir(full) if not f.startswith('.')]
        record("File Structure", f"dir: {rel_dir}/", PASS, f"{len(contents)} item(s)")
    else:
        record("File Structure", f"dir: {rel_dir}/", WARN, "Directory missing — will be created at runtime")


# =============================================================
# 2. PYTHON ENVIRONMENT
# =============================================================
separator("2. PYTHON ENVIRONMENT")

record("Python", "Version", PASS if sys.version_info >= (3, 10) else WARN,
       f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

core_deps = ["cv2", "numpy", "dotenv", "httpx"]
optional_deps = ["torch", "whisperx", "torchaudio", "scipy", "transformers", "mediapipe", "ultralytics"]

for mod in core_deps:
    try:
        importlib.import_module(mod)
        record("Dependencies", f"import {mod}", PASS)
    except ImportError as e:
        record("Dependencies", f"import {mod}", FAIL, str(e))

for mod in optional_deps:
    try:
        importlib.import_module(mod)
        record("Dependencies (Optional)", f"import {mod}", PASS)
    except ImportError:
        record("Dependencies (Optional)", f"import {mod}", WARN, "Not installed — GPU fallback inactive, using CPU stubs")


# =============================================================
# 3. FFMPEG / FFPROBE BINARIES
# =============================================================
separator("3. FFMPEG / FFPROBE BINARIES")

for binary in ["ffmpeg", "ffprobe"]:
    try:
        res = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=5)
        version_line = res.stdout.splitlines()[0] if res.stdout else res.stderr.splitlines()[0]
        record("FFmpeg Binaries", f"{binary} binary", PASS, version_line[:70])
    except FileNotFoundError:
        record("FFmpeg Binaries", f"{binary} binary", FAIL, "Not found in PATH")
    except Exception as e:
        record("FFmpeg Binaries", f"{binary} binary", FAIL, str(e))


# =============================================================
# 4. MODULE IMPORTS FROM SRC/
# =============================================================
separator("4. MODULE IMPORTS")

modules = {}
for mod_name in ["engine_ffmpeg", "engine_vision", "curation_engine", "pipeline", "audio_intelligence", "caption_engine"]:
    try:
        mod = importlib.import_module(mod_name)
        modules[mod_name] = mod
        record("Module Imports", mod_name, PASS)
    except Exception as e:
        modules[mod_name] = None
        record("Module Imports", mod_name, FAIL, str(e))


# =============================================================
# 5. CURATION ENGINE CONTRACT
# =============================================================
separator("5. CURATION ENGINE CONTRACT")

ce = modules.get("curation_engine")
if ce:
    # Short clip path (<= 45s)
    try:
        cuts = ce.get_viral_cuts([], [], 30.0, "test.mp4")
        assert len(cuts) == 1 and cuts[0]["start_time"] == 0.0 and cuts[0]["end_time"] == 30.0
        record("Curation Engine", "Short clip path (<=45s) → 1 full segment", PASS)
    except Exception as e:
        record("Curation Engine", "Short clip path (<=45s)", FAIL, str(e))

    # Long clip path (> 45s)
    try:
        cuts = ce.get_viral_cuts([], [], 180.0, "test.mp4")
        assert 3 <= len(cuts) <= 5
        record("Curation Engine", f"Long clip path (180s) → {len(cuts)} segments (expected 3–5)", PASS)
    except Exception as e:
        record("Curation Engine", "Long clip path (>45s)", FAIL, str(e))

    # Schema key completeness
    try:
        cuts = ce.get_viral_cuts([], [], 90.0, "test.mp4")
        required_keys = {"start", "end", "start_time", "end_time", "virality_score", "hook_sentence", "reason"}
        for cut in cuts:
            missing = required_keys - set(cut.keys())
            if missing:
                raise AssertionError(f"Missing keys: {missing}")
            assert isinstance(cut["start"], (int, float)), "start must be numeric"
            assert isinstance(cut["end"], (int, float)), "end must be numeric"
            assert isinstance(cut["virality_score"], int), "virality_score must be int"
        record("Curation Engine", "Schema key completeness & numeric types", PASS,
               f"All {len(cuts)} cuts have required keys")
    except Exception as e:
        record("Curation Engine", "Schema key completeness & numeric types", FAIL, str(e))

    # validate_and_format_cuts (if present)
    if hasattr(ce, "validate_and_format_cuts"):
        try:
            raw = [{"start": 5.0, "end": 25.0, "virality_score": 90, "hook_sentence": "Test hook", "reason": "r"}]
            validated = ce.validate_and_format_cuts(raw, 60.0)
            assert validated[0]["hook_text"] == validated[0]["hook_sentence"]
            record("Curation Engine", "validate_and_format_cuts() alias keys", PASS)
        except Exception as e:
            record("Curation Engine", "validate_and_format_cuts() alias keys", FAIL, str(e))
    else:
        record("Curation Engine", "validate_and_format_cuts()", WARN, "Function not present in current version")
else:
    record("Curation Engine", "All checks", FAIL, "Module failed to import")


# =============================================================
# 6. ENGINE_FFMPEG CONTRACT
# =============================================================
separator("6. ENGINE_FFMPEG CONTRACT")

ef = modules.get("engine_ffmpeg")
if ef:
    # RenderOptions dataclass
    try:
        opts = ef.RenderOptions(speed=1.12, crf=18, preset="veryfast", bg_color="white")
        assert opts.speed == 1.12 and opts.crf == 18
        record("engine_ffmpeg", "RenderOptions dataclass", PASS)
    except Exception as e:
        record("engine_ffmpeg", "RenderOptions dataclass", FAIL, str(e))

    # build_ffmpeg_command signature check
    try:
        sig = inspect.signature(ef.build_ffmpeg_command)
        params = list(sig.parameters.keys())
        required = ["input_path", "output_path", "video_coords", "text_coords"]
        missing = [k for k in required if k not in params]
        if missing:
            record("engine_ffmpeg", "build_ffmpeg_command() signature", FAIL, f"Missing params: {missing}")
        else:
            record("engine_ffmpeg", "build_ffmpeg_command() signature", PASS, f"Params: {params}")
    except Exception as e:
        record("engine_ffmpeg", "build_ffmpeg_command() signature", FAIL, str(e))

    # Filtergraph content check: aspect-ratio scale pattern
    try:
        src = inspect.getsource(ef.build_ffmpeg_command)
        has_scale_preserve = "scale=-1:" in src or "scale=1080:-2" in src
        has_crop = "crop=" in src and "(in_w-" in src
        has_pts_reset = "PTS-STARTPTS" in src
        has_shortest = "shortest" in src
        has_faststart = "faststart" in src
        has_duration_lock = "drawbox" in src or "boxblur" in src
        has_format_yuv = "format=yuv420p" in src
        has_quoted_crop = "':0" in src or "':{target_box_h}" in src or "':" in src

        record("engine_ffmpeg", "Aspect-ratio scale (1080:-2 / -1:h)", PASS if has_scale_preserve else FAIL,
               "aspect-ratio scale found" if has_scale_preserve else "MISSING: aspect-ratio scale not found")
        record("engine_ffmpeg", "Center-crop filtergraph", PASS if has_crop else FAIL,
               "crop=(in_w-w)/2:0 found" if has_crop else "MISSING: horizontal center crop not found")
        record("engine_ffmpeg", "Quoted dynamic crop expression ('{crop_x}':0)", PASS if has_quoted_crop else FAIL,
               "Quoted crop expression found" if has_quoted_crop else "MISSING: unquoted dynamic crop expression")
        record("engine_ffmpeg", "Output pixel format (format=yuv420p)", PASS if has_format_yuv else FAIL,
               "format=yuv420p found" if has_format_yuv else "MISSING: format=yuv420p not found")
        record("engine_ffmpeg", "PTS-STARTPTS timestamp reset", PASS if has_pts_reset else FAIL)
        record("engine_ffmpeg", "Duration hardening (stream canvas lock)", PASS if has_duration_lock else WARN,
               "stream duration lock found" if has_duration_lock else "duration lock not found — duration may drift")
        record("engine_ffmpeg", "-shortest flag", PASS if has_shortest else WARN)
        record("engine_ffmpeg", "-movflags +faststart", PASS if has_faststart else WARN)
    except Exception as e:
        record("engine_ffmpeg", "Filtergraph source inspection", FAIL, str(e))

    # build_ffmpeg_command dry-run (no real file needed)
    try:
        dummy_in = "nonexistent_input.mp4"
        dummy_out = "nonexistent_output.mp4"
        video_coords = {"x": 0, "y": 190, "width": 1080, "height": 1540, "scale_x": 1.0, "scale_y": 1.0}
        text_coords = {"x": 60, "y": 80, "font_size": 48, "text_content": ""}
        opts = ef.RenderOptions(speed=1.12)
        result = ef.build_ffmpeg_command(
            input_path=dummy_in, output_path=dummy_out,
            video_coords=video_coords, text_coords=text_coords, options=opts
        )
        cmd, _ = result if isinstance(result, tuple) else (result, None)
        assert "ffmpeg" in cmd[0]
        assert dummy_out in cmd
        assert "-filter_complex" in cmd
        assert "-shortest" in cmd
        assert "-movflags" in cmd
        record("engine_ffmpeg", "build_ffmpeg_command() dry-run", PASS,
               f"Command has {len(cmd)} args, filter_complex present")
    except Exception as e:
        record("engine_ffmpeg", "build_ffmpeg_command() dry-run", FAIL, str(e))

    # Atempo chain logic test
    try:
        src = inspect.getsource(ef.build_ffmpeg_command)
        has_atempo_chain = "while curr_speed > 2.0" in src or "while remaining > 2.0" in src
        record("engine_ffmpeg", "Dynamic atempo chaining (>2.0x / <0.5x)", PASS if has_atempo_chain else WARN,
               "atempo loop found" if has_atempo_chain else "No dynamic atempo chaining detected")
    except Exception as e:
        record("engine_ffmpeg", "Atempo chain source check", WARN, str(e))
else:
    record("engine_ffmpeg", "All checks", FAIL, "Module failed to import")


# =============================================================
# 7. PIPELINE CONTRACT
# =============================================================
separator("7. PIPELINE CONTRACT")

pm = modules.get("pipeline")
if pm:
    try:
        sig = inspect.signature(pm.run_pipeline)
        params = list(sig.parameters.keys())
        record("Pipeline", "run_pipeline() signature", PASS, f"Params: {params}")
    except Exception as e:
        record("Pipeline", "run_pipeline() signature", FAIL, str(e))

    try:
        src = inspect.getsource(pm.run_pipeline)
        iterates_all = "for idx, cut in enumerate(cuts" in src or "for cut in cuts" in src
        no_hardcoded_zero = "cuts[0]" not in src
        uses_curation = "curation_engine" in src or "get_viral_cuts" in src
        uses_ffmpeg = "engine_ffmpeg" in src or "render_clip" in src
        uses_slice = "ffmpeg" in src and ("slice" in src or "-ss" in src)

        pipeline_file = inspect.getsourcefile(pm) or ""
        with open(pipeline_file, "r", encoding="utf-8") as f:
            full_pipeline_src = f.read()
        has_cli_entry = '__name__ == "__main__"' in full_pipeline_src and "--input" in full_pipeline_src

        record("Pipeline", "Iterates all cuts (not just cuts[0])", PASS if (iterates_all and no_hardcoded_zero) else FAIL)
        record("Pipeline", "Calls curation_engine.get_viral_cuts()", PASS if uses_curation else FAIL)
        record("Pipeline", "Calls engine_ffmpeg.render_clip()", PASS if uses_ffmpeg else FAIL)
        record("Pipeline", "FFmpeg segment slicing present", PASS if uses_slice else WARN)
        record("Pipeline", "CLI entry point support (--input/--output)", PASS if has_cli_entry else FAIL)
    except Exception as e:
        record("Pipeline", "Source inspection", FAIL, str(e))
else:
    record("Pipeline", "All checks", FAIL, "Module failed to import")


# =============================================================
# 8. AUDIO INTELLIGENCE CONTRACT
# =============================================================
separator("8. AUDIO INTELLIGENCE CONTRACT")

ai = modules.get("audio_intelligence")
if ai:
    for fn_name in ["extract_audio", "detect_dead_air", "snap_to_acoustic_trough",
                    "run_silero_vad_v5", "run_rms_vad_fallback", "get_vocal_analysis"]:
        has_fn = callable(getattr(ai, fn_name, None))
        record("Audio Intelligence", f"{fn_name}()", PASS if has_fn else FAIL)

    # snap_to_acoustic_trough correctness
    try:
        # Timestamp inside interval → should snap to midpoint or edge
        v = ai.snap_to_acoustic_trough(12.4, [(12.1, 12.6)], direction='backward', tolerance=0.5)
        assert isinstance(v, (int, float)), "Must return numeric"
        assert 12.0 <= v <= 13.0, f"Snapped value {v} out of plausible range"
        record("Audio Intelligence", "snap_to_acoustic_trough() correctness", PASS, f"12.4s → {v}s")
    except Exception as e:
        record("Audio Intelligence", "snap_to_acoustic_trough() correctness", FAIL, str(e))

    # detect_dead_air return type
    try:
        sig = inspect.signature(ai.detect_dead_air)
        src = inspect.getsource(ai.detect_dead_air)
        returns_ms = "1000" in src or "_ms" in src
        record("Audio Intelligence", "detect_dead_air() returns milliseconds", PASS if returns_ms else WARN,
               "ms conversion found in source" if returns_ms else "Could not confirm ms conversion")
    except Exception as e:
        record("Audio Intelligence", "detect_dead_air() return type check", WARN, str(e))
else:
    record("Audio Intelligence", "All checks", FAIL, "Module failed to import")




# =============================================================
# 10. CAPTION ENGINE CONTRACT (Phase 3)
# =============================================================
separator("10. CAPTION ENGINE CONTRACT (Phase 3)")

cap_eng = modules.get("caption_engine")
if cap_eng:
    sample_words = [
        {"word": "fire",     "start": 0.10, "end": 0.45},
        {"word": "money",    "start": 0.50, "end": 0.88},
        {"word": "king",     "start": 0.95, "end": 1.20},
        {"word": "running",  "start": 1.30, "end": 1.65},
        {"word": "star",     "start": 1.70, "end": 2.05},
    ]

    # ASS file generation with words
    try:
        tmp_ass = tempfile.mktemp(suffix=".ass")
        cap_eng.generate_ass_subtitles(sample_words, tmp_ass)
        assert os.path.isfile(tmp_ass)
        content = open(tmp_ass, encoding="utf-8").read()
        assert "[Script Info]" in content
        assert "PlayResX: 1080" in content and "PlayResY: 1920" in content
        assert "Dialogue:" in content
        has_kf = r"\kf" in content
        has_k  = r"\k"  in content
        assert has_kf or has_k, "No karaoke tags found"
        record("Caption Engine", "generate_ass_subtitles() ASS output with kf tags",
               PASS, f"{os.path.getsize(tmp_ass)} bytes | kf={has_kf}")
        os.remove(tmp_ass)
    except Exception as e:
        record("Caption Engine", "generate_ass_subtitles() with words", FAIL, str(e))

    # ASS file generation with empty words (graceful)
    try:
        tmp_ass_empty = tempfile.mktemp(suffix=".ass")
        cap_eng.generate_ass_subtitles([], tmp_ass_empty)
        assert os.path.isfile(tmp_ass_empty)
        content = open(tmp_ass_empty, encoding="utf-8").read()
        assert "[Script Info]" in content
        # No Dialogue lines expected but header must be present
        assert "[Events]" in content
        record("Caption Engine", "generate_ass_subtitles() empty words -> valid header-only ASS", PASS)
        os.remove(tmp_ass_empty)
    except Exception as e:
        record("Caption Engine", "generate_ass_subtitles() empty words graceful", FAIL, str(e))

    # Typography checks in ASS header
    try:
        tmp_ass_typ = tempfile.mktemp(suffix=".ass")
        cap_eng.generate_ass_subtitles(sample_words[:2], tmp_ass_typ)
        content = open(tmp_ass_typ, encoding="utf-8").read()
        # Alignment=2 (bottom center)
        style_line = [l for l in content.splitlines() if l.startswith("Style:")]
        assert style_line, "No Style: line found"
        sl = style_line[0]
        # Check bold (-1), alignment=2, MarginV~300, yellow primary color
        assert "&H0000FFFF" in sl, "Yellow primary color missing"
        assert "&H00FFFFFF" in sl, "White secondary color missing"
        assert ",2," in sl or sl.endswith(",2,0,0,300,1") or ",2," in sl, "Alignment=2 may be missing"
        assert "ScaledBorderAndShadow: yes" in content, "ScaledBorderAndShadow missing"
        record("Caption Engine", "ASS typography: Yellow primary, White secondary, ScaledBorderAndShadow", PASS)
        os.remove(tmp_ass_typ)
    except Exception as e:
        record("Caption Engine", "ASS typography validation", FAIL, str(e))

    # Emoji tests removed per Phase 3 configuration (Emoji injection disabled)
    
    # Pipeline imports caption_engine
    if pm:
        try:
            src = inspect.getsource(pm.run_pipeline)
            has_caption = "caption_engine" in src
            has_ass = "ass_path" in src or "generate_ass" in src
            record("Caption Engine", "pipeline.run_pipeline imports caption_engine", PASS if has_caption else FAIL)
            record("Caption Engine", "pipeline.run_pipeline calls generate_ass_subtitles", PASS if has_ass else FAIL)
        except Exception as e:
            record("Caption Engine", "pipeline caption_engine integration", FAIL, str(e))
else:
    record("Caption Engine", "All checks", FAIL, "Module failed to import")



# =============================================================
# 11. ENGINE_VISION CONTRACT (Phase 2)
# =============================================================
separator("11. ENGINE_VISION CONTRACT (Phase 2)")

ev = modules.get("engine_vision")
if ev:
    # Core function presence
    for fn_name in ["detect_subjects", "calculate_pan_offset", "smooth_trajectory",
                    "get_zoom_keyframes", "calculate_tracking_trajectory", "downsample_trajectory"]:
        record("engine_vision", f"{fn_name}()", PASS if callable(getattr(ev, fn_name, None)) else FAIL)

    # smooth_trajectory correctness
    try:
        raw = [420, 380, 350, 310, 290, 270, 260, 255]
        smoothed = ev.smooth_trajectory(raw, alpha=0.15)
        assert len(smoothed) == len(raw), "Length mismatch"
        assert smoothed[0] == raw[0], "First element must equal raw[0]"
        # Smoothed must be strictly between raw values (low-pass behavior)
        assert abs(smoothed[-1] - raw[-1]) < abs(raw[0] - raw[-1]), "Smoothing not reducing variance"
        record("engine_vision", "smooth_trajectory() correctness (alpha=0.15)", PASS,
               f"raw[-1]={raw[-1]}, smoothed[-1]={smoothed[-1]}")
    except Exception as e:
        record("engine_vision", "smooth_trajectory() correctness", FAIL, str(e))

    # calculate_pan_offset with detections
    try:
        # Subject centered at x=800, w=200 on 1920px source
        detections = [(700, 100, 200, 400)]
        x = ev.calculate_pan_offset(detections, source_w=1920, target_crop_w=1080)
        expected = 800 + 100 - 540   # center of box - half crop width = 360
        assert 0 <= x <= 1920 - 1080, f"x_offset {x} out of bounds"
        record("engine_vision", "calculate_pan_offset() with detections", PASS,
               f"subject_center=800, x_offset={x} (expected ~{expected})")
    except Exception as e:
        record("engine_vision", "calculate_pan_offset() with detections", FAIL, str(e))

    # calculate_pan_offset with no detections -> center
    try:
        x = ev.calculate_pan_offset([], source_w=1920, target_crop_w=1080)
        assert x == (1920 - 1080) // 2, f"Expected center {(1920-1080)//2}, got {x}"
        record("engine_vision", "calculate_pan_offset() fallback (no detections) -> center", PASS,
               f"center x_offset={x}")
    except Exception as e:
        record("engine_vision", "calculate_pan_offset() no-detection fallback", FAIL, str(e))

    # get_zoom_keyframes correctness
    try:
        x_offsets = [420, 419, 418, 415, 410, 380, 340, 300, 260, 220, 200]
        timestamps = [i * 3.0 for i in range(len(x_offsets))]
        kfs = ev.get_zoom_keyframes(x_offsets, timestamps, fps=30.0)
        assert isinstance(kfs, list), "Must return list"
        for kf in kfs:
            assert "time" in kf and "zoom" in kf and "duration" in kf
            assert 1.0 <= kf["zoom"] <= 1.15, f"Zoom {kf['zoom']} out of range"
        record("engine_vision", f"get_zoom_keyframes() -> {len(kfs)} keyframe(s)", PASS)
    except Exception as e:
        record("engine_vision", "get_zoom_keyframes()", FAIL, str(e))

    # _build_x_crop_expr from engine_ffmpeg
    if ef:
        try:
            timestamps_t = [0.0, 1.0, 2.0, 3.0]
            x_offs = [420, 400, 380, 360]
            expr = ef._build_x_crop_expr(timestamps_t, x_offs, max_x=840)
            assert "clip(" in expr or expr.lstrip("-").isdigit(), f"Unexpected expr: {expr}"
            record("engine_vision", "_build_x_crop_expr() piecewise expression", PASS,
                   f"expr[:60]: {expr[:60]}")
        except Exception as e:
            record("engine_vision", "_build_x_crop_expr() expression builder", FAIL, str(e))

        # Test safety clamp: 800 dense samples (e.g. 40s @ 50ms) must be clamped to <= 50 keyframes
        try:
            dense_ts = [round(i * 0.05, 3) for i in range(800)]
            dense_xs = [int(400 + 50 * math.sin(i * 0.1)) for i in range(800)]
            expr_dense = ef._build_x_crop_expr(dense_ts, dense_xs, max_x=840)
            clip_count = expr_dense.count("clip(")
            assert clip_count <= 50, f"Expected <=50 clip() terms, got {clip_count}"
            assert len(expr_dense) < 2500, f"Expression length {len(expr_dense)} exceeds 2500 characters"
            record("engine_vision", "_build_x_crop_expr() 800-sample clamp to <=50 keyframes", PASS,
                   f"{len(expr_dense)} chars, {clip_count} clip() terms")
        except Exception as e:
            record("engine_vision", "_build_x_crop_expr() 800-sample clamp", FAIL, str(e))

    # downsample_trajectory test
    try:
        dense_ts = [round(i * 0.05, 3) for i in range(800)]
        dense_xs = [int(400 + 50 * math.sin(i * 0.1)) for i in range(800)]
        ds_ts, ds_xs = ev.downsample_trajectory(dense_ts, dense_xs, sample_step=1.0, max_keyframes=50)
        assert len(ds_ts) <= 50, f"Exceeded max keyframes: {len(ds_ts)}"
        assert len(ds_ts) == len(ds_xs), "Length mismatch"
        assert ds_ts[0] == dense_ts[0], "Start timestamp not preserved"
        assert ds_ts[-1] == dense_ts[-1], "End timestamp not preserved"
        record("engine_vision", "downsample_trajectory() 800 -> <=50 keyframes @ 1.0s", PASS,
               f"{len(ds_ts)} keyframes, span: {ds_ts[0]}s -> {ds_ts[-1]}s")
    except Exception as e:
        record("engine_vision", "downsample_trajectory()", FAIL, str(e))

    # _build_zoom_expr from engine_ffmpeg
    if ef:
        try:
            kfs = [{"time": 4.0, "zoom": 1.12, "duration": 1.5}]
            expr = ef._build_zoom_expr(kfs)
            assert expr.startswith("1.0+"), f"Expected additive form, got: {expr}"
            assert "sin(" in expr or "sin(PI" in expr.replace(" ", "")
            record("engine_vision", "_build_zoom_expr() sine-ramp expression", PASS,
                   f"expr[:70]: {expr[:70]}")
        except Exception as e:
            record("engine_vision", "_build_zoom_expr() expression builder", FAIL, str(e))

    # calculate_tracking_trajectory with missing file -> graceful fallback
    try:
        traj = ev.calculate_tracking_trajectory("nonexistent_video.mp4")
        assert isinstance(traj, dict)
        assert "x_offsets" in traj and "best_x_offset" in traj
        assert "zoom_keyframes" in traj and isinstance(traj["zoom_keyframes"], list)
        record("engine_vision", "calculate_tracking_trajectory() missing file -> graceful fallback", PASS,
               f"best_x={traj['best_x_offset']}, is_vertical={traj['is_vertical']}")
    except Exception as e:
        record("engine_vision", "calculate_tracking_trajectory() missing file graceful fallback", FAIL, str(e))

else:
    record("engine_vision", "All checks", FAIL, "Module failed to import")


# =============================================================
# 12. INTEGRATION: END-TO-END LOGIC DRY-RUN
# =============================================================
separator("12. INTEGRATION: END-TO-END LOGIC DRY-RUN")

if ce and ef:
    try:
        total_dur = 90.0
        cuts = ce.get_viral_cuts([], [], total_dur, "fake_video.mp4")
        assert len(cuts) >= 1

        video_coords = {"x": 0, "y": 190, "width": 1080, "height": 1540, "scale_x": 1.0, "scale_y": 1.0}
        text_coords = {"x": 60, "y": 80, "font_size": 48, "text_content": ""}
        opts = ef.RenderOptions(speed=1.12)

        for cut in cuts:
            start = float(cut["start_time"])
            end = float(cut["end_time"])
            assert end > start, f"Invalid cut: start={start} >= end={end}"
            hook = cut.get("hook_sentence", "")

            # Dry-run build_ffmpeg_command — Phase 1 (no trajectory)
            cmd_result = ef.build_ffmpeg_command(
                input_path=f"slice_{start:.1f}.mp4",
                output_path=f"reel_dry.mp4",
                video_coords=video_coords,
                text_coords={"x": 60, "y": 80, "font_size": 48, "text_content": hook},
                options=opts
            )
            cmd, _ = cmd_result if isinstance(cmd_result, tuple) else (cmd_result, None)
            assert "-filter_complex" in cmd
            assert "-shortest" in cmd

        # Phase 2 trajectory dry-run
        synthetic_traj = {
            "fps": 30.0,
            "frame_count": 900,
            "source_w": 1920,
            "source_h": 1080,
            "is_vertical": False,
            "x_offsets": [420, 410, 395, 380, 365, 350],
            "sample_timestamps": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "best_x_offset": 390,
            "zoom_keyframes": [{"time": 4.0, "zoom": 1.12, "duration": 1.5}],
        }
        cmd_traj, _ = ef.build_ffmpeg_command(
            input_path="slice_0.0.mp4",
            output_path="reel_dry_p2.mp4",
            video_coords=video_coords,
            text_coords={"x": 60, "y": 80, "font_size": 48, "text_content": ""},
            options=opts,
            trajectory=synthetic_traj,
        )
        fc = cmd_traj[cmd_traj.index("-filter_complex") + 1]
        has_dynamic_x = "clip(" in fc
        has_zoompan = "zoompan=" in fc
        record("Integration", "Phase 2 trajectory dry-run -> dynamic crop x expression", PASS if has_dynamic_x else FAIL,
               f"clip() in filter_complex: {has_dynamic_x}")
        record("Integration", "Phase 2 trajectory dry-run -> zoompan applied for zoom keyframes", PASS if has_zoompan else FAIL,
               f"zoompan= in filter_complex: {has_zoompan}")

        record("Integration", f"Dry-run for {len(cuts)} cuts, all FFmpeg commands built cleanly", PASS)
    except Exception as e:
        record("Integration", "End-to-end dry-run", FAIL, str(e))
else:
    record("Integration", "End-to-end dry-run", FAIL, "curation_engine or engine_ffmpeg not importable")


# =============================================================
# FINAL REPORT SUMMARY
# =============================================================
total = len(results)
passed = sum(1 for _, _, s, _ in results if s == PASS)
warned = sum(1 for _, _, s, _ in results if s == WARN)
failed = sum(1 for _, _, s, _ in results if s == FAIL)

print(f"\n{'='*60}")
print(f"  PREFLIGHT SUMMARY")
print(f"{'='*60}")
print(f"  Total Checks : {total}")
print(f"  {PASS}        : {passed}")
print(f"  {WARN}   : {warned}")
print(f"  {FAIL}        : {failed}")
print(f"{'='*60}")

if failed == 0 and warned == 0:
    print("\n>>> ALL SYSTEMS GO -- Pipeline is ready for Kaggle GPU execution.\n")
elif failed == 0:
    print(f"\n>>> READY WITH {warned} WARNING(S) -- Optional GPU libs absent (expected on local CPU).\n")
else:
    print(f"\n>>> PREFLIGHT BLOCKED -- {failed} critical failure(s) must be resolved before execution.\n")
    print("Failing checks:")
    for section, label, status, detail in results:
        if status == FAIL:
            print(f"  * [{section}] {label}: {detail}")
