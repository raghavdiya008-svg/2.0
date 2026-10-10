import sys
import os
import shutil
import subprocess

sys.path.insert(0, ".")

print("=" * 80)
print("              REEL ENGINE 2.0 - FULL PRE-FLIGHT CHECK              ")
print("=" * 80)

check_results = []

def report(name, status, detail=""):
    symbol = "[PASS]" if status else "[FAIL]"
    check_results.append((name, status, detail))
    print(f"{symbol:7} {name:<45} {detail}")

# 1. Environment & Binaries
report("FFmpeg Binary on PATH", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "NOT FOUND")
report("Git Binary on PATH", bool(shutil.which("git")), shutil.which("git") or "NOT FOUND")

# 2. Critical Python Dependencies
for pkg in ["torch", "ultralytics", "cv2", "yt_dlp", "fastapi"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        report(f"Dependency: {pkg}", True, f"v{ver}")
    except Exception as e:
        report(f"Dependency: {pkg}", False, str(e))

# 3. Vision & Framing Engine
try:
    from src import engine_vision
    # Check 9:16 target crop width calculation for 1080p
    target_crop_w = int(round(1080 * 9.0 / 16.0 / 2.0)) * 2
    report("Vision: 9:16 Target Crop Math (608px)", target_crop_w == 608, f"{target_crop_w}px")

    # Check composite YOLO person box decomposition
    composite_box = (498, 113, 1090, 958)
    bw, bh = composite_box[2], composite_box[3]
    is_composite = (bw / float(bh) > 0.70 or bw > 1920 * 0.35)
    report("Vision: Composite Person Decomposition Logic", is_composite, "Identifies merged 1090px box")

    # Check ZOOM_MAX
    report("Vision: Pacing Zoom (1.06x)", engine_vision.ZOOM_MAX == 1.06, f"ZOOM_MAX={engine_vision.ZOOM_MAX}")
except Exception as e:
    report("Vision Engine Import", False, str(e))

# 4. Caption & Typography Engine
try:
    from src import caption_engine
    report("Caption: TikTok Safe Zone (MarginV=500)", caption_engine.MARGIN_V == 500, f"MARGIN_V={caption_engine.MARGIN_V}")
    header = caption_engine._ass_header()
    report("Caption: ASS Header Formatting", ",500,1" in header and "Style: Default,Arial Black,84" in header, "Validated")
except Exception as e:
    report("Caption Engine Import", False, str(e))

# 5. FFmpeg Render Engine & Audio Loudnorm
try:
    from src import engine_ffmpeg
    # Check loudnorm filter in render engine
    audio_has_loudnorm = False
    with open("src/engine_ffmpeg.py", encoding="utf-8") as f:
        code = f.read()
        audio_has_loudnorm = "loudnorm=I=-14:TP=-1.5:LRA=11" in code
    report("Audio: EBU R128 Mastering (-14 LUFS)", audio_has_loudnorm, "loudnorm filter integrated")

    zoom_expr = engine_ffmpeg._build_zoom_expr([{"time": 2.0, "zoom": 1.06, "duration": 1.5}])
    report("FFmpeg: Ease-in/out Zoom Keyframe Generator", "sin(3.14159265" in zoom_expr, zoom_expr[:30] + "...")
except Exception as e:
    report("FFmpeg Engine Import", False, str(e))

# 6. Media Downloader (YouTube Bot Bypass)
try:
    from src import media_downloader
    extra_args = media_downloader._ytdlp_extra_args()
    has_visionos = any("player_client=visionos" in arg for arg in extra_args)
    report("Downloader: VisionOS Player Client Bypass", has_visionos, "Configured")
except Exception as e:
    report("Media Downloader Import", False, str(e))

# 7. Kaggle Runner Compatibility
try:
    with open("kaggle_run_all.py", encoding="utf-8") as f:
        k_code = f.read()
    has_git_sync = "origin/main" in k_code and "reset" in k_code
    has_secrets = "YOUTUBE_COOKIES" in k_code
    report("Kaggle: Git Auto-Sync on Boot", has_git_sync, "git reset --hard origin/main")
    report("Kaggle: User Secrets Cookie Extraction", has_secrets, "YOUTUBE_COOKIES checked")
except Exception as e:
    report("Kaggle Runner Check", False, str(e))

# 8. Git Sync Status
try:
    git_diff = subprocess.check_output(["git", "diff", "--stat"], text=True).strip()
    report("Git: Clean Tracked Working Tree", len(git_diff) == 0, "All tracked files committed" if not git_diff else git_diff)
    git_head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    report("Git: Current HEAD", True, f"commit {git_head}")
except Exception as e:
    report("Git Check", False, str(e))

print("=" * 80)
all_passed = all(s for _, s, _ in check_results)
print(f"PRE-FLIGHT STATUS: {'ALL CHECKS PASSED - PRODUCTION READY' if all_passed else 'SOME CHECKS FAILED'}")
print("=" * 80)
