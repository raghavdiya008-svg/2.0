import os
import sys
import subprocess
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return e.stderr.strip()

def check_gpu():
    print("--- GPU Check ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"✅ VRAM: {vram:.2f} GB")
        if vram < 14.0:
            print("⚠️ WARNING: VRAM is less than 15GB. You may experience OOM errors on Kaggle T4.")
        else:
            print("✅ VRAM sufficient for pipeline (>=14GB).")
    else:
        print("❌ No GPU detected. Kaggle accelerator must be set to GPU T4x2 or P100.")
        print("   -> In Kaggle: Notebook Options > Accelerator > GPU T4x2")

def check_ffmpeg():
    print("\n--- FFmpeg Check ---")
    try:
        ffmpeg_version = run_cmd("ffmpeg -version").split('\n')[0]
        if "ffmpeg version" in ffmpeg_version:
            print(f"✅ FFmpeg installed: {ffmpeg_version}")
        else:
            print("❌ FFmpeg not found or not in PATH.")
        
        # Check NVENC
        encoders = run_cmd("ffmpeg -encoders")
        if "h264_nvenc" in encoders or "hevc_nvenc" in encoders:
            print("✅ NVENC hardware encoders found.")
        else:
            print("⚠️ WARNING: NVENC encoders not found. Hardware encoding may fail.")
            print("   -> Fallback to libx264 will be used, but rendering will be slower.")
    except Exception as e:
        print("❌ FFmpeg check failed. Ensure ffmpeg is installed.")
        print("   -> In Kaggle: !apt-get update && apt-get install -y ffmpeg")

def check_dependencies(auto_install=True):
    print("\n--- Python Dependencies Check ---")
    is_cloud = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/content")
    deps = {
        "torch": ("PyTorch", "torch"),
        "whisperx": ("WhisperX", "whisperx"),
        "scenedetect": ("PySceneDetect", "scenedetect[opencv]"),
        "insightface": ("InsightFace", "insightface onnxruntime-gpu"),
        "cv2": ("OpenCV", "opencv-python-headless"),
        "fastapi": ("FastAPI", "fastapi"),
        "pydantic": ("Pydantic", "pydantic")
    }
    for module, (name, pip_pkg) in deps.items():
        try:
            __import__(module)
            print(f"✅ {name} ({module}) ready.")
        except ImportError:
            if is_cloud and auto_install:
                print(f"🔄 Auto-installing missing dependency: {pip_pkg}...")
                subprocess.run(f"{sys.executable} -m pip install {pip_pkg} -q", shell=True, check=False)
                try:
                    __import__(module)
                    print(f"✅ {name} ({module}) installed and ready.")
                except ImportError:
                    print(f"❌ Failed to load {name} after install attempt.")
            else:
                print(f"❌ Missing dependency: {name}. Run: !pip install {pip_pkg}")

def check_fonts():
    print("\n--- Fonts Check ---")
    os.makedirs("fonts", exist_ok=True)
    font_path = os.path.join("fonts", "Anton.ttf")
    bold_path = os.path.join("fonts", "TheBoldFont.ttf")
    if not os.path.isfile(font_path) or os.path.getsize(font_path) < 1000:
        print("🔄 Auto-downloading viral subtitle font (Anton.ttf)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
                font_path
            )
            import shutil
            shutil.copy(font_path, bold_path)
            print("✅ Viral font installed: fonts/Anton.ttf & fonts/TheBoldFont.ttf")
        except Exception as e:
            print(f"⚠️ Could not download font automatically: {e}")
    else:
        print("✅ Viral subtitle font ready: fonts/Anton.ttf")

def check_dirs():
    print("\n--- Directory Structure Check ---")
    expected_dirs = ["inputs", "outputs", "temp", "models", "fonts"]
    for d in expected_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Directory ready: {d}/")

def run_all():
    print("🚀 Starting Kaggle Pre-Flight Checks for Autonomous Video Pipeline...\n")
    check_gpu()
    check_ffmpeg()
    check_dependencies()
    check_fonts()
    check_dirs()
    print("\n🏁 Pre-Flight Checks Completed.")
    print("If all checks are green (✅), you are ready to start the pipeline in Kaggle!")

if __name__ == "__main__":
    run_all()
