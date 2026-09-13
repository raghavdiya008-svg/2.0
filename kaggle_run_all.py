#!/usr/bin/env python3
"""
kaggle_run_all.py
-----------------
Standalone Kaggle execution runner script.
Performs GPU checks, installs dependencies if missing in the Kaggle context,
and processes all videos in inputs/ through the end-to-end auto-clipping pipeline.
"""

import os
import sys
import subprocess
import time

def setup_kaggle_environment():
    """Installs required Kaggle GPU dependencies if not already installed."""
    print("=" * 80)
    print("KAGGLE RUNNER: INITIALIZING ENVIRONMENT")
    print("=" * 80)
    
    # Check if running in Kaggle
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    print(f"Detecting Kaggle Environment: {is_kaggle}")
    
    dependencies = [
        "whisperx",
        "faster-whisper",
        "ultralytics",
        "mediapipe",
        "soundfile",
        "python-dotenv",
        "pyannote.audio"
    ]
    
    if is_kaggle:
        print("\nUpdating system packages (ffmpeg, libass)...")
        try:
            subprocess.run(["apt-get", "update", "-y", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg", "libass-dev"], check=True)
        except Exception as e:
            print(f"Failed to update system packages: {e}")
            
        print("\nInstalling Ollama (if missing)...")
        try:
            subprocess.run(["curl -fsSL https://ollama.ai/install.sh | sh"], shell=True, check=True)
        except Exception as e:
            print(f"Failed to install Ollama: {e}")

        print("\nInstalling Kaggle GPU dependencies...")
        for pkg in dependencies:
            try:
                print(f"Installing {pkg}...")
                subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--no-warn-script-location", "-q"], check=True)
            except Exception as e:
                print(f"Failed to install {pkg}: {e}")
    else:
        print("\nLocal system run detected. Skipping heavy Kaggle package installations.")
        print("Using local mock / CPU fallbacks in source code.")

    # GPU Check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"\nCUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Device Count: {torch.cuda.device_count()}")
        else:
            print("WARNING: No GPU detected. Kaggle accelerator must be set to GPU T4 x2 or P100.")
    except ImportError:
        print("\nPyTorch not available in local environment.")
        print("WARNING: No GPU detected. Kaggle accelerator must be set to GPU T4 x2 or P100.")

    print("\nEnsuring YOLO face model is downloaded...")
    if not os.path.exists("yolo11n.pt"):
        print("Downloading yolo11n.pt model...")
        try:
            from ultralytics import YOLO
            YOLO("yolo11n.pt")
        except Exception as e:
            print(f"Failed to auto-download yolo11n.pt: {e}")

def main():
    setup_kaggle_environment()
    
    # Add src to python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")
    sys.path.insert(0, src_dir)
    
    try:
        import pipeline
    except ImportError as e:
        print(f"\nError: Could not import pipeline modules from {src_dir}: {e}")
        sys.exit(1)
        
    inputs_dir = os.path.join(project_root, "inputs")
    outputs_dir = os.path.join(project_root, "outputs")
    temp_dir = os.path.join(project_root, "temp")
    emojis_dir = os.path.join(project_root, "assets", "emojis")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("RUNNING END-TO-END PIPELINE ON INPUT CLIPS")
    print("=" * 80)
    
    # Discover inputs
    import glob
    video_files = glob.glob(os.path.join(inputs_dir, "*.mp4")) + glob.glob(os.path.join(inputs_dir, "*.mov"))
    
    if not video_files:
        print(f"Warning: No video files found in inputs folder: {inputs_dir}")
        print("Please place .mp4 or .mov clips into inputs/ to process.")
        sys.exit(0)
        
    print(f"Discovered {len(video_files)} video file(s) for processing.")
    
    all_reels = []
    
    for idx, video_path in enumerate(video_files, start=1):
        filename = os.path.basename(video_path)
        print(f"\n[{idx}/{len(video_files)}] Processing file: {filename}")
        
        t0 = time.time()
        try:
            reels = pipeline.run_pipeline(
                video_path=video_path,
                output_dir=outputs_dir,
                logo_path=None,
                logo_coords=None,
                bg_color="white",
                emojis_dir=emojis_dir,
                temp_dir=temp_dir
            )
            elapsed = time.time() - t0
            print(f"File {filename} processed in {elapsed:.2f}s. Generated {len(reels)} reel(s).")
            
            for r in reels:
                all_reels.append({
                    "input": filename,
                    "index": r["reel_index"],
                    "start": r["start"],
                    "end": r["end"],
                    "score": r["virality_score"],
                    "reason": r["reason"],
                    "path": r["path"]
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "=" * 120)
    print("KAGGLE EXECUTION SUMMARY")
    print("=" * 120)
    
    headers = ["Input Clip", "Reel #", "Start (s)", "End (s)", "Virality Score", "Output Reel Path"]
    row_fmt = "{:<25} | {:<6} | {:<9} | {:<7} | {:<14} | {:<45}"
    print(row_fmt.format(*headers))
    print("-" * 120)
    
    for r in all_reels:
        print(row_fmt.format(
            r["input"][:23],
            r["index"],
            f"{r['start']:.2f}",
            f"{r['end']:.2f}",
            f"{r['score']}/100",
            os.path.basename(r["path"])[:43]
        ))
    print("=" * 120)
    print("\nKaggle execution completed successfully! Generated video files saved to outputs/.")


if __name__ == "__main__":
    main()
