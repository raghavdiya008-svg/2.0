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

    # ── Pull latest code & bust stale .pyc cache ─────────────────────────────
    print("\nPulling latest code from git...")
    try:
        subprocess.run(["git", "pull", "--ff-only"], check=False)
    except Exception as e:
        print(f"git pull skipped (non-fatal): {e}")

    print("Clearing stale Python bytecode cache...")
    try:
        import glob as _glob
        for pyc in _glob.glob("**/__pycache__/*.pyc", recursive=True):
            try:
                os.remove(pyc)
            except OSError:
                pass
        for pycache in _glob.glob("**/__pycache__", recursive=True):
            try:
                os.rmdir(pycache)
            except OSError:
                pass
    except Exception as e:
        print(f"Cache clear skipped (non-fatal): {e}")
    # ─────────────────────────────────────────────────────────────────────────
    
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
        print("\nUpdating system packages (ffmpeg, libass, zstd, nodejs, npm)...")
        try:
            subprocess.run(["apt-get", "update", "-y", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg", "libass-dev", "zstd", "nodejs", "npm"], check=True)
        except Exception as e:
            print(f"Failed to update system packages: {e}")
            
        # ── Ollama Local Runtime Setup (Only if not using Gemini workflow or if explicitly requested) ──
        skip_ollama = os.environ.get("SKIP_OLLAMA", "").lower() in ["1", "true", "yes"] or "--no-ollama" in sys.argv
        if not skip_ollama:
            print("\nEnsuring local Ollama runtime is available on Kaggle/Colab GPU...")
            try:
                subprocess.run(["curl -fsSL https://ollama.com/install.sh | sh"], shell=True, check=True)
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3)
                default_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
                print(f"Pre-pulling Ollama model '{default_model}'...")
                subprocess.run(["ollama", "pull", default_model], check=False)
            except Exception as e:
                print(f"Ollama local runtime setup note: {e}")
        else:
            print("\nSkipping Ollama installation (Gemini / External workflow active).")

        print("\nChecking Kaggle GPU dependencies...")
        import_map = {
            "whisperx": "whisperx",
            "faster-whisper": "faster_whisper",
            "ultralytics": "ultralytics",
            "mediapipe": "mediapipe",
            "soundfile": "soundfile",
            "python-dotenv": "dotenv",
            "pyannote.audio": "pyannote.audio"
        }
        for pkg in dependencies:
            imp_name = import_map.get(pkg, pkg)
            try:
                __import__(imp_name)
                print(f"  ✓ {pkg} already installed.")
            except ImportError:
                try:
                    print(f"  ⬇ Installing {pkg}...")
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
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Runner for 2.0 Autonomous Video Pipeline")
    parser.add_argument("--ui", "--serve", action="store_true", help="Launch the interactive Web UI Studio with Cloudflare tunnel")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Web Studio")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable public Cloudflare tunnel")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama download (use Gemini timestamps workflow)")
    args, _ = parser.parse_known_args()

    setup_kaggle_environment()
    
    # Add src to python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")
    sys.path.insert(0, src_dir)
    
    if args.ui:
        print("\n" + "=" * 80)
        print("LAUNCHING WEB STUDIO WITH CLOUDFLARE TUNNEL")
        print("=" * 80)
        import server
        import uvicorn
        import tunnel
        import atexit

        port = args.port
        should_tunnel = not args.no_tunnel
        public_url = None
        tunnel_proc = None

        if should_tunnel:
            print("[server] Launching temporary Cloudflare tunnel for external access...")
            public_url, tunnel_proc = tunnel.start_cloudflare_tunnel(port=port)
            if tunnel_proc:
                atexit.register(lambda: tunnel_proc.terminate() if tunnel_proc else None)

        print(f"\n=======================================================")
        print(f" 🚀 2.0 Autonomous Video Studio is live!")
        print(f" 👉 Local:  http://127.0.0.1:{port}")
        if public_url:
            print(f" 🌐 Public: {public_url}")
        print(f"=======================================================\n")
        uvicorn.run(server.app, host="0.0.0.0", port=port)
        return

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
        print("Please place .mp4 or .mov clips into inputs/ to process, or run with --ui to launch the web studio.")
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
