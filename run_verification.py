import os
import sys
import shutil
import uuid
import subprocess
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import src.pipeline as pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== RUNNING VERIFICATION END-TO-END ===")
    
    if os.path.exists("outputs2"):
        shutil.rmtree("outputs2")
    os.makedirs("outputs2")
        
    print("\n--- Testing Video 1 (simulate gaming clip) ---")
    results1 = pipeline.run_pipeline(input_video_path="inputs/video1.mp4", output_dir="outputs2/v1")
    
    print("\n--- Testing Video 2 (simulate talking head clip) ---")
    results2 = pipeline.run_pipeline(input_video_path="inputs/video2.mp4", output_dir="outputs2/v2")
    
    print("\n=== VERIFICATION CHECKS ===")
    print("1. Did temp folder survive?", os.path.exists("temp"))
    
    print("2. Extracting screenshots from outputs...")
    for i, res in enumerate(results1 + results2):
        path = res["path"]
        if os.path.exists(path):
            subprocess.run(["ffmpeg", "-y", "-i", path, "-vframes", "1", f"outputs2/screenshot_{i}_start.jpg"], capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-i", path, "-ss", "00:00:05", "-vframes", "1", f"outputs2/screenshot_{i}_mid.jpg"], capture_output=True)
            print(f"Extracted screenshots for {path}")

