import os
import glob
import json
import subprocess
import cv2
import numpy as np

OUTPUT_DIR = r"C:\Users\dksha\Downloads\New folder 2"
INSPECT_DIR = os.path.join(OUTPUT_DIR, "audit_frames")
os.makedirs(INSPECT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")))

print(f"=== DETAILED AUDIT FOR {len(files)} CLIPS IN {OUTPUT_DIR} ===\n")

for idx, fpath in enumerate(files, 1):
    fname = os.path.basename(fpath)
    cap = cv2.VideoCapture(fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    bitrate_kbps = (size_mb * 8 * 1024) / duration if duration > 0 else 0
    
    print(f"[{idx}/10] {fname}")
    print(f"  - Resolution: {width}x{height} (Target: 1080x1920)")
    print(f"  - FPS: {fps:.2f} (Target: 30.00 CFR)")
    print(f"  - Duration: {duration:.2f}s ({total_frames} frames)")
    print(f"  - File Size: {size_mb:.2f} MB (~{bitrate_kbps:.0f} kbps)")

    # Extract 4 sample frames across the video
    timestamps = [0.15 * duration, 0.40 * duration, 0.65 * duration, 0.85 * duration]
    clip_prefix = f"clip_{idx:02d}"
    
    for t_idx, t in enumerate(timestamps, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            # Check for horizontal split line around height/2 (960)
            # A split frame typically has a thin black/border line at y=960 or distinct upper and lower frames
            sample_name = f"{clip_prefix}_t{int(t)}s_sample{t_idx}.jpg"
            sample_path = os.path.join(INSPECT_DIR, sample_name)
            cv2.imwrite(sample_path, frame)
            
    cap.release()

print("\nDONE: Frame extraction complete. Saved sample snapshots to:", INSPECT_DIR)
