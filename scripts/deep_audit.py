import os
import glob
import json
import subprocess
import cv2

OUTPUT_DIR = r"C:\Users\dksha\Downloads\New folder 2"

files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")))

print("=" * 80)
print(f"DEEP AUDIT OF {len(files)} CLIPS IN: {OUTPUT_DIR}")
print("=" * 80)

for idx, fpath in enumerate(files, 1):
    fname = os.path.basename(fpath)
    cap = cv2.VideoCapture(fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    bitrate_mbps = (size_mb * 8) / duration if duration > 0 else 0
    cap.release()

    print(f"\n[Clip {idx:02d}] {fname}")
    print(f"  * Resolution : {w}x{h} ({'PASS 1080x1920' if (w==1080 and h==1920) else 'FAIL'})")
    print(f"  * FPS        : {fps:.2f} ({'PASS 30.00 CFR' if abs(fps-30.0)<0.01 else 'NON-STANDARD'})")
    print(f"  * Duration   : {duration:.2f}s ({total_frames} frames)")
    print(f"  * File Size  : {size_mb:.2f} MB ({bitrate_mbps:.2f} Mbps)")

print("\n" + "=" * 80)
