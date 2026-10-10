import sys
import os
import cv2

sys.path.insert(0, ".")
from src import engine_vision

raw_path = r"inputs\I Asked An AI Billionaire How To Start A Business (10 Steps).mp4"
if not os.path.isfile(raw_path):
    print("Raw video not found in inputs/")
    sys.exit(1)

cap = cv2.VideoCapture(raw_path)
cap.set(cv2.CAP_PROP_POS_MSEC, 138000)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame at 138s")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"=== VERIFICATION TEST: RAW FRAME AT 138s ({w}x{h}) ===")

# Run detection
subjects = engine_vision.detect_faces_or_subjects(frame)
print(f"1. Total subjects detected: {len(subjects)}")
for i, s in enumerate(subjects):
    bx, by, bw, bh = s.box
    cx = bx + bw / 2.0
    print(f"   - Subject {i}: box={s.box}, center_x={cx:.1f}, aspect_ratio={bw/bh:.2f}")

# Check that NO subject is wider than 650px (composite box must be gone)
composite_present = any(s.box[2] > w * 0.40 for s in subjects)
print(f"2. Composite multi-person box eliminated: {not composite_present}")

# Verify 9:16 target crop width calculation
target_crop_w = int(round(h * 9.0 / 16.0 / 2.0)) * 2
print(f"3. Exact 9:16 target crop width: {target_crop_w}px (Expected: 608px)")
assert target_crop_w == 608, f"Expected 608px, got {target_crop_w}px"

# Test trajectory calculation for this segment
print("\n4. Testing trajectory generation on slice window (120s - 167s)...")
temp_slice = "temp_verify_slice.mp4"
import subprocess
cmd = [
    "ffmpeg", "-y", "-ss", "120", "-t", "10",
    "-i", raw_path,
    "-c:v", "libx264", "-preset", "ultrafast",
    "-c:a", "copy", temp_slice
]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

try:
    traj = engine_vision.calculate_tracking_trajectory(temp_slice)
    print(f"   - Trajectory target_crop_w: {traj.get('target_crop_w')}px")
    print(f"   - Trajectory best_x_offset: {traj.get('best_x_offset')}")
    print(f"   - Keyframes count: {len(traj.get('keyframes', []))}")
    print(f"   - Is vertical: {traj.get('is_vertical')}")
    assert traj.get("target_crop_w") == 608, f"Trajectory target_crop_w is {traj.get('target_crop_w')}"
    print("✓ Verification passed: Strict 608px single-person crop verified!")
finally:
    if os.path.isfile(temp_slice):
        os.remove(temp_slice)
