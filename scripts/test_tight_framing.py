import cv2
import subprocess
import os

video_path = r"inputs\I Asked An AI Billionaire How To Start A Business (10 Steps).mp4"
out_dir = r"temp\framing_test"
os.makedirs(out_dir, exist_ok=True)

# Test tighter split-screen zoom:
# Let's crop Speaker A and Speaker B with tighter 9:8 framing:
# Speaker A: center ~ (640, 442) -> box (280, 200, 720, 640)
# Speaker B: center ~ (1322, 472) -> box (962, 230, 720, 640)

crop_a = "crop=720:640:280:200,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960"
crop_b = "crop=720:640:962:230,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960"

filter_str = f"[0:v]split=2[v1][v2];[v1]{crop_a}[top];[v2]{crop_b}[bot];[top][bot]vstack=inputs=2,drawbox=x=0:y=958:w=1080:h=4:color=black@0.8:t=fill"

out_img = os.path.join(out_dir, "tight_split_preview.jpg")
cmd = [
    "ffmpeg", "-y", "-ss", "15", "-i", video_path,
    "-filter_complex", filter_str,
    "-vframes", "1", "-q:v", "2", out_img
]
subprocess.run(cmd, capture_output=True)
print("Preview generated at:", out_img)
