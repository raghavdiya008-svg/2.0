import os
import json
import subprocess
import glob

folder = r"C:\Users\dksha\Downloads\New folder 2"
files = glob.glob(os.path.join(folder, "*.mp4"))

print(f"Total clips found: {len(files)}\n")
for f in sorted(files):
    name = os.path.basename(f)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration,pix_fmt,sample_rate,channels",
        "-of", "json", f
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
        
        fmt_dur = float(fmt.get("duration", 0))
        v_dur = float(v_stream.get("duration", fmt_dur))
        a_dur = float(a_stream.get("duration", fmt_dur))
        diff = abs(v_dur - a_dur)
        w = v_stream.get("width", "N/A")
        h = v_stream.get("height", "N/A")
        v_codec = v_stream.get("codec_name", "N/A")
        a_codec = a_stream.get("codec_name", "N/A")
        fps = v_stream.get("r_frame_rate", "N/A")
        pix_fmt = v_stream.get("pix_fmt", "N/A")
        size_mb = int(fmt.get("size", 0)) / (1024*1024)
        print(f"=== {name} ===")
        print(f"  Size: {size_mb:.2f} MB | Total Duration: {fmt_dur:.3f}s")
        print(f"  Video: {w}x{h} ({v_codec}), FPS: {fps}, Dur: {v_dur:.3f}s, PixFmt: {pix_fmt}")
        print(f"  Audio: {a_codec} {a_stream.get('sample_rate')}Hz {a_stream.get('channels')}ch, Dur: {a_dur:.3f}s")
        print(f"  A/V Sync Diff: {diff:.4f}s\n")
