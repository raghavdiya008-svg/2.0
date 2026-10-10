import os
import glob
import subprocess
import json

folder = r"C:\Users\dksha\Downloads\New folder 2"
files = sorted(glob.glob(os.path.join(folder, "*.mp4")))

print("=" * 80)
print("DETAILED CLIP ANALYSIS REPORT")
print("=" * 80)

for idx, f in enumerate(files, 1):
    name = os.path.basename(f)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,duration,pix_fmt,sample_rate,channels",
        "-of", "json", f
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(res.stdout) if res.returncode == 0 else {}
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    
    fmt_dur = float(fmt.get("duration", 0))
    v_dur = float(v.get("duration", fmt_dur))
    a_dur = float(a.get("duration", fmt_dur))
    size_mb = int(fmt.get("size", 0)) / (1024 * 1024)
    v_bitrate_kbps = int(v.get("bit_rate", fmt.get("bit_rate", 0))) / 1000
    
    print(f"\nClip #{idx}: {name}")
    print(f"  • Duration: {fmt_dur:.3f}s (Video: {v_dur:.3f}s, Audio: {a_dur:.3f}s, Sync Drift: {abs(v_dur-a_dur)*1000:.1f}ms)")
    print(f"  • Size: {size_mb:.2f} MB | Bitrate: {v_bitrate_kbps:.0f} kbps")
    print(f"  • Stream: {v.get('width')}x{v.get('height')} @ {v.get('r_frame_rate')} fps ({v.get('codec_name')}/{v.get('pix_fmt')})")
    print(f"  • Audio: {a.get('codec_name')} @ {a.get('sample_rate')} Hz, {a.get('channels')} channels")
