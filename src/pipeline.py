"""
src/pipeline.py
---------------
Phase 1 Production Pipeline:
  Stage 1 — Audio Extraction + WhisperX ASR  (evict after)
  Stage 2 — Speaker Diarization / pyannote    (evict after)
  Stage 3 — LLM Curation / Ollama             (evict after)
  Stage 4 — Per-clip render loop              (VRAM guard per clip)

Each stage explicitly logs torch.cuda.memory_allocated() before and after
eviction to satisfy the Kaggle sequential-VRAM constraint.
"""

import gc
import hashlib
import json
import os
import sys
import shutil
import subprocess
import uuid
import tempfile
from typing import Any, Dict, List, Optional, Tuple
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def project_timestamp(orig_t: float, intervals: list) -> float:
    """Projects original timestamp to pruned video timeline."""
    if not intervals:
        return orig_t
    new_t = 0.0
    for start, end in intervals:
        if orig_t <= start:
            break
        if orig_t < end:
            new_t += orig_t - start
            break
        new_t += end - start
    return new_t

import curation_engine
import engine_ffmpeg
import engine_vision
import caption_engine
import engine_vad
import engine_watermark
from engine_watermark import detect_blurred_logos, WatermarkRejectionError

try:
    from licensing import verify_lease, LicenseError
except ImportError:
    try:
        from src.licensing import verify_lease, LicenseError
    except ImportError:
        verify_lease = None
        LicenseError = Exception

try:
    from task_queue import vram_guard, cleanup_vram
except ImportError:
    try:
        from src.task_queue import vram_guard, cleanup_vram
    except ImportError:
        import contextlib
        @contextlib.contextmanager
        def vram_guard(label=""):
            yield
        def cleanup_vram():
            return {}

_EMOJIS_DIR = os.path.join(_PROJECT_ROOT, "assets", "emojis")

GLOBAL_AUDIO_CACHE_PATH = (
    "/kaggle/working/global_audio_cache.json"
    if os.path.exists("/kaggle/working")
    else os.path.join(_PROJECT_ROOT, "global_audio_cache.json")
)


def _get_audio_cache_key(input_path: str, start: float = 0.0, duration: Optional[float] = None) -> str:
    """
    Scopes the audio cache key using a SHA256 hash of the input file path, file size,
    and target slice offsets: hashlib.sha256(f"{input_path}:{file_size}:{start}:{duration}".encode()).hexdigest()[:16]
    """
    file_size = os.path.getsize(input_path) if os.path.isfile(input_path) else 0
    cache_str = f"{input_path}:{file_size}:{start}:{duration}"
    return hashlib.sha256(cache_str.encode("utf-8")).hexdigest()[:16]


def compute_audio_stream_md5(wav_path: str) -> str:
    """Computes an MD5 hex digest from the extracted audio stream file."""
    if not os.path.isfile(wav_path):
        return ""
    hasher = hashlib.md5()
    with open(wav_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_audio_cache(cache_key: str) -> Optional[dict]:
    """Loads cached audio metadata for the given scoped cache key."""
    if not os.path.isfile(GLOBAL_AUDIO_CACHE_PATH):
        return None
    try:
        with open(GLOBAL_AUDIO_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
            return cache.get(cache_key)
    except Exception as e:
        print(f"[pipeline] Warning: Failed to load audio cache: {e}")
        return None


def _save_audio_cache(cache_key: str, data: dict) -> None:
    """Saves scoped audio metadata to the global cache file."""
    try:
        cache_dir = os.path.dirname(os.path.abspath(GLOBAL_AUDIO_CACHE_PATH))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cache = {}
        if os.path.isfile(GLOBAL_AUDIO_CACHE_PATH):
            try:
                with open(GLOBAL_AUDIO_CACHE_PATH, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        cache[cache_key] = data
        temp_cache_file = os.path.join(cache_dir, f"tmp_cache_{uuid.uuid4().hex[:8]}.json")
        with open(temp_cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_cache_file, GLOBAL_AUDIO_CACHE_PATH)
        print(f"[pipeline] Audio cache saved for key [{cache_key}].")
    except Exception as e:
        print(f"[pipeline] Warning: Failed to save audio cache: {e}")


def _log_vram(label: str) -> None:
    """Logs torch CUDA memory_allocated safely on CPU machines."""
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[VRAM] {label}: {mb:.1f} MB allocated.")
        else:
            print(f"[VRAM] {label}: CPU-only (no GPU).")
    except Exception:
        pass


def _evict_vram(label: str = "") -> None:
    """Forces GC and GPU cache flush. Logs before/after."""
    _log_vram(f"before eviction ({label})")
    try:
        if hasattr(engine_vision, "purge_yolo_model"):
            engine_vision.purge_yolo_model()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _log_vram(f"after eviction ({label})")


class ReelPath(str):
    """Path string with dictionary metadata for universal caller compatibility."""
    def __new__(cls, path: str, metadata: dict | None = None):
        instance = super().__new__(cls, path)
        instance._metadata = metadata or {}
        return instance

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._metadata.get(key, "")
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self._metadata.get(key, default)


def run_pipeline(
    input_video_path: str = "",
    output_dir: str = "outputs",
    words: list | None = None,
    start: float = 0.0,
    duration: float | None = None,
    **kwargs,
) -> list:
    """
    End-to-end pipeline with real AI curation.

    Phase 1: Global Audio
    Phase 2: Global Curation
    Phase 3: Global Vision & Tracking (YOLO)
    Phase 4: Batch Render Engine
    """
    input_video_path = input_video_path or kwargs.get("video_path", "")
    base_temp_dir    = kwargs.get("temp_dir",     "temp")
    temp_dir         = os.path.join(base_temp_dir, f"run_{uuid.uuid4().hex[:8]}")
    logo_path    = kwargs.get("logo_path")
    logo_coords  = kwargs.get("logo_coords")
    bg_color     = kwargs.get("bg_color",     "white")
    hf_token     = kwargs.get("hf_token",     os.environ.get("HF_TOKEN", ""))
    silence_gaps = kwargs.get("silence_gaps") or []

    start_offset = max(0.0, float(kwargs.get("start", start)))
    raw_duration = kwargs.get("duration", duration)
    duration_limit = float(raw_duration) if raw_duration is not None and float(raw_duration) > 0 else None

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # ── License gate ────────────────────────────────────────────────────────
    if verify_lease is not None:
        try:
            verify_lease(auto_issue_dev=True)
        except LicenseError as e:
            print(f"[FATAL] License Verification Failed: {e}")
            raise
        except Exception as e:
            print(f"[FATAL] Unexpected error during license verification: {e}")
            raise LicenseError(f"License verification error: {e}")

    if not input_video_path or not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    # Optional slice windowing for targeted testing without external pre-slicing hacks
    original_input_path = input_video_path
    if start_offset > 0.0 or duration_limit is not None:
        windowed_input_path = os.path.join(temp_dir, "windowed_source.mp4")
        win_cmd = ["ffmpeg", "-y", "-ss", str(start_offset)]
        if duration_limit is not None:
            win_cmd.extend(["-t", str(duration_limit)])
        win_cmd.extend([
            "-i", input_video_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            *engine_ffmpeg.get_cfr_args(),
            "-c:a", "aac",
            "-af", "aresample=async=1",
            windowed_input_path,
        ])
        print(f"[pipeline] Slicing targeted window: start={start_offset}s, duration={duration_limit}s")
        subprocess.run(win_cmd, check=True, capture_output=True)
        input_video_path = windowed_input_path

    # ── Probe duration and dimensions ────────────────────────────────────────
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=width,height",
        "-of", "json", input_video_path,
    ]
    total_dur = 0.0
    video_w, video_h = 1920, 1080
    try:
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(res.stdout)
        total_dur = float(probe_data["format"]["duration"])
        for stream in probe_data.get("streams", []):
            if "width" in stream and "height" in stream:
                video_w = int(stream["width"])
                video_h = int(stream["height"])
                break
    except Exception as e:
        print(f"Warning: Could not parse video duration/dimensions: {e}")
    print(f"[pipeline] Loaded {input_video_path} | {video_w}x{video_h} | Duration: {total_dur:.2f}s")

    strip_social_ui = kwargs.get("strip_social_ui")
    if strip_social_ui is None:
        # Auto-Detection Heuristic: default True for vertical videos
        strip_social_ui = (video_h > video_w)
        if strip_social_ui:
            print(f"[pipeline] Auto-enabled strip_social_ui for vertical aspect ratio ({video_w}x{video_h})")

    # ── Blurred Watermark & Logo Pre-Render Rejection (Phase 5) ─────────────
    check_watermarks = kwargs.get("check_watermarks", kwargs.get("reject_watermarks", True))
    if check_watermarks and not strip_social_ui:
        if detect_blurred_logos(input_video_path):
            print("\n" + "!" * 80)
            print("[REJECTED] Video contains a blurred logo/watermark. Dropping to prevent shadowban.")
            print("!" * 80 + "\n")
            raise WatermarkRejectionError("Video contains a blurred logo/watermark. Dropping to prevent shadowban.")

    try:
        # ====================================================================
        # PHASE 1: Global Audio
        # ====================================================================
        print("\n" + "=" * 60)
        print("PHASE 1: Global Audio (WhisperX, VAD, Pyannote)")
        print("=" * 60)
        _log_vram("Phase 1 start")

        import audio_intelligence

        full_wav_path = os.path.join(temp_dir, "full_audio.wav")
        try:
            audio_intelligence.extract_audio(input_video_path, full_wav_path)
        except Exception as exc:
            raise RuntimeError(f"Audio extraction failed: {exc}") from exc

        # Check MD5 hash of the extracted audio stream (Phase 1 Cache Interceptor)
        audio_md5 = compute_audio_stream_md5(full_wav_path)
        cache_key = _get_audio_cache_key(original_input_path, start_offset, duration_limit)
        cached_audio = (_load_audio_cache(audio_md5) if audio_md5 else None) or _load_audio_cache(cache_key)

        words = list(words) if words else []
        speaker_segments = []

        if cached_audio:
            print(f"[pipeline] Audio cache HIT (MD5: {audio_md5[:12]}... / Key: {cache_key}) — skipping Phase 1 WhisperX/Pyannote compute.")
            if not words and "words" in cached_audio:
                words = cached_audio.get("words", [])
            if not silence_gaps and "silence_gaps" in cached_audio:
                silence_gaps = cached_audio.get("silence_gaps", [])
            if "speaker_segments" in cached_audio:
                speaker_segments = cached_audio.get("speaker_segments", [])

        if not words:
            print("[pipeline] words not supplied — running WhisperX alignment...")
            try:
                if os.path.isfile(full_wav_path):
                    words = audio_intelligence.run_whisperx_alignment(full_wav_path, language="en")
                    print(f"[pipeline] ASR complete: {len(words)} word(s) aligned.")
                else:
                    words = []
            except Exception as exc:
                print(f"[pipeline] WhisperX failed ({exc}); proceeding with empty transcript.")
                words = []

        if not silence_gaps and words:
            try:
                # Use decoupled VAD to prevent running WhisperX a second time
                silence_gaps = audio_intelligence.run_silero_vad_v5(full_wav_path)
                print(f"[pipeline] VAD detected {len(silence_gaps)} silence gap(s).")
            except Exception as exc:
                print(f"[pipeline] VAD analysis failed ({exc}); silence_gaps=[].")
                silence_gaps = []

        # Dynamic VAD Bypass (Speech-Aware Routing)
        disable_vad = False
        if total_dur > 0:
            wpm = len(words) / (total_dur / 60.0)
            print(f"[pipeline] Speech density: {wpm:.1f} WPM.")
            if wpm < 25.0:
                print("[pipeline] WPM < 25. Disabling VAD dead-air pruning to preserve action/gameplay.")
                disable_vad = True
        else:
            raise ValueError(f"Failed to probe valid video duration for {input_video_path}.")

        if not speaker_segments:
            if hf_token:
                try:
                    if os.path.isfile(full_wav_path):
                        speaker_segments = audio_intelligence.run_speaker_diarization(
                            full_wav_path, hf_token=hf_token
                        )
                        print(f"[pipeline] Diarization complete: {len(speaker_segments)} speaker turn(s).")
                except RuntimeError as exc:
                    # Clear actionable message from run_speaker_diarization if token invalid
                    print(f"[pipeline] Diarization skipped: {exc}")
                except Exception as exc:
                    print(f"[pipeline] Diarization failed ({exc}); continuing without speaker data.")
            else:
                # Try zero-token offline local CAM++ model (FunClip)
                try:
                    speaker_segments = audio_intelligence.run_funasr_campp_diarization(full_wav_path)
                    if speaker_segments:
                        print(f"[pipeline] FunASR CAM++ Diarization complete: {len(speaker_segments)} speaker turn(s).")
                except Exception as exc:
                    logger.debug(f"[pipeline] Offline CAM++ diarization skipped: {exc}")
                if not speaker_segments:
                    print(
                        "[pipeline] HF_TOKEN not set and CAM++ inactive — diarization skipped.\n"
                        "  To enable pyannote: set HF_TOKEN in Kaggle Secrets and accept the pyannote licence.\n"
                        "  https://hf.co/pyannote/speaker-diarization-3.1"
                    )

        # Save to scoped global audio cache under MD5 and cache_key
        cache_data = {
            "words": words,
            "silence_gaps": silence_gaps,
            "speaker_segments": speaker_segments,
            "audio_md5": audio_md5,
        }
        if audio_md5:
            _save_audio_cache(audio_md5, cache_data)
        _save_audio_cache(cache_key, cache_data)

        _evict_vram("Phase 1")

        # ====================================================================
        # PHASE 2: Global Curation
        # ====================================================================
        print("\n" + "=" * 60)
        print("PHASE 2: Global Curation (Ollama)")
        print("=" * 60)
        _log_vram("Phase 2 start")

        cuts = curation_engine.get_viral_cuts(
            words, silence_gaps, total_dur, input_video_path
        )
        print(f"[pipeline] Curation complete: {len(cuts)} segment(s) selected.")
        _evict_vram("Phase 2")

        # ====================================================================
        # PHASE 3: Global Vision & Tracking (YOLO)
        # ====================================================================
        print("\n" + "=" * 60)
        print("PHASE 3: Global Vision & Tracking (YOLO)")
        print("=" * 60)
        _log_vram("Phase 3 start")

        try:
            engine_vision._try_load_yolo()
        except Exception as e:
            print(f"[WARN] Failed to load YOLO: {e}")

        try:
            if hasattr(engine_vision, "detect_gaming_layout"):
                engine_vision.detect_gaming_layout(input_video_path)
        except Exception as e:
            print(f"[WARN] detect_gaming_layout failed: {e}")

        slice_paths = {}
        trajectories = {}
        pruned_intervals_map = {}
        for idx, cut in enumerate(cuts, 1):
            start = float(cut["start_time"])
            end   = float(cut["end_time"])
            dur   = end - start

            if dur <= 0.0:
                print(f"[WARN] Invalid slice duration {dur:.2f}s for Hook #{idx}; skipping.")
                continue

            slice_path       = os.path.join(temp_dir, f"slice_{idx}.mp4")
            clean_slice_path = os.path.join(temp_dir, f"clean_slice_{idx}.mp4")

            print(f"\n--- Slicing Hook #{idx} ({start:.2f}s -> {end:.2f}s, {dur:.2f}s) ---")
            vf_args = []
            if strip_social_ui:
                vf_args.append("crop=iw:ih*0.73:0:ih*0.12")
            vf_args.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

            slice_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-t", str(dur),
                "-i", input_video_path,
                "-vf", ",".join(vf_args),
            ]
            if engine_ffmpeg.is_nvenc_available():
                slice_cmd.extend([
                    "-c:v", "h264_nvenc", "-preset", "p1",
                    "-cq", "18", "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
                ])
            else:
                slice_cmd.extend([
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "18",
                ])
            slice_cmd.extend([
                *engine_ffmpeg.get_cfr_args(),
                "-c:a", "aac",
                "-af", "aresample=async=1",
                slice_path,
            ])
            
            try:
                subprocess.run(slice_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                # If NVENC falsely reported available or failed during run, fallback
                if "h264_nvenc" in slice_cmd:
                    fallback_cmd = list(slice_cmd)
                    idx_v = fallback_cmd.index("h264_nvenc")
                    fallback_cmd[idx_v] = "libx264"
                    idx_preset = fallback_cmd.index("-preset")
                    fallback_cmd[idx_preset+1] = "ultrafast"
                    
                    cq_idx = fallback_cmd.index("-cq")
                    fallback_cmd[cq_idx] = "-crf"
                    # Remove NVENC specific bitrates
                    b_idx = fallback_cmd.index("-b:v")
                    del fallback_cmd[b_idx:b_idx+6]
                    
                    subprocess.run(fallback_cmd, check=True, capture_output=True)
                else:
                    raise
            print(f"Detecting voice activity for Hook #{idx}...")
            try:
                if disable_vad:
                    print(f"[pipeline] Skipping VAD dead-air pruning for Hook #{idx} (WPM < 25).")
                    active_slice_path = slice_path
                    pruned_intervals_map[idx] = []
                else:
                    speech_segments  = engine_vad.detect_speech_segments(slice_path)
                    active_slice_path, pruned_intervals, total_pruned = engine_vad.strip_dead_air(slice_path, clean_slice_path, speech_segments)
                    pruned_intervals_map[idx] = pruned_intervals
            except Exception as e:
                print(f"[WARN] VAD processing failed: {e}")
                active_slice_path = slice_path
                pruned_intervals_map[idx] = []
                
            slice_paths[idx] = active_slice_path

            print(f"Computing tracking trajectory for Hook #{idx}...")
            try:
                clip_speakers = []
                for s in speaker_segments:
                    if s["end"] > start and s["start"] < end:
                        # Project speaker segments to the pruned timeline
                        s_rel_start = s["start"] - start
                        s_rel_end = s["end"] - start
                        proj_start = project_timestamp(s_rel_start, pruned_intervals_map[idx])
                        proj_end = project_timestamp(s_rel_end, pruned_intervals_map[idx])
                        # engine_vision expects absolute time from start of source
                        clip_speakers.append({
                            **s,
                            "start": proj_start + start,
                            "end": proj_end + start
                        })

                scene_cuts = engine_vision.detect_scene_cuts(
                    active_slice_path,
                    start_time=0.0,
                    end_time=(end - start)
                )
                if scene_cuts:
                    print(f"[pipeline] Hook #{idx}: Detected {len(scene_cuts)} scene cut(s) at {scene_cuts}s")

                trajectory = engine_vision.calculate_tracking_trajectory(
                    active_slice_path,
                    speaker_segments=clip_speakers,
                    slice_start_time=start,
                    scene_cuts=scene_cuts
                )
            except Exception as e:
                print(f"[WARN] Tracking trajectory failed: {e}")
                trajectory = []
            trajectories[idx] = trajectory

        _evict_vram("Phase 3")

        # ====================================================================
        # PHASE 4: Batch Render Engine
        # ====================================================================
        print("\n" + "=" * 60)
        print("PHASE 4: Batch Render Engine")
        print("=" * 60)
        _log_vram("Phase 4 start")

        rendered_files = []

        for idx, cut in enumerate(cuts, 1):
            start = float(cut["start_time"])
            end   = float(cut["end_time"])
            dur   = end - start

            if dur <= 0.0 or idx not in slice_paths:
                continue

            active_slice_path = slice_paths[idx]
            trajectory = trajectories.get(idx, [])
            out_path = os.path.join(output_dir, f"reel_{idx}_input_raw.mp4")

            slice_words = []
            if words:
                for w in words:
                    w_start = w.get("start", 0)
                    w_end = w.get("end", 0)
                    if w_start >= start and w_end <= end:
                        rel_start = w_start - start
                        rel_end = w_end - start
                        # Project timestamps using pruned intervals
                        proj_start = project_timestamp(rel_start, pruned_intervals_map.get(idx, []))
                        proj_end = project_timestamp(rel_end, pruned_intervals_map.get(idx, []))
                        slice_words.append({**w, "start": proj_start, "end": proj_end})

            if not slice_words:
                print(f"[WARNING] Hook #{idx}: 0 words after ASR — no subtitle overlay.")

            slice_ass_path = os.path.join(temp_dir, f"slice_{idx}.ass")
            traj_data = trajectory if isinstance(trajectory, dict) else {}
            is_dual = bool(
                traj_data.get("is_dual_speaker", False)
                or traj_data.get("layout") == "dual_speaker_split"
            )
            ass_margin_v = 960 if is_dual else caption_engine.MARGIN_V

            ass_path = caption_engine.generate_karaoke_ass(
                words=slice_words,
                output_ass_path=slice_ass_path,
                margin_v=ass_margin_v,
            )
            if not (ass_path and os.path.isfile(ass_path)):
                ass_path = None
                print(f"Caption: None (no subtitle for Hook #{idx})")
            else:
                print(f"Caption: {slice_ass_path} ({len(slice_words)} words)")

            emoji_overlays = []

            video_coords = {
                "x": 0, "y": (1920 - 1540) // 2,
                "width": 1080, "height": 1540,
                "scale_x": 1.0, "scale_y": 1.0,
            }
            text_coords = {"x": 60, "y": 80, "font_size": 48, "text_content": ""}
            opts = engine_ffmpeg.RenderOptions(
                speed=1.12, preset="p6", cq=18, crf=18, bg_color=bg_color,
                strip_social_ui=strip_social_ui
            )

            print(f"Rendering Reel #{idx}...")
            with vram_guard(f"Render Reel #{idx}"):
                engine_ffmpeg.render_clip(
                    input_path=active_slice_path,
                    output_path=out_path,
                    video_coords=video_coords,
                    text_coords=text_coords,
                    logo_path=logo_path,
                    logo_coords=logo_coords,
                    options=opts,
                    trajectory=trajectory,
                    ass_path=ass_path,
                    emojis=emoji_overlays,
                )

            clip_speakers = [
                s for s in speaker_segments
                if s["end"] > start and s["start"] < end
            ]

            reel_obj = ReelPath(out_path, metadata={
                "reel_index":      idx,
                "start":           start,
                "end":             end,
                "virality_score":  cut.get("virality_score", 85),
                "reason":          cut.get("reason", ""),
                "hook_sentence":   cut.get("hook_sentence", ""),
                "path":            out_path,
                "speaker_segments": clip_speakers,
            })
            rendered_files.append(reel_obj)

        print(f"\n[OK] All {len(rendered_files)} reel(s) rendered successfully.")
        return rendered_files

    finally:
        # Aggressive disk cleanup
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"[pipeline] Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"[pipeline] Warning: Failed to clean temp dir: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Viral Video Pipeline")
    parser.add_argument("--input",  "-i", required=True,  help="Path to input video")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds to process")
    parser.add_argument("--strip-social-ui", action="store_true", help="Strip top/bottom UI from vertical videos")
    args = parser.parse_args()
    run_pipeline(
        input_video_path=args.input, 
        output_dir=args.output, 
        words=[],
        start=args.start,
        duration=args.duration,
        strip_social_ui=args.strip_social_ui
    )
