#!/usr/bin/env python3
"""
audio_intelligence.py
---------------------
Production-grade Speech & Audio Intelligence module.
Phase 1: Semantic Curation & Audio Tightening implementation.

Handles 16kHz mono WAV extraction, WhisperX / faster-whisper phoneme-level forced alignment
(with GPU FP16 acceleration and INT8/CPU fallback), Silero VAD v5 vocal activity mapping,
dead air detection, and acoustic trough boundary snapping.
"""

import os
import subprocess
import json
import wave
import struct
import math
import logging
import gc
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger("audio_intelligence")



def extract_audio(video_path: str, output_wav_path: str) -> str:
    """
    Extracts audio from a video or audio container to a 16kHz mono 16-bit PCM WAV file.

    Args:
        video_path: Absolute or relative path to source media file.
        output_wav_path: Target path for the output 16kHz mono WAV file.

    Returns:
        The target WAV file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-af", "afftdn=nf=-25",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_wav_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError(f"Failed to extract 16kHz mono audio from {video_path}") from e
    return output_wav_path


def run_silero_vad_v5(wav_path: str, threshold: float = 0.5) -> List[Tuple[float, float]]:
    """
    Silero VAD v5 speech activity detection processing 16kHz mono audio.
    Identifies non-vocal / silence gap intervals in seconds.

    Args:
        wav_path: Path to 16kHz mono WAV file.
        threshold: Silero VAD speech probability decision threshold (default 0.5).

    Returns:
        List of (start_sec, end_sec) silence interval tuples.
    """
    try:
        import torch

        # Load Silero VAD v5 model via Torch Hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            verbose=False
        )
        (get_speech_timestamps, _, read_audio, _, _) = utils

        # Load and process audio tensor (16kHz mono float32)
        wav_tensor = read_audio(wav_path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(
            wav_tensor, model, sampling_rate=16000, threshold=threshold
        )

        speech_segments = [
            (ts['start'] / 16000.0, ts['end'] / 16000.0)
            for ts in speech_timestamps
        ]

        # Calculate total audio duration
        with wave.open(wav_path, 'rb') as wf:
            total_duration = wf.getnframes() / float(wf.getframerate())

        # Invert speech segments to obtain silence gap intervals
        silence_gaps = []
        last_end = 0.0
        for start, end in speech_segments:
            if start - last_end >= 0.05:  # Minimum silence threshold 50ms
                silence_gaps.append((last_end, start))
            last_end = end
        if total_duration - last_end >= 0.05:
            silence_gaps.append((last_end, total_duration))

        return silence_gaps

    except Exception as e:
        logger.warning(f"Silero VAD v5 execution error ({e}). Falling back to RMS energy calculation.")
        return run_rms_vad_fallback(wav_path)

    finally:
        if "model" in locals():
            try:
                del model
            except Exception:
                pass
        if "wav_tensor" in locals():
            try:
                del wav_tensor
            except Exception:
                pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# Backward compatibility alias
run_silero_vad_local = run_silero_vad_v5


def run_rms_vad_fallback(wav_path: str, threshold: float = 0.02, min_silence_len: float = 0.5) -> List[Tuple[float, float]]:
    """
    Lightweight short-time RMS energy voice activity detection fallback.

    Args:
        wav_path: Path to 16kHz mono WAV file.
        threshold: RMS energy silence threshold.
        min_silence_len: Minimum silence length in seconds.

    Returns:
        List of silence gap intervals (start_sec, end_sec).
    """
    filename = os.path.basename(wav_path)

    try:
        with wave.open(wav_path, 'rb') as wf:
            rate = wf.getframerate()
            nframes = wf.getnframes()
            raw_data = wf.readframes(nframes)

        samples = struct.unpack(f"{nframes}h", raw_data)
        float_samples = [s / 32768.0 for s in samples]

        window_sec = 0.03
        window_size = int(rate * window_sec)
        num_windows = len(float_samples) // window_size

        is_silent = []
        for i in range(num_windows):
            win = float_samples[i * window_size: (i + 1) * window_size]
            rms = math.sqrt(sum(x * x for x in win) / len(win)) if win else 0.0
            is_silent.append(rms < threshold)

        silence_gaps = []
        in_silence = False
        silence_start = 0.0

        for i, silent in enumerate(is_silent):
            t = i * window_sec
            if silent and not in_silence:
                in_silence = True
                silence_start = t
            elif not silent and in_silence:
                in_silence = False
                dur = t - silence_start
                if dur >= min_silence_len:
                    silence_gaps.append((silence_start, t))

        if in_silence:
            dur = (num_windows * window_sec) - silence_start
            if dur >= min_silence_len:
                silence_gaps.append((silence_start, num_windows * window_sec))

        return silence_gaps
    except Exception as e:
        logger.error(f"RMS VAD fallback failed: {e}")
        return []


def detect_dead_air(audio_path: str, max_pause_sec: float = 0.5) -> List[Tuple[int, int]]:
    """
    Detects intervals where no vocal cord activity exists exceeding max_pause_sec (default 500ms).

    Args:
        audio_path: Path to audio file (or video container).
        max_pause_sec: Minimum pause duration in seconds to classify as dead air (default 0.5s = 500ms).

    Returns:
        List of exact millisecond interval tuples [(start_ms, end_ms), ...].
    """
    temp_wav_created = False
    wav_path = audio_path

    # If input is a video or non-WAV audio, extract temporary 16kHz mono WAV
    if not audio_path.lower().endswith(".wav"):
        temp_wav_path = audio_path + "_temp_deadair.wav"
        extract_audio(audio_path, temp_wav_path)
        wav_path = temp_wav_path
        temp_wav_created = True

    try:
        # Run Silero VAD v5 to identify silence intervals (in seconds)
        silence_gaps = run_silero_vad_v5(wav_path)

        dead_air_ms = []
        for start_sec, end_sec in silence_gaps:
            duration = end_sec - start_sec
            if duration >= max_pause_sec:
                start_ms = int(round(start_sec * 1000.0))
                end_ms = int(round(end_sec * 1000.0))
                dead_air_ms.append((start_ms, end_ms))

        return dead_air_ms
    finally:
        if temp_wav_created and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


def snap_to_acoustic_trough(
    timestamp: Union[float, int],
    silence_intervals: Union[List[Tuple[Union[float, int], Union[float, int]]], List[Dict[str, Any]]],
    direction: str = 'backward',
    tolerance: float = 0.4,
    is_ms: bool = False
) -> Union[float, int]:
    """
    Snaps a target cut timestamp to the nearest natural breath or silence trough within tolerance.

    Args:
        timestamp: Target cut timestamp (in seconds or milliseconds).
        silence_intervals: List of (start, end) tuples or {'start': float, 'end': float} dicts.
        direction: Preferred snap direction ('backward', 'forward', or 'nearest').
        tolerance: Maximum displacement allowed for snapping (in seconds, default 0.4s).
        is_ms: Explicitly declare if timestamp and intervals are in milliseconds.

    Returns:
        Snapped timestamp maintaining original unit scale.
    """
    if not silence_intervals:
        return timestamp

    # Scale tolerance matching timestamp units
    tol_scale = tolerance * 1000.0 if is_ms else tolerance
    ts = float(timestamp)

    candidates = []

    for item in silence_intervals:
        if isinstance(item, dict):
            s_start = float(item.get("start", 0.0))
            s_end = float(item.get("end", 0.0))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            s_start = float(item[0])
            s_end = float(item[1])
        else:
            continue

        midpoint = (s_start + s_end) / 2.0

        # Case 1: Timestamp falls directly inside a silence interval [s_start, s_end]
        if s_start <= ts <= s_end:
            if direction == 'backward':
                target = s_end if ts >= s_end else midpoint
            elif direction == 'forward':
                target = s_start if ts <= s_start else midpoint
            else:
                target = midpoint
            candidates.append((0.0, target))
            continue

        # Case 2: Silence interval is BEFORE timestamp
        if s_end < ts:
            dist = ts - s_end
            if dist <= tol_scale and direction in ('backward', 'nearest'):
                candidates.append((dist, s_end))

        # Case 3: Silence interval is AFTER timestamp
        if s_start > ts:
            dist = s_start - ts
            if dist <= tol_scale and direction in ('forward', 'nearest'):
                candidates.append((dist, s_start))

    if not candidates:
        return timestamp

    # Sort candidates by distance (ascending)
    candidates.sort(key=lambda x: x[0])
    best_snap = candidates[0][1]

    if isinstance(timestamp, int):
        return int(round(best_snap))
    return float(round(best_snap, 4))


def run_whisperx_alignment(
    wav_path: str,
    device: str = "cuda",
    compute_type: str = "float16",
    model_name: str = "large-v2",
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Executes WhisperX speech transcription (faster-whisper backend) and Wav2Vec2 forced alignment.
    Runs FP16 on GPU (or INT8/CPU fallback).

    Args:
        wav_path: Path to 16kHz mono WAV file.
        device: Preferred device ('cuda' or 'cpu').
        compute_type: Precision mode ('float16' on GPU, 'int8' on CPU).
        model_name: Whisper model architecture name.
        language: Language code to transcribe and align (defaults to 'en').

    Returns:
        List of aligned word dictionaries [{'word': str, 'start': float, 'end': float, 'score': float}].
    """
    try:
        import torch
        import whisperx

        # Hardware execution selection: FP16 on GPU, INT8 on CPU
        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "int8"
        else:
            device = "cuda"
            compute_type = "float16"

        logger.info(f"Running WhisperX ({model_name}) on device={device}, compute_type={compute_type}, language={language}")

        # 1. Transcribe audio with faster-whisper backend (locked to English to kill hallucinations)
        audio = whisperx.load_audio(wav_path)
        try:
            model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)
            result = model.transcribe(audio, batch_size=16, language=language)
        except Exception as trans_err:
            if "out of memory" in str(trans_err).lower() and device == "cuda":
                logger.warning(f"CUDA OOM during WhisperX transcription ({trans_err}). Retrying with batch_size=4...")
                if "model" in locals():
                    try:
                        del model
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)
                    result = model.transcribe(audio, batch_size=4, language=language)
                except Exception as retry_err:
                    logger.warning(f"WhisperX GPU retry failed ({retry_err}). Falling back to CPU transcription (int8)...")
                    if "model" in locals():
                        try:
                            del model
                        except Exception:
                            pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    device = "cpu"
                    compute_type = "int8"
                    model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)
                    result = model.transcribe(audio, batch_size=4, language=language)
            else:
                raise trans_err

        language_code = result.get("language") or language or "en"

        # Explicitly free Whisper transcription model from VRAM before loading alignment model
        if "model" in locals():
            try:
                del model
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Wav2Vec2 Phoneme / Alignment stage
        try:
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            aligned_result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device=device,
                return_char_alignments=False
            )
        except Exception as align_err:
            if "out of memory" in str(align_err).lower() and device == "cuda":
                logger.warning(f"CUDA OOM during WhisperX alignment ({align_err}). Retrying alignment on CPU...")
                if "model_a" in locals():
                    try:
                        del model_a
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model_a, metadata = whisperx.load_align_model(language_code=language_code, device="cpu")
                aligned_result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device="cpu",
                    return_char_alignments=False
                )
            else:
                raise align_err
        finally:
            if "model_a" in locals():
                try:
                    del model_a
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 3. Structure aligned word output
        words_out = []
        for segment in aligned_result.get("segments", []):
            for w in segment.get("words", []):
                if "start" in w and "end" in w:
                    words_out.append({
                        "word": str(w["word"]).strip(),
                        "start": round(float(w["start"]), 3),
                        "end": round(float(w["end"]), 3),
                        "score": round(float(w.get("score", 1.0)), 3)
                    })
        return words_out

    except Exception as e:
        logger.warning(f"WhisperX alignment failed or unavailable ({e}). Using fallback engine.")
        return run_whisper_fallback_local(wav_path)


def run_whisper_fallback_local(wav_path: str) -> List[Dict[str, Any]]:
    """
    Fallback word timestamp generator using faster-whisper on CPU (int8).
    If unavailable or fails, returns [] so pipeline skips subtitle burning.
    """
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        segments, _ = model.transcribe(wav_path, word_timestamps=True)
        words = []
        for segment in segments:
            for word in getattr(segment, "words", []):
                w_start = getattr(word, "start", None)
                w_end = getattr(word, "end", None)
                if w_start is not None and w_end is not None:
                    words.append({
                        "word": str(getattr(word, "word", "")).strip(),
                        "start": round(float(w_start), 2),
                        "end": round(float(w_end), 2),
                        "score": round(float(getattr(word, "probability", 1.0) or 1.0), 2)
                    })
                
        if not words:
            logger.warning("[audio_intelligence] faster-whisper returned empty alignment.")
        return words
    except ImportError:
        try:
            import whisper
            model = whisper.load_model("base", device="cpu")
            result = model.transcribe(wav_path, word_timestamps=True)
            words = []
            for segment in result.get("segments", []):
                for w in segment.get("words", []):
                    w_start = w.get("start")
                    w_end = w.get("end")
                    if w_start is not None and w_end is not None:
                        words.append({
                            "word": str(w.get("word", "")).strip(),
                            "start": round(float(w_start), 2),
                            "end": round(float(w_end), 2),
                            "score": round(float(w.get("probability", 1.0) or 1.0), 2)
                        })
            if words:
                return words
        except Exception:
            pass
        logger.warning("[audio_intelligence] neither faster-whisper nor whisper installed; skipping local fallback captions.")
        return []
    except Exception as e:
        logger.warning(f"[audio_intelligence] faster-whisper local fallback failed: {e}")
        return []


# Backward compatibility alias for contract verification
get_vocal_analysis = lambda *args, **kwargs: {}





# ---------------------------------------------------------------------------
# Speaker Diarization  (pyannote.audio)
# ---------------------------------------------------------------------------

def run_speaker_diarization(
    wav_path: str,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Runs speaker diarization on a 16kHz mono WAV using pyannote/speaker-diarization-3.1.

    Sequential VRAM rule: caller MUST have evicted WhisperX (del model + gc.collect +
    torch.cuda.empty_cache) before calling this function.

    Args:
        wav_path:  Path to 16kHz mono WAV file.
        hf_token:  Hugging Face access token (must have accepted pyannote licence).
                   If None, reads os.environ["HF_TOKEN"].

    Returns:
        List of {"speaker": str, "start": float, "end": float} dicts sorted by start.
        Returns [] (with warning) if pyannote not installed or HF_TOKEN absent.

    Raises:
        RuntimeError: With a clear actionable message if HF_TOKEN is missing.
    """
    import gc as _gc

    token = hf_token or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "[diarization] HF_TOKEN is not set.\n"
            "Action required:\n"
            "  1. Accept the pyannote licence at https://hf.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept the segmentation model licence at https://hf.co/pyannote/segmentation-3.0\n"
            "  3. Add your Hugging Face token as a Kaggle Secret named HF_TOKEN."
        )

    if not os.path.isfile(wav_path):
        logger.warning(f"[diarization] WAV file not found: {wav_path}. Returning empty segments.")
        return []

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            before_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[VRAM] Before diarization load: {before_mb:.1f} MB allocated.")
    except Exception:
        gpu_available = False

    diarization_pipeline = None
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.warning(
            "[diarization] pyannote.audio not installed. "
            "Add 'pyannote.audio' to your Kaggle dependencies and re-run. "
            "Returning empty speaker segments."
        )
        return []

    try:
        logger.info("[diarization] Loading pyannote/speaker-diarization-3.1...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        if gpu_available:
            diarization_pipeline = diarization_pipeline.to(
                __import__("torch").device("cuda")
            )
        logger.info("[diarization] Model loaded. Running inference...")

        diarization_result = diarization_pipeline(wav_path)

        annotation = getattr(diarization_result, "speaker_diarization", diarization_result)
        if hasattr(annotation, "annotation"):
            annotation = annotation.annotation

        segments: List[Dict[str, Any]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": str(speaker),
                "start":   round(float(turn.start), 4),
                "end":     round(float(turn.end),   4),
            })

        logger.info(f"[diarization] Completed: {len(segments)} speaker turn(s) detected.")
        return sorted(segments, key=lambda x: x["start"])

    except Exception as exc:
        logger.error(f"[diarization] Failed: {exc}")
        return []

    finally:
        # Mandatory VRAM eviction after diarization
        if diarization_pipeline is not None:
            try:
                del diarization_pipeline
            except Exception:
                pass
        _gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                print(f"[VRAM] After diarization eviction: {after_mb:.1f} MB allocated.")
        except Exception:
            pass


def run_funasr_campp_diarization(wav_path: str) -> List[Dict[str, Any]]:
    """
    Runs local speaker diarization using FunASR CAM++ model (from FunClip).
    Provides a zero-token offline alternative to pyannote.
    """
    if not os.path.isfile(wav_path):
        return []
    try:
        from funasr import AutoModel
        # Use CAM++ speaker verification / diarization pipeline
        model = AutoModel(model="damo/speech_campplus_sv_zh-cn_16k-common", disable_update=True)
        res = model.generate(input=wav_path)
        segments = []
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict) and "spk" in item:
                    segments.append({
                        "speaker": f"SPEAKER_{item.get('spk', 0)}",
                        "start": round(float(item.get("start", 0.0)) / 1000.0, 4),
                        "end": round(float(item.get("end", 0.0)) / 1000.0, 4),
                    })
        return sorted(segments, key=lambda x: x["start"])
    except Exception as exc:
        logger.debug(f"[funasr_campp] CAM++ diarization unavailable or failed: {exc}")
        return []

