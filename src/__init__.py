"""
2.0 Autonomous Video Repurposing Pipeline
----------------------------------------
Core package exposing high-level video processing, AI curation,
audio intelligence, caption styling, and FFmpeg rendering engines.
"""

from .pipeline import run_pipeline, ReelPath
from .engine_ffmpeg import RenderOptions, render_clip, build_ffmpeg_command
from .curation_engine import get_viral_cuts, validate_and_format_cuts
from .caption_engine import generate_karaoke_ass, generate_ass_subtitles
from .audio_intelligence import (
    extract_audio,
    run_silero_vad_v5,
    run_whisperx_alignment,
    snap_to_acoustic_trough,
    detect_dead_air,
)

__all__ = [
    "run_pipeline",
    "ReelPath",
    "RenderOptions",
    "render_clip",
    "build_ffmpeg_command",
    "get_viral_cuts",
    "validate_and_format_cuts",
    "generate_karaoke_ass",
    "generate_ass_subtitles",
    "extract_audio",
    "run_silero_vad_v5",
    "run_whisperx_alignment",
    "snap_to_acoustic_trough",
    "detect_dead_air",
]
