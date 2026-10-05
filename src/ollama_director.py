"""
src/ollama_director.py
----------------------
Ollama-powered Virtual TV Broadcast Director for multi-speaker podcasts and gaming videos.
Directs camera cuts (Solo Host, Solo Guest, 9:8 Vertical Split-Stack, Wide Two-Shot)
based on timestamped dialogue semantics, conversational floor-holding, and filler-word suppression.
Includes a fully deterministic rule-based fallback director if Ollama is offline or unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from camera_framing import CameraFramingConfig, CameraZone

logger = logging.getLogger("ollama_director")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

FILLER_WORDS = {
    "yeah", "yep", "yes", "uh-huh", "mhm", "um", "uh", "ah", "oh",
    "okay", "ok", "right", "sure", "cool", "haha", "hahaha", "laugh",
    "laughter", "wow", "true", "totally", "exactly"
}


def is_ollama_available(base_url: str = OLLAMA_BASE_URL, timeout: float = 1.5) -> bool:
    """Checks if the local Ollama server is responding."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", headers={"User-Agent": "Antigravity/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_ollama_director_llm(
    prompt: str,
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    timeout: float = 8.0,
) -> Optional[str]:
    """Sends prompt to local Ollama API and retrieves completion response."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        }
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Antigravity/2.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response")
    except Exception as e:
        logger.debug(f"[ollama_director] Ollama generation failed: {e}")
    return None


class TVDirector:
    """
    Virtual TV Studio Broadcast Director.
    Combines conversational transcript reasoning with creator-defined camera zones.
    """

    def __init__(
        self,
        camera_config: CameraFramingConfig,
        min_shot_duration: float = 2.0,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL
    ):
        self.config = camera_config
        self.min_shot_duration = max(1.5, min_shot_duration)
        self.base_url = base_url
        self.model = model

    def direct_clip(
        self,
        words_or_dialogue: List[Dict[str, Any]],
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
        use_ollama: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Plans shot cuts for a clip.
        Returns a list of shot segments with normalized timestamps [0.0, clip_duration]:
        [
            {"start": 0.0, "end": 5.2, "type": "single", "camera": "zone_host", "zone": {...}},
            {"start": 5.2, "end": 12.0, "type": "single", "camera": "zone_guest", "zone": {...}},
            {"start": 12.0, "end": 20.0, "type": "split_stack", "camera": "split", "top_zone": {...}, "bot_zone": {...}}
        ]
        """
        # Group word-level timings into coherent dialogue turns
        cues = self._group_dialogue_cues(words_or_dialogue, clip_start, clip_end)

        clip_dur = (clip_end - clip_start) if (clip_end is not None and clip_end > clip_start) else None
        if not clip_dur and cues:
            clip_dur = max(c["end"] for c in cues)
        if not clip_dur or clip_dur <= 0:
            clip_dur = 30.0

        raw_cuts = None
        if use_ollama and is_ollama_available(self.base_url):
            try:
                raw_cuts = self._run_ollama_director(cues, clip_dur)
            except Exception as e:
                logger.warning(f"[ollama_director] Ollama TV director error: {e}. Falling back to state-machine.")

        if not raw_cuts:
            # Deterministic TV Director Rule Engine
            raw_cuts = self._run_rule_based_director(cues, clip_dur)

        # Enforce minimum shot duration (hysteresis) and timeline smoothing
        smoothed_cuts = self._post_process_cuts(raw_cuts, clip_dur)

        # Attach concrete CameraZone coordinates for FFmpeg rendering
        resolved_shots = self._resolve_zone_coordinates(smoothed_cuts)
        return resolved_shots

    def _group_dialogue_cues(
        self,
        words: List[Dict[str, Any]],
        clip_start: float,
        clip_end: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Groups raw word segments into conversational sentences with relative timestamps."""
        if not words:
            return []

        cues: List[Dict[str, Any]] = []
        cur_speaker = None
        cur_text_words = []
        cur_start = None
        cur_end = None

        for w in words:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", 0.0))

            # Filter if outside bounds
            if clip_end is not None and ws >= clip_end:
                continue
            if we <= clip_start:
                continue

            rel_start = max(0.0, ws - clip_start)
            rel_end = max(0.1, we - clip_start)
            speaker = w.get("speaker") or w.get("speaker_id") or "SPEAKER_00"
            word_str = str(w.get("word") or w.get("text") or "").strip()

            if cur_speaker is None or cur_speaker != speaker or (cur_end and (rel_start - cur_end > 1.2)):
                if cur_text_words and cur_start is not None and cur_end is not None:
                    cues.append({
                        "start": round(cur_start, 2),
                        "end": round(cur_end, 2),
                        "speaker": cur_speaker,
                        "text": " ".join(cur_text_words),
                    })
                cur_speaker = speaker
                cur_text_words = [word_str]
                cur_start = rel_start
                cur_end = rel_end
            else:
                cur_text_words.append(word_str)
                cur_end = rel_end

        if cur_text_words and cur_start is not None and cur_end is not None:
            cues.append({
                "start": round(cur_start, 2),
                "end": round(cur_end, 2),
                "speaker": cur_speaker,
                "text": " ".join(cur_text_words),
            })

        return cues

    def _run_ollama_director(self, cues: List[Dict[str, Any]], clip_duration: float) -> Optional[List[Dict[str, Any]]]:
        """Prompts Ollama TV Broadcast Director to produce camera switching plan."""
        zone_descriptions = []
        for z in self.config.zones:
            zone_descriptions.append(f"- Camera '{z.id}': {z.label} (Speaker: {z.speaker_label or 'Any'})")

        dialogue_lines = []
        for c in cues:
            dialogue_lines.append(f"[{c['start']:.1f}s - {c['end']:.1f}s] {c['speaker']}: \"{c['text']}\"")

        prompt = f"""You are an elite TV studio broadcast director for vertical video.
Direct camera cuts for this conversation (Total Duration: {clip_duration:.1f}s).

Available Camera Angles:
{chr(10).join(zone_descriptions)}
- Camera 'split': 9:8 Dual Split-Screen (Host on top, Guest on bottom)

Dialogue Transcript:
{chr(10).join(dialogue_lines) if dialogue_lines else "[Continuous dialogue]"}

Directing Rules:
1. Hold camera on the speaker who has the floor.
2. DO NOT cut away for quick 1-2 word filler words ('yeah', 'uh-huh', 'mhm', laughs under 1.5s). Keep camera on the main speaker.
3. Switch to 'split' or 'zone_wide' when both speakers debate, banter back and forth, or laugh together.
4. Every shot MUST last at least {self.min_shot_duration:.1f} seconds.
5. The shots must cover the full clip from 0.0 to {clip_duration:.1f} without gaps.

Output MUST be a JSON array of objects with keys: "start", "end", "camera", "type".
Example format:
[
  {{"start": 0.0, "end": 6.5, "camera": "zone_host", "type": "single"}},
  {{"start": 6.5, "end": 14.0, "camera": "zone_guest", "type": "single"}},
  {{"start": 14.0, "end": {clip_duration:.1f}, "camera": "split", "type": "split_stack"}}
]
Respond strictly with valid JSON."""

        resp_text = call_ollama_director_llm(prompt, self.base_url, self.model)
        if not resp_text:
            return None

        # Clean JSON from response
        clean_json = resp_text.strip()
        match = re.search(r'\[\s*\{.*\}\s*\]', clean_json, re.DOTALL)
        if match:
            clean_json = match.group(0)

        data = json.loads(clean_json)
        if isinstance(data, list) and len(data) > 0:
            valid_cuts = []
            for item in data:
                if isinstance(item, dict) and "start" in item and "end" in item:
                    valid_cuts.append({
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "camera": str(item.get("camera", "zone_host")),
                        "type": str(item.get("type", "single")),
                    })
            if valid_cuts:
                return valid_cuts
        return None

    def _run_rule_based_director(self, cues: List[Dict[str, Any]], clip_duration: float) -> List[Dict[str, Any]]:
        """
        Deterministic TV Director state-machine:
        1. Filters out filler words.
        2. Detects rapid banter / speaker collisions -> switches to 9:8 split-stack.
        3. Holds camera on the dominant speaker with a minimum 2.0s dwell time.
        """
        host_zone = self.config.get_host_zone()
        guest_zone = self.config.get_guest_zone()
        wide_zone = self.config.get_wide_zone()

        host_id = host_zone.id if host_zone else "zone_host"
        guest_id = guest_zone.id if guest_zone else "zone_guest"
        wide_id = wide_zone.id if wide_zone else "zone_wide"

        if not cues:
            # Default to split or host
            if self.config.split_preference == "split_stack" and len(self.config.zones) >= 2:
                return [{"start": 0.0, "end": clip_duration, "camera": "split", "type": "split_stack"}]
            return [{"start": 0.0, "end": clip_duration, "camera": host_id, "type": "single"}]

        # 1. Filter out filler words from speaker switches
        cleaned_cues = []
        for cue in cues:
            text = cue["text"].lower().strip(".,!?\"' ")
            duration = cue["end"] - cue["start"]
            is_filler = (text in FILLER_WORDS or all(w in FILLER_WORDS for w in text.split())) and duration < 1.8
            if not is_filler:
                cleaned_cues.append(cue)

        if not cleaned_cues:
            cleaned_cues = cues

        shots = []
        i = 0
        while i < len(cleaned_cues):
            cue = cleaned_cues[i]
            speaker = cue.get("speaker") or "SPEAKER_00"

            # Check if there is rapid banter or simultaneous overlapping speech
            has_rapid_banter = False
            banter_end = cue["end"]
            j = i + 1
            while j < len(cleaned_cues):
                next_cue = cleaned_cues[j]
                if next_cue["start"] <= cue["end"] + 1.2 and next_cue.get("speaker") != speaker:
                    has_rapid_banter = True
                    banter_end = max(banter_end, next_cue["end"])
                    j += 1
                else:
                    break

            if has_rapid_banter and (banter_end - cue["start"] >= self.min_shot_duration):
                shots.append({
                    "start": cue["start"],
                    "end": banter_end,
                    "camera": "split",
                    "type": "split_stack"
                })
                i = j
                continue

            # Solo speaker shot
            zone = self.config.get_zone_by_speaker(speaker)
            cam_id = zone.id if zone else (host_id if "00" in speaker or "host" in speaker.lower() else guest_id)

            shots.append({
                "start": cue["start"],
                "end": cue["end"],
                "camera": cam_id,
                "type": "single"
            })
            i += 1

        return shots

    def _post_process_cuts(self, cuts: List[Dict[str, Any]], clip_duration: float) -> List[Dict[str, Any]]:
        """
        Smooths shot cuts:
        - Fills gaps so timeline starts at 0.0 and ends at clip_duration.
        - Enforces minimum 2.0s shot duration (hysteresis).
        - Merges identical consecutive camera shots.
        """
        if not cuts:
            return [{"start": 0.0, "end": clip_duration, "camera": "zone_host", "type": "single"}]

        # Sort cuts
        cuts.sort(key=lambda c: c["start"])

        # Extend first shot to 0.0
        cuts[0]["start"] = 0.0

        # Close any internal gaps between cuts
        for k in range(len(cuts) - 1):
            cuts[k]["end"] = cuts[k + 1]["start"]

        # Extend last shot to clip_duration
        cuts[-1]["end"] = max(cuts[-1]["end"], clip_duration)

        # Merge shots shorter than min_shot_duration into neighbor
        merged: List[Dict[str, Any]] = []
        for shot in cuts:
            dur = shot["end"] - shot["start"]
            if merged and dur < self.min_shot_duration:
                # Merge into previous shot
                merged[-1]["end"] = shot["end"]
            else:
                merged.append(dict(shot))

        # Merge identical consecutive camera angles
        final_cuts: List[Dict[str, Any]] = []
        for shot in merged:
            if final_cuts and final_cuts[-1]["camera"] == shot["camera"] and final_cuts[-1]["type"] == shot["type"]:
                final_cuts[-1]["end"] = shot["end"]
            else:
                final_cuts.append(shot)

        # Final sanity clamp
        if final_cuts:
            final_cuts[0]["start"] = 0.0
            final_cuts[-1]["end"] = clip_duration

        return final_cuts

    def _resolve_zone_coordinates(self, cuts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolves concrete pixel coordinates and layout specifications for each shot."""
        host_zone = self.config.get_host_zone() or CameraZone("zone_host", "Host", 0, 0, 960, 1080)
        guest_zone = self.config.get_guest_zone() or CameraZone("zone_guest", "Guest", 960, 0, 960, 1080)
        wide_zone = self.config.get_wide_zone() or CameraZone("zone_wide", "Wide", 0, 0, self.config.source_width, self.config.source_height, is_wide=True)
        screencast_zone = self.config.get_screencast_zone()

        resolved = []
        for shot in cuts:
            shot_type = shot.get("type", "single")
            cam_id = shot.get("camera", "zone_host")

            # Screencast Stack layout: Facecam Top, Screen Content Bottom
            if (self.config.mode == "screencast" or screencast_zone is not None) and (shot_type == "screencast_stack" or cam_id == "screencast"):
                face_zone = host_zone if not host_zone.is_screencast else guest_zone
                resolved.append({
                    "start": round(shot["start"], 2),
                    "end": round(shot["end"], 2),
                    "type": "split_stack",
                    "camera": "screencast_stack",
                    "top_zone": face_zone.to_dict(),
                    "bot_zone": (screencast_zone or wide_zone).to_dict(),
                })
            elif shot_type == "split_stack" or cam_id == "split":
                resolved.append({
                    "start": round(shot["start"], 2),
                    "end": round(shot["end"], 2),
                    "type": "split_stack",
                    "camera": "split",
                    "top_zone": host_zone.to_dict(),
                    "bot_zone": guest_zone.to_dict(),
                })
            elif cam_id == "zone_wide" or shot_type == "wide":
                resolved.append({
                    "start": round(shot["start"], 2),
                    "end": round(shot["end"], 2),
                    "type": "wide",
                    "camera": "zone_wide",
                    "zone": wide_zone.to_dict(),
                })
            else:
                target_zone = self.config.get_zone(cam_id) or host_zone
                resolved.append({
                    "start": round(shot["start"], 2),
                    "end": round(shot["end"], 2),
                    "type": "single",
                    "camera": target_zone.id,
                    "zone": target_zone.to_dict(),
                })

        return resolved
