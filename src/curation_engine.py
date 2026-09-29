"""
src/curation_engine.py
----------------------
Phase 1: Real AI Curation Intelligence.

get_viral_cuts() now routes through score_transcript_with_llm() which:
  1. Spins up an Ollama daemon (killing stale instances first).
  2. Handles Kaggle model-weight caching via manifests-copy + blobs-symlink.
  3. Scores transcript windows with llama3.1:8b in JSON mode.
  4. Validates all output against validate_and_format_cuts().
  5. Unloads the model from VRAM after scoring.
  6. Falls back to even-split logic if Ollama is unreachable or JSON fails.
"""

import gc
import json
import math
import logging
import os
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("curation_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CLIP_DURATION_SEC = 30.0
MAX_CLIP_DURATION_SEC = 50.0
OLLAMA_MODEL       = "llama3.1:8b"
OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_PORT        = 11434
CHUNK_MINUTES      = 12        # 12-minute sliding window for long-video chunking
CHUNK_OVERLAP_SEC  = 120       # 2-minute overlap between chunks
MAX_RETRIES        = 2         # strict-prompt retry on JSON parse failure
_llm_cache = {}                # SHA-256 local cache for scored windows


# ---------------------------------------------------------------------------
# 1. validate_and_format_cuts  (unchanged contract)
# ---------------------------------------------------------------------------

def validate_and_format_cuts(raw_cuts: list, total_duration: float, min_duration: float = 20.0) -> list:
    """Validates raw cut dictionaries and populates unified key aliases."""
    validated = []
    for item in raw_cuts:
        if not isinstance(item, dict):
            continue
        s = float(item.get("start", item.get("start_time", 0.0)))
        try:
            td = float(total_duration)
            if math.isnan(td) or td <= 0:
                td = 1.0
        except (TypeError, ValueError):
            td = 1.0
        e = float(item.get("end", item.get("end_time", td)))
        s = max(0.0, s)
        e = max(s + 0.1, min(e, td))
        sentence = str(item.get("hook_sentence", item.get("hook_text", "")))
        try:
            v_score = int(float(item.get("virality_score", 85)))
            v_score = max(0, min(100, v_score))
        except (TypeError, ValueError):
            v_score = 85
        validated.append({
            "start":          s,
            "end":            e,
            "start_time":     s,
            "end_time":       e,
            "virality_score": v_score,
            "hook_sentence":  sentence,
            "hook_text":      sentence,
            "reason":         str(item.get("reason", "")),
        })
    # NMS Deduplication: Limit overlap to 15%, prioritize highest score
    # First, sort by virality score descending
    validated_sorted = sorted(validated, key=lambda x: x["virality_score"], reverse=True)
    
    deduped = []
    effective_min_dur = min(min_duration, max(1.0, float(total_duration) * 0.8))
    for c in validated_sorted:
        clip_len = c["end"] - c["start"]
        if clip_len < effective_min_dur:
            continue
            
        # Check overlap with already selected deduped reels
        overlap_violation = False
        for r in deduped:
            overlap_start = max(c["start"], r["start"])
            overlap_end = min(c["end"], r["end"])
            if overlap_start < overlap_end:
                overlap_dur = overlap_end - overlap_start
                if overlap_dur > clip_len * 0.15:
                    overlap_violation = True
                    break
                    
        if not overlap_violation:
            deduped.append(c)
            
    # Return sorted chronologically
    return sorted(deduped, key=lambda x: x["start"])


# ---------------------------------------------------------------------------
# 2. Ollama lifecycle helpers
# ---------------------------------------------------------------------------

def _port_in_use(port: int) -> bool:
    """Returns True if something is already bound to the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) == 0


_ollama_process: Optional[subprocess.Popen] = None

def _kill_stale_ollama() -> None:
    """Kills the specific child `ollama serve` process if we started it."""
    global _ollama_process
    if _ollama_process is not None:
        try:
            _ollama_process.terminate()
            _ollama_process.wait(timeout=5)
        except Exception as exc:
            logger.debug(f"[curation] kill child ollama: {exc}")
        finally:
            _ollama_process = None

import atexit
atexit.register(_kill_stale_ollama)


def _setup_ollama_model_cache() -> None:
    """
    Kaggle model-weight caching strategy:
      1. Look for a read-only dataset mount at /kaggle/input/ollama-* containing
         a blobs/ directory (Ollama content-addressed store).
      2. Copy only manifests/ (small, mutable) into a writable working dir.
      3. Symlink blobs/ from the read-only mount into that writable dir.
      4. Point OLLAMA_MODELS at the writable dir.
    Falls back silently (letting Ollama use ~/.ollama) on non-Kaggle or on error.
    """
    kaggle_input = "/kaggle/input"
    if not os.path.isdir(kaggle_input):
        return  # Not a Kaggle environment

    # Find any attached dataset that looks like an Ollama store
    dataset_root: Optional[str] = None
    for entry in os.listdir(kaggle_input):
        candidate = os.path.join(kaggle_input, entry)
        if os.path.isdir(os.path.join(candidate, "blobs")):
            dataset_root = candidate
            break

    if dataset_root is None:
        logger.info("[curation] No Ollama dataset mount found; will pull model fresh.")
        return

    working_models = "/kaggle/working/ollama_models"
    os.makedirs(working_models, exist_ok=True)

    try:
        # Copy manifests (small, needs write access)
        src_manifests = os.path.join(dataset_root, "manifests")
        dst_manifests = os.path.join(working_models, "manifests")
        if os.path.isdir(src_manifests) and not os.path.isdir(dst_manifests):
            shutil.copytree(src_manifests, dst_manifests)
            logger.info(f"[curation] Copied Ollama manifests -> {dst_manifests}")

        # Symlink blobs (large, read-only is fine)
        src_blobs = os.path.join(dataset_root, "blobs")
        dst_blobs = os.path.join(working_models, "blobs")
        if os.path.isdir(src_blobs) and not os.path.lexists(dst_blobs):
            os.symlink(src_blobs, dst_blobs)
            logger.info(f"[curation] Symlinked Ollama blobs {src_blobs} -> {dst_blobs}")

        os.environ["OLLAMA_MODELS"] = working_models
        logger.info(f"[curation] OLLAMA_MODELS set to {working_models}")

    except PermissionError as exc:
        logger.warning(f"[curation] Permission error setting up Ollama cache ({exc}); using default.")
    except Exception as exc:
        logger.warning(f"[curation] Ollama cache setup failed ({exc}); using default.")


def _start_ollama_server() -> bool:
    """
    Ensures a fresh Ollama server is running on port 11434.
    Returns True if server is ready, False otherwise.
    """
    global _ollama_process

    if _port_in_use(OLLAMA_PORT):
        logger.info("[curation] Port 11434 already in use — assuming external or existing Ollama is ready.")
        return True

    _setup_ollama_model_cache()

    try:
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"[curation] Ollama serve started (PID {_ollama_process.pid}). Waiting for port...")
    except FileNotFoundError:
        logger.warning("[curation] `ollama` binary not found. Falling back to even-split curation.")
        return False

    # Poll until ready (30s timeout)
    deadline = time.time() + 30
    while time.time() < deadline:
        if _port_in_use(OLLAMA_PORT):
            logger.info("[curation] Ollama server ready on port 11434.")
            return True
        time.sleep(0.8)

    logger.warning("[curation] Ollama server did not become ready within 30s.")
    return False


def _ensure_model_pulled(model: str = OLLAMA_MODEL) -> bool:
    """Pulls the model if not already available. Returns True on success."""
    try:
        import urllib.request
        data = json.dumps({"name": model}).encode()
        req  = urllib.request.Request(
            f"{OLLAMA_HOST}/api/pull",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info(f"[curation] Ensuring {model} is pulled (may take a few minutes)...")
        with urllib.request.urlopen(req, timeout=600) as resp:
            # Drain streaming response
            for line in resp:
                if b'\"status\"' in line:
                    pass  # Just drain
        return True
    except Exception as exc:
        logger.warning(f"[curation] Model pull failed: {exc}")
        return False


def _unload_model_from_vram(model: str = OLLAMA_MODEL) -> None:
    """Sends keep_alive=0 to evict the model from GPU memory."""
    try:
        import urllib.request
        data = json.dumps({"model": model, "keep_alive": 0}).encode()
        req  = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        logger.info(f"[curation] Model {model} unloaded from VRAM (keep_alive=0).")
    except Exception as exc:
        logger.debug(f"[curation] Model unload request failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# 3. LLM scoring helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a viral content strategist. Given a speech transcript with timestamps, identify the best short-form video clip segments (strictly 30 to 50 seconds each, with virality score >= 75).

Score each candidate segment on:
- HOOK_STRENGTH: Does the opening grab attention immediately?
- EMOTIONAL_PAYOFF: Is there a strong emotional arc or revelation?
- QUOTABILITY: Is the content memorable, shareable, tweet-worthy?

You MUST return a JSON object with a single key "clips" containing an array of objects. No conversational intro text, zero explanation, and no markdown code fences.
Each clip object must contain EXACTLY these keys: "start" (float), "end" (float), "viral_score" (integer 75-100), "title" (string), "hook_summary" (string), "virality_reason" (string).
"""

STRICT_SYSTEM_PROMPT = """You MUST return a JSON object with a single key "clips" containing an array of objects.
No conversational intro text, zero explanation, and no markdown code fences (do not use ```json).
Each clip object must have EXACTLY these keys: "start" (float), "end" (float), "viral_score" (integer 75-100), "title" (string), "hook_summary" (string), "virality_reason" (string).
Duration (end - start) must be strictly between 30 and 50 seconds.
If you cannot comply, return {"clips": []}."""


def _build_transcript_text(words: List[Dict], start_offset: float = 0.0) -> str:
    """Converts word list into readable timestamped transcript text for the LLM."""
    if not words:
        return ""
        
    sentences = []
    current_sentence_words = []
    sentence_start = float(words[0].get("start", 0.0)) + start_offset
    
    for i, w in enumerate(words):
        word_text = w.get("word", "").strip()
        current_sentence_words.append(word_text)
        
        # Check if word ends with punctuation or it's the last word
        if word_text.endswith(('.', '?', '!', ',', ';')) or i == len(words) - 1:
            sentence_end = float(w.get("end", float(w.get("start", 0.0)) + 0.1)) + start_offset
            sentences.append(f"[{sentence_start:.2f}s - {sentence_end:.2f}s] {' '.join(current_sentence_words)}")
            
            # Reset for next sentence
            if i + 1 < len(words):
                sentence_start = float(words[i + 1].get("start", 0.0)) + start_offset
                current_sentence_words = []
                
    return " ".join(sentences)


CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "duration": {"type": "number"},
                    "viral_score": {"type": "integer"},
                    "title": {"type": "string"},
                    "hook_summary": {"type": "string"},
                    "virality_reason": {"type": "string"}
                },
                "required": ["start", "end", "viral_score", "title", "hook_summary", "virality_reason"]
            }
        }
    },
    "required": ["clips"]
}

def _call_ollama(prompt: str, system: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """Calls Ollama generate endpoint, returns raw response text or None."""
    try:
        import urllib.request
        payload = json.dumps({
            "model":  model,
            "format": CURATION_SCHEMA,
            "stream": False,
            "system": system,
            "prompt": prompt,
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
            )
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = json.loads(resp.read().decode())
            return raw.get("response", "")
    except Exception as exc:
        logger.warning(f"[curation] Ollama call failed: {exc}")
        return None


def _parse_llm_json(text: str, total_duration: float) -> Optional[List[Dict]]:
    """
    Parses LLM response text as a JSON list of cut dicts.
    Extracts the array slice between first '[' and last ']',
    handles conversational wrappers, strips code fences, and normalizes key names.
    """
    if not text:
        return None
    try:
        import json
        import re

        clean_text = text.strip()
        # Remove any markdown code fences
        clean_text = re.sub(r'```(?:json|JSON)?', '', clean_text)
        clean_text = clean_text.replace('```', '').strip()

        start_idx_arr = clean_text.find('[')
        end_idx_arr = clean_text.rfind(']')
        start_idx_obj = clean_text.find('{')
        end_idx_obj = clean_text.rfind('}')

        parsed = None
        if start_idx_obj != -1 and end_idx_obj != -1 and end_idx_obj >= start_idx_obj:
            try:
                candidate = clean_text[start_idx_obj:end_idx_obj + 1]
                parsed_obj = json.loads(candidate)
                if isinstance(parsed_obj, dict) and "clips" in parsed_obj:
                    parsed = parsed_obj["clips"]
            except Exception:
                pass

        if parsed is None and start_idx_arr != -1 and end_idx_arr != -1 and end_idx_arr >= start_idx_arr:
            try:
                candidate = clean_text[start_idx_arr:end_idx_arr + 1]
                parsed = json.loads(candidate)
            except Exception:
                pass
                
        if not isinstance(parsed, list):
            # Fallback to direct parse if Regex extraction failed
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict) and "clips" in parsed:
                parsed = parsed["clips"]
                
        if not isinstance(parsed, list):
            return None

        valid = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            
            # Support both 'start'/'end' and 'start_time'/'end_time'
            s = float(item.get("start", item.get("start_time", -1)))
            e = float(item.get("end", item.get("end_time", -1)))

            if s < 0 or e <= s or s >= (total_duration + 0.1):
                logger.debug(f"[curation] Rejected hallucinated cut {s:.1f}-{e:.1f} (duration={total_duration:.1f})")
                continue

            if (e - s) < MIN_CLIP_DURATION_SEC:
                logger.debug(f"[curation] Rejected sub-minimum cut {s:.1f}-{e:.1f} ({e-s:.1f}s < {MIN_CLIP_DURATION_SEC}s minimum)")
                continue

            if (e - s) > MAX_CLIP_DURATION_SEC:
                logger.debug(f"[curation] Clamping cut {s:.1f}-{e:.1f} to {MAX_CLIP_DURATION_SEC}s")
                e = s + MAX_CLIP_DURATION_SEC
                if e > total_duration:
                    e = total_duration
                    if (e - s) < MIN_CLIP_DURATION_SEC:
                        continue

            # Align and filter virality score >= 75
            v_score = 85
            if "viral_score" in item:
                try:
                    v_score = int(float(item["viral_score"]))
                except (ValueError, TypeError):
                    v_score = 85
            elif "virality_score" in item:
                try:
                    v_score = int(float(item["virality_score"]))
                except (ValueError, TypeError):
                    v_score = 85
            elif "score" in item:
                try:
                    v_score = int(float(item["score"]))
                except (ValueError, TypeError):
                    v_score = 85

            if v_score < 75:
                logger.debug(f"[curation] Rejected low virality cut {s:.1f}-{e:.1f} (score {v_score} < 75)")
                continue

            item["start"] = s
            item["end"] = e
            item["start_time"] = s
            item["end_time"] = e
            item["virality_score"] = v_score
                
            if "title" in item:
                item["hook_sentence"] = item["title"]
            if "virality_reason" in item:
                item["reason"] = item["virality_reason"]
                
            valid.append(item)

        return valid

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug(f"[curation] JSON parse error: {exc}")
        return None


def _chunk_words(words: List[Dict], chunk_sec: float, overlap_sec: float) -> List[Tuple[float, float, List[Dict]]]:
    """
    Splits word list into overlapping time windows.
    Returns list of (window_start, window_end, words_in_window).
    """
    if not words:
        return []
    first_t = float(words[0].get("start", 0.0))
    last_t  = float(words[-1].get("end",   float(words[-1].get("start", 0.0)) + 0.1))
    total   = last_t - first_t

    if total <= chunk_sec:
        return [(first_t, last_t, words)]

    chunks = []
    win_start = first_t
    while win_start < last_t:
        win_end   = min(win_start + chunk_sec, last_t)
        win_words = [w for w in words if float(w.get("start", 0)) >= win_start - 1.0
                     and float(w.get("end",   0)) <= win_end   + 1.0]
        if win_words:
            chunks.append((win_start, win_end, win_words))
        if win_end >= last_t:
            break
        win_start += chunk_sec - overlap_sec
    return chunks


# ---------------------------------------------------------------------------
# 3b. Cloud Frontier LLM API Provider (Kimi K3, DeepSeek, Gemini, OpenAI)
# ---------------------------------------------------------------------------

def get_cloud_api_config() -> Optional[Dict[str, str]]:
    """
    Detects configured Cloud LLM API credentials from environment variables.
    Supports Kimi / Moonshot, DeepSeek, Gemini, OpenAI, or generic OpenAI-compatible endpoints.
    Priority order:
      1. Explicit LLM_API_KEY + optional LLM_BASE_URL / LLM_MODEL
      2. KIMI_API_KEY or MOONSHOT_API_KEY -> https://api.moonshot.cn/v1, moonshot-v1-128k
      3. DEEPSEEK_API_KEY -> https://api.deepseek.com/v1, deepseek-chat
      4. GEMINI_API_KEY -> https://generativelanguage.googleapis.com/v1beta/openai, gemini-2.0-flash
      5. OPENAI_API_KEY -> https://api.openai.com/v1, gpt-4o-mini
    """
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        return {
            "api_key": api_key,
            "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "provider": "custom_openai",
        }

    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if kimi_key:
        return {
            "api_key": kimi_key,
            "base_url": os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1").rstrip("/"),
            "model": os.environ.get("KIMI_MODEL", os.environ.get("MOONSHOT_MODEL", "moonshot-v1-128k")),
            "provider": "kimi",
        }

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        return {
            "api_key": deepseek_key,
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "provider": "deepseek",
        }

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return {
            "api_key": gemini_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            "provider": "gemini",
        }

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return {
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "provider": "groq",
        }

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "api_key": openrouter_key,
            "base_url": "https://openrouter.ai/api/v1",
            "model": os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
            "provider": "openrouter",
        }

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return {
            "api_key": openai_key,
            "base_url": "https://api.openai.com/v1",
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "provider": "openai",
        }

    return None


def _call_openai_compatible_api(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system: str,
    timeout: float = 120.0
) -> Optional[str]:
    """Sends a chat completion request to any OpenAI-compatible API endpoint using urllib."""
    import urllib.request
    import urllib.error

    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "AutonomousStudio/2.0",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                return message.get("content", "")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        logger.warning(f"[curation] Cloud API HTTP {e.code}: {e.reason} - {err_body[:200]}")
    except Exception as exc:
        logger.warning(f"[curation] Cloud API call failed: {exc}")

    return None


def score_transcript_with_api(
    words: List[Dict],
    silence_gaps: List,
    total_duration: float,
    api_config: Dict[str, str],
) -> List[Dict]:
    """
    Scores transcript using cloud frontier model API (Kimi K3, DeepSeek, Gemini, etc.)
    with a single whole-video pass or large window chunks.
    Takes 0 VRAM and executes in seconds.
    """
    if not words:
        logger.info("[curation] No words supplied; skipping Cloud API scoring.")
        return []

    provider = api_config.get("provider", "cloud_api")
    model = api_config.get("model", "unknown")
    base_url = api_config.get("base_url", "")
    api_key = api_config.get("api_key", "")

    logger.info(f"[curation] [API] Cloud Frontier API Active: Provider={provider}, Model={model}")
    print(f"[curation] [API] Cloud Frontier API Active: Provider={provider}, Model={model}")

    target_clips = max(3, min(15, int(round((total_duration / 3600.0) * 12.0))))
    transcript_text = _build_transcript_text(words)

    # Check cache by SHA-256
    import hashlib
    cache_key = hashlib.sha256((model + transcript_text).encode()).hexdigest()
    if cache_key in _llm_cache:
        logger.info(f"[curation] Cloud API cache hit for {model} (key: {cache_key[:12]})")
        print(f"[curation] Cloud API cache hit for {model}")
        return _llm_cache[cache_key]

    system_prompt = (
        "You are an elite short-form video viral content strategist (TikTok, YouTube Shorts, Reels).\n"
        "Your task is to identify the highest-retention, most viral 30-50 second moments from this video transcript.\n\n"
        "Evaluation criteria for virality:\n"
        "1. HOOK POWER: The first 3 seconds must provoke intense curiosity, present a bold counter-intuitive claim, or drop right into high drama/action.\n"
        "2. NARRATIVE COMPLETENESS: The soundbite must make sense standalone without external context.\n"
        "3. HIGH PAYOFF: Concludes with a punchline, valuable insight, or mind-blowing revelation.\n"
        "4. QUOTABILITY: High shareability and comment provocation.\n\n"
        "You MUST return a JSON object with a single key 'clips' containing an array of objects.\n"
        "Each clip object must have EXACTLY these keys:\n"
        "  - 'start': float (start timestamp in seconds)\n"
        "  - 'end': float (end timestamp in seconds)\n"
        "  - 'viral_score': integer (75 to 100)\n"
        "  - 'title': string (viral click-worthy headline)\n"
        "  - 'hook_summary': string (1-sentence summary of the hook)\n"
        "  - 'virality_reason': string (why this specific moment retains viewers)\n"
        "Duration (end - start) MUST be strictly between 30 and 50 seconds."
    )

    user_prompt = (
        f"Video total duration: {total_duration:.1f}s ({total_duration/60.0:.1f} minutes).\n\n"
        f"Timestamped Transcript:\n{transcript_text}\n\n"
        f"Identify the top {target_clips} most viral, high-retention segments across this entire video.\n"
        f"All timestamps must be between 0.0s and {total_duration:.1f}s.\n"
        f"Return strictly JSON formatted as {{\"clips\": [...]}}."
    )

    raw_response = _call_openai_compatible_api(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=user_prompt,
        system=system_prompt,
        timeout=120.0
    )

    if not raw_response:
        logger.warning(f"[curation] Cloud API call to {provider} returned empty response.")
        return []

    cuts = _parse_llm_json(raw_response, total_duration)
    if cuts:
        validated = validate_and_format_cuts(cuts, total_duration)
        logger.info(f"[curation] Cloud API {provider} selected {len(validated)} validated viral cuts.")
        print(f"[curation] Cloud API {provider} selected {len(validated)} validated viral cuts.")
        _llm_cache[cache_key] = validated
        return validated

    logger.warning(f"[curation] Failed to parse valid cuts from Cloud API {provider} response.")
    return []


# ---------------------------------------------------------------------------
# 4. Main scoring function (Local Ollama)
# ---------------------------------------------------------------------------

def score_transcript_with_llm(
    words: List[Dict],
    silence_gaps: List,
    total_duration: float,
    model: str = OLLAMA_MODEL,
) -> List[Dict]:
    """
    Scores transcript with Ollama LLM to find viral moments.
    Returns validated list of cut dicts, or [] if LLM unavailable.
    Falls back to even-split if Ollama unreachable or JSON fails.
    Caller is responsible for VRAM eviction of prior models before calling this.
    """
    _log_vram("before LLM curation")

    if not words:
        logger.info("[curation] No words supplied; skipping LLM scoring.")
        return []

    # Start Ollama server
    server_ok = _start_ollama_server()
    if not server_ok:
        logger.warning("[curation] Ollama unavailable — using even-split fallback.")
        return []

    # Ensure model is available
    model_ok = _ensure_model_pulled(model)
    if not model_ok:
        logger.warning("[curation] Model pull failed — using even-split fallback.")
        _unload_model_from_vram(model)
        return []

    # Chunk transcript for long videos
    chunk_sec = CHUNK_MINUTES * 60.0
    chunks = _chunk_words(words, chunk_sec, CHUNK_OVERLAP_SEC)
    logger.info(f"[curation] Scoring {len(chunks)} transcript window(s) with {model}.")

    all_cuts: List[Dict] = []

    import hashlib
    for win_start, win_end, chunk_words_list in chunks:
        # VAD Pruning check
        window_duration = win_end - win_start
        silence_in_window = 0.0
        for gap in silence_gaps:
            if not isinstance(gap, (list, tuple)) or len(gap) < 2:
                continue
            gap_s = max(win_start, float(gap[0]))
            gap_e = min(win_end, float(gap[1]))
            if gap_e > gap_s:
                silence_in_window += (gap_e - gap_s)
                
        if window_duration > 0 and (silence_in_window / window_duration) > 0.7:
            logger.info(f"[curation] Pruned window {win_start:.0f}-{win_end:.0f}s: >70% silence ({silence_in_window:.1f}s / {window_duration:.1f}s)")
            continue

        transcript_text = _build_transcript_text(chunk_words_list)
        
        # SHA-256 Cache
        window_hash = hashlib.sha256(transcript_text.encode()).hexdigest()
        if window_hash in _llm_cache:
            logger.info(f"[curation] Cache hit for window {win_start:.0f}-{win_end:.0f}s")
            cuts = _llm_cache[window_hash]
            all_cuts.extend(cuts)
            continue
            
        user_prompt = (
            f"Transcript window [{win_start:.0f}s – {win_end:.0f}s] "
            f"(video total: {total_duration:.0f}s):\n\n{transcript_text}\n\n"
            f"Identify the best 1-3 viral clip segments from this window only (each clip must be between 30 and 50 seconds long). "
            f"All timestamps must be between {win_start:.2f} and {win_end:.2f}."
        )

        raw = None
        for attempt in range(1, MAX_RETRIES + 1):
            sys_prompt = SYSTEM_PROMPT if attempt == 1 else STRICT_SYSTEM_PROMPT
            raw = _call_ollama(user_prompt, sys_prompt, model)
            cuts = _parse_llm_json(raw or "", total_duration)
            if cuts:
                logger.info(f"[curation] Window {win_start:.0f}-{win_end:.0f}s: {len(cuts)} cut(s) on attempt {attempt}.")
                all_cuts.extend(cuts)
                _llm_cache[window_hash] = cuts
                break
            elif cuts is not None:
                logger.warning(f"[curation] 0 valid cuts (attempt {attempt}/{MAX_RETRIES}) for window {win_start:.0f}-{win_end:.0f}s.")
            else:
                logger.warning(f"[curation] JSON parse failed (attempt {attempt}/{MAX_RETRIES}) for window {win_start:.0f}-{win_end:.0f}s.")

    # Evict model from GPU
    _unload_model_from_vram(model)
    _log_vram("after LLM curation (model unloaded)")

    if not all_cuts:
        logger.warning("[curation] LLM returned no valid cuts across all windows — using even-split fallback.")
        return []

    # Validate and dedup through existing contract function
    validated = validate_and_format_cuts(all_cuts, total_duration)
    logger.info(f"[curation] LLM curation complete: {len(validated)} validated segment(s).")
    return validated


# ---------------------------------------------------------------------------
# 5. VRAM telemetry helper
# ---------------------------------------------------------------------------

def _log_vram(label: str) -> None:
    """Logs torch CUDA memory allocated (safe on CPU machines)."""
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"[VRAM] {label}: {mb:.1f} MB allocated.")
            print(f"[VRAM] {label}: {mb:.1f} MB allocated.")
        else:
            logger.debug(f"[VRAM] {label}: no GPU — CPU-only mode.")
    except Exception:
        pass


def _audio_energy_fallback(video_path: str, target_clips: int) -> list:
    """Finds high-energy audio moments (RMS spikes) to center fallback clips around."""
    import tempfile
    import wave
    import struct
    import math
    
    spikes = []
    if not video_path or not os.path.isfile(video_path):
        return spikes
        
    temp_wav = None
    try:
        try:
            import audio_intelligence
        except ImportError:
            import src.audio_intelligence as audio_intelligence
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_wav = tf.name
            
        audio_intelligence.extract_audio(video_path, temp_wav)
        
        with wave.open(temp_wav, 'rb') as wf:
            rate = wf.getframerate()
            nframes = wf.getnframes()
            # Read in chunks of 1 second
            chunk_size = rate
            num_chunks = int(nframes / chunk_size)
            energies = []
            for i in range(num_chunks):
                raw = wf.readframes(chunk_size)
                # handle if raw is empty or short
                if len(raw) < 2: continue
                samples = struct.unpack(f"{len(raw)//2}h", raw)
                rms = math.sqrt(sum(s*s for s in samples) / len(samples)) if samples else 0
                energies.append((i, rms))
                
        # Find top N spikes, spaced by at least 15 seconds
        energies.sort(key=lambda x: x[1], reverse=True)
        for i, rms in energies:
            if all(abs(i - s) >= 15 for s in spikes):
                spikes.append(i)
                if len(spikes) >= target_clips:
                    break
    except Exception as e:
        logger.warning(f"[curation] Audio energy fallback failed: {e}")
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass
                
    spikes.sort()
    return spikes


def compute_audio_energy_profile(media_path: str) -> List[float]:
    """Computes a per-second RMS audio energy profile for multi-modal signal fusion."""
    if not media_path:
        return []
    target_path = media_path
    if not os.path.isfile(target_path):
        candidate = os.path.join("inputs", media_path)
        if os.path.isfile(candidate):
            target_path = candidate
        else:
            return []

    temp_wav = None
    try:
        if target_path.lower().endswith(".wav"):
            wav_file = target_path
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_wav = tf.name
            try:
                import audio_intelligence
            except ImportError:
                import src.audio_intelligence as audio_intelligence
            audio_intelligence.extract_audio(target_path, temp_wav)
            wav_file = temp_wav

        import wave, struct, math
        with wave.open(wav_file, 'rb') as wf:
            rate = wf.getframerate()
            nframes = wf.getnframes()
            chunk_size = rate
            if chunk_size <= 0:
                return []
            num_chunks = int(nframes / chunk_size)
            energies = []
            for _ in range(num_chunks):
                raw = wf.readframes(chunk_size)
                if len(raw) < 2:
                    break
                samples = struct.unpack(f"{len(raw)//2}h", raw)
                rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
                energies.append(rms)
            return energies
    except Exception as e:
        logger.debug(f"[curation] Audio energy profile generation failed: {e}")
        return []
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


def compute_visual_hook_score(media_path: str, start_t: float, end_t: float) -> Tuple[float, str]:
    """
    Evaluates visual dynamism and face presence in the first 2.0s of a candidate clip.
    Returns (score 0..100, tag).
    """
    if not media_path:
        return 75.0, ""
    target_path = media_path
    if not os.path.isfile(target_path):
        candidate = os.path.join("inputs", media_path)
        if os.path.isfile(candidate):
            target_path = candidate
        else:
            return 75.0, ""

    ext = os.path.splitext(target_path)[1].lower()
    if ext not in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
        return 75.0, ""

    try:
        import cv2
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            return 75.0, ""

        sample_times = [start_t + offset for offset in [0.0, 0.5, 1.0, 1.5] if (start_t + offset) <= end_t]
        frames = []
        for st in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, st * 1000.0))
            ret, frame = cap.read()
            if ret and frame is not None:
                small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
        cap.release()

        if len(frames) < 2:
            return 75.0, ""

        # 1. Motion Dynamics (Frame difference)
        diffs = []
        for i in range(len(frames) - 1):
            diff = cv2.absdiff(frames[i], frames[i + 1])
            diffs.append(float(diff.mean()))
        avg_motion = sum(diffs) / len(diffs) if diffs else 0.0

        # 2. Fast Face Presence Check
        face_detected = False
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.isfile(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                for f in frames:
                    faces = face_cascade.detectMultiScale(f, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
                    if len(faces) > 0:
                        face_detected = True
                        break
        except Exception:
            pass

        if avg_motion >= 6.0 and face_detected:
            return 90.0, "[Visual Hook: Dynamic Motion & Face Present]"
        elif avg_motion >= 4.0:
            return 82.0, "[Visual Hook: Active Motion]"
        elif face_detected:
            return 80.0, "[Visual Hook: Face Present]"
        elif avg_motion < 1.5:
            return 55.0, "[Visual Alert: Static Scene]"
        else:
            return 72.0, ""
    except Exception as e:
        logger.debug(f"[curation] Visual hook analysis failed: {e}")
        return 75.0, ""


def get_viral_cuts(
    words: list,
    silence_gaps: list,
    total_duration: float,
    file_name: str = "",
) -> list:
    """
    Main entry point for curation. Routes through tri-modal LLM + acoustic + visual
    scoring when words are available, then falls back to even-interval / RMS
    split on failure or empty words.
    Always runs validate_and_format_cuts() on the final output.
    """
    try:
        total_duration = float(total_duration)
        if math.isnan(total_duration) or total_duration <= 1.0:
            return []
    except (TypeError, ValueError):
        return []

    # Short clip: return the whole thing
    if total_duration <= 45.0:
        return [{
            "start": 0.0, "end": total_duration,
            "start_time": 0.0, "end_time": total_duration,
            "virality_score": 85,
            "hook_sentence": "Full Segment Clip",
            "reason": "Optimal standalone soundbite meeting duration threshold.",
        }]

    # Attempt Cloud Frontier API or local Ollama path when transcript words are available
    if words and len(words) >= 80:
        try:
            llm_cuts = []
            api_config = get_cloud_api_config()
            if api_config:
                llm_cuts = score_transcript_with_api(words, silence_gaps, total_duration, api_config)

            # Fall back to local Ollama if Cloud API returned no cuts or was not configured
            if not llm_cuts:
                llm_cuts = score_transcript_with_llm(words, silence_gaps, total_duration)

            if llm_cuts:
                # Tri-Modal Signal Fusion: Modulate LLM semantic virality with acoustic + visual dynamics
                if file_name:
                    energy_profile = compute_audio_energy_profile(file_name)
                    global_avg_rms = sum(energy_profile) / len(energy_profile) if energy_profile else 1.0
                    
                    for cut in llm_cuts:
                        c_start = float(cut.get("start", 0.0))
                        c_end = float(cut.get("end", 0.0))
                        
                        # 1. Acoustic Energy Modulation
                        if energy_profile:
                            s_sec = int(max(0, math.floor(c_start)))
                            e_sec = int(min(len(energy_profile), math.ceil(c_end)))
                            clip_rms = energy_profile[s_sec:e_sec]
                            if clip_rms:
                                clip_avg = sum(clip_rms) / len(clip_rms)
                                ratio = clip_avg / max(global_avg_rms, 1e-6)
                                if ratio >= 1.4:
                                    boost = 6
                                    tag = "[Acoustic Peak: High Vocal Energy]"
                                elif ratio >= 1.15:
                                    boost = 3
                                    tag = "[Acoustic Peak: Elevated Energy]"
                                elif ratio <= 0.65:
                                    boost = -5
                                    tag = "[Acoustic Dampener: Low Energy Pitch]"
                                else:
                                    boost = 0
                                    tag = ""
                                if tag:
                                    cut["reason"] = f"{cut.get('reason', '')} {tag}".strip()
                                orig_score = cut.get("virality_score", 85)
                                cut["virality_score"] = max(0, min(100, int(orig_score + boost)))

                        # 2. Visual Scene Dynamics & Face Hook Modulation
                        v_score, v_tag = compute_visual_hook_score(file_name, c_start, c_end)
                        if v_score >= 85:
                            cut["virality_score"] = min(100, cut.get("virality_score", 85) + 3)
                        elif v_score <= 60:
                            cut["virality_score"] = max(0, cut.get("virality_score", 85) - 3)
                        if v_tag:
                            cut["reason"] = f"{cut.get('reason', '')} {v_tag}".strip()

                    llm_cuts = validate_and_format_cuts(llm_cuts, total_duration)

                logger.info("[curation] Path: Tri-Modal LLM + Acoustic + Visual scoring.")
                print("[curation] Path: Tri-Modal LLM + Acoustic + Visual scoring.")
                return llm_cuts
        except Exception as exc:
            logger.warning(f"[curation] LLM path raised exception ({exc}); using fallback.")

    target_clips = max(3, int(round((total_duration / 3600.0) * 40.0)))
    
    # Try audio energy spike fallback first
    spikes = _audio_energy_fallback(file_name, target_clips)
    
    reels = []
    if spikes:
        logger.info("[curation] Path: Audio RMS energy spike fallback (no LLM / no words).")
        print("[curation] Path: Audio RMS energy spike fallback (no LLM / no words).")
        
        # NMS Deduplication: Limit overlap to 15%
        for i, spike_t in enumerate(spikes):
            start = round(max(0.0, float(spike_t) - 5.0), 2)
            clip_len = min(35.0, total_duration - start)
            if clip_len < MIN_CLIP_DURATION_SEC:
                start = max(0.0, total_duration - MIN_CLIP_DURATION_SEC)
                clip_len = min(MAX_CLIP_DURATION_SEC, total_duration - start)
            end = round(start + clip_len, 2)
            
            # Check overlap with already selected reels
            overlap_violation = False
            for r in reels:
                overlap_start = max(start, r["start"])
                overlap_end = min(end, r["end"])
                if overlap_start < overlap_end:
                    overlap_dur = overlap_end - overlap_start
                    if overlap_dur > clip_len * 0.15:
                        overlap_violation = True
                        break
                        
            if not overlap_violation:
                reels.append({
                    "start": start, "end": end,
                    "start_time": start, "end_time": end,
                    "virality_score": max(75, 95 - (len(reels) * 2)),
                    "hook_sentence": f"Action Hook #{len(reels) + 1}",
                    "reason": f"Blind fallback: Audio energy spike detected at {spike_t}s (no transcript analysis).",
                })
                
                if len(reels) >= target_clips:
                    break
                    
        return validate_and_format_cuts(reels, total_duration)
    else:
        # Even-split fallback
        logger.info("[curation] Path: even-split fallback (no LLM / no words).")
        print("[curation] Path: even-split fallback (no LLM / no words).")
        interval = total_duration / target_clips
        clip_len = min(MAX_CLIP_DURATION_SEC, max(MIN_CLIP_DURATION_SEC, min(35.0, interval)))
        for i in range(target_clips):
            start = round(i * interval, 2)
            end = round(min(total_duration, start + clip_len), 2)
            if end - start < MIN_CLIP_DURATION_SEC:
                start = max(0.0, end - MIN_CLIP_DURATION_SEC)
            reels.append({
                "start": start, "end": end,
                "start_time": start, "end_time": end,
                "virality_score": max(75, 95 - (i % 20)),
                "hook_sentence": f"Action Hook #{i + 1}",
                "reason": f"Blind fallback: Even-split interval {i + 1} (no transcript or audio signal).",
            })
            
    return validate_and_format_cuts(reels, total_duration)


def curate_video_with_warning(
    words: list,
    silence_gaps: list,
    total_duration: float,
    file_name: str = "",
    min_score: int = 75,
) -> dict:
    """
    Curates candidate clips with strict duration and volume constraints.
    Returns:
        clips: list of validated and deduplicated clips
        target_clips: targeted count of clips based on duration (~40 per hour)
        low_yield_warning: boolean flag if yield is under expected threshold
        message: status or warning explanation
    """
    try:
        td = float(total_duration)
        if math.isnan(td) or td <= 1.0:
            return {
                "clips": [],
                "target_clips": 0,
                "low_yield_warning": True,
                "message": "Invalid video duration (< 1.0s)."
            }
    except (TypeError, ValueError):
        return {
            "clips": [],
            "target_clips": 0,
            "low_yield_warning": True,
            "message": "Invalid video duration parameter."
        }

    if td <= 45.0:
        target_clips = 1
    else:
        target_clips = max(3, int(round((td / 3600.0) * 40.0)))

    all_cuts = get_viral_cuts(words, silence_gaps, td, file_name=file_name)

    filtered_clips = []
    for c in all_cuts:
        score = c.get("virality_score", 0)
        dur = c.get("end", 0) - c.get("start", 0)
        if score < min_score:
            continue
        if td > 45.0 and (dur < MIN_CLIP_DURATION_SEC - 0.5 or dur > MAX_CLIP_DURATION_SEC + 0.5):
            continue
        filtered_clips.append(c)

    # Sort clips descending by virality score for review
    filtered_clips.sort(key=lambda x: x.get("virality_score", 0), reverse=True)

    is_low_yield = False
    if target_clips > 0 and len(filtered_clips) < target_clips:
        is_low_yield = True
        msg = f"Only {len(filtered_clips)} high-quality viral hooks were found."
    else:
        msg = f"Successfully curated {len(filtered_clips)} clip(s) (target: {target_clips})."

    return {
        "clips": filtered_clips,
        "target_clips": target_clips,
        "low_yield_warning": is_low_yield,
        "message": msg
    }

