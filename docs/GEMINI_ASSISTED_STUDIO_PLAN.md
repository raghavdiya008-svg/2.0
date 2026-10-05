# Blueprint: Gemini-Powered Autonomous Video Studio (2.0)

## 1. Executive Summary & Architecture Overview
This blueprint establishes a hybrid workflow that combines **Gemini's multimodal full-video comprehension** with our **Kaggle GPU-accelerated rendering and Virtual TV Director pipeline**. 

By delegating macro-intelligence (identifying 40–45 viral hooks across a 1-hour video in batches of 10) to Gemini via YouTube URL, we:
1. Eliminate heavy local/in-container LLM inference (Ollama `llama3.2:3b`), reclaiming ~5 GB VRAM and cutting 10 minutes of transcript chunking per run.
2. Ensure exact, predictable volume without arbitrary quality dropouts ("Quality Fallback Notice").
3. Retain WhisperX word-level subtitle alignment and hardware FFmpeg 9:16 rendering with camera tracking and split-screen director logic on Kaggle.

---

## 2. Core Operational Workflow

```
+-----------------------------------------------------------------------------------------+
|                                    1. MACRO SELECTION                                   |
|   You provide YouTube URL to Gemini -> Gemini segments 1-hour video into 4 Quarters     |
|   Gemini outputs Batch 1 (10 clips), Batch 2 (10 clips), ..., Batch 5 on demand         |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                  2. STUDIO INGESTION & UI                               |
|   Web UI Studio: Ingest video URL (cached in inputs/)                                   |
|   "Import Gemini Batch" Modal: Paste Batch JSON / timestamps                            |
|   Appends 10 cards to Review Gallery with dynamic In/Out trim scrubbers                 |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                              3. HYBRID CAMERA FRAMING & PRESETS                         |
|   Camera Studio: Auto-detects presets (Host, Guest, Screencast/Presentation)           |
|   Human-in-the-Loop: Review, adjust bounding boxes, test live 9:16 vertical crop monitor|
|   Saved to Vault: Presets persisted to memory/camera_zones/{video}.json                 |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                               4. KAGGLE GPU BATCH RENDERING                             |
|   Audio Cache: WhisperX transcribes full audio ONCE (MD5 hash cached)                   |
|   TV Director: Cuts between Speaker A / Speaker B / 9:8 Split Stack / Screencast Stack  |
|   Captions & Banners: Karaoke yellow glow subtitles + Top hook title bar                |
|   VRAM Flush: torch.cuda.empty_cache() after batch completion (No session restart needed|
+-----------------------------------------------------------------------------------------+
```

---

## 3. Implementation Plan & Action Items

### Step 1: Web UI Enhancement (`src/templates/index.html`)
1. **"Import Gemini Batch" Modal**:
   - Add button `[+] Import Gemini Batch` in the Review Gallery toolbar.
   - Modal contains a multi-line input box accepting:
     - Strict JSON arrays `[{"title": "...", "start": 600, "end": 640, ...}]`
     - Plain text timestamps `04:15 - 04:55 | Hook Title` or `1. [12:30 - 13:10] Title`
   - Option to **Append to Gallery** (building up 10 -> 20 -> 30 -> 45) or **Replace Gallery**.
2. **Camera Studio: Screencast / Presentation Preset**:
   - Add `+ Screencast / Slides` preset button:
     - Zone 1: Screencast (1920x1080 display)
     - Zone 2: Speaker Facecam (upper/corner crop)
   - Layout mode selection: `Podcast (Dual Speaker)` vs `Screencast Demo (Facecam + Screen Stack)`.

### Step 2: Server & Director Pipeline Updates (`src/server.py` & `src/pipeline.py`)
1. **Screencast / Presentation Filtergraph**:
   - Support `screencast_stack` in FFmpeg filtergraph: 1080x960 Facecam top, 1080x960 Screen content bottom.
2. **Batch Render In-Session Memory Guard**:
   - Explicit `torch.cuda.empty_cache()` and `gc.collect()` at end of each batch queue execution.

### Step 3: Verification & Test Coverage
1. Execute unit tests on timestamp parser with both JSON and raw string formats.
2. Verify camera framing serialization for screencast zones.

### Step 4: Senior Developer Gemini Production Prompt
1. Deliver the 1-page senior-grade prompt with full context, batch querying instructions, negative constraints, and schema specification.
