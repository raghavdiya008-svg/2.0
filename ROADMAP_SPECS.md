# Video Engine Roadmap & Architecture Specifications

This document outlines the detailed plans and technical specifications for upcoming video engine enhancements, covering typography, file naming, presentation layouts, and gaming workflows.

---

## 1. Headline Font with Real Personality

### Objective
Upgrade top hook headline banners from standard, sterile sans-serif text to a punchy, high-impact aesthetic suited for short-form social platforms.

### Technical Specification
1. **Typography Discovery & Hierarchy**:
   - Check `fonts/` for high-impact display fonts (`Montserrat-Black.ttf`, `Anton-Regular.ttf`, `Impact`, `Rubik-Black`).
   - If not found locally, bundle or resolve a dedicated ultra-bold display typeface specifically for hook banners rather than falling back to generic Arial.
2. **Visual Styling**:
   - **Contrast & Depth**: Add stroke and drop shadow (`borderw=3:bordercolor=black@0.9:shadowx=4:shadowy=4:shadowcolor=black@0.8`) to ensure legibility across bright or chaotic backgrounds.
   - **Pill / Badge Styling**: Render sleek semi-translucent dark pill badges with rounded padding or electric accent tags (`#FFE600` / `#00F0FF`).
   - **Keyword Highlighting**: Automatically parse strong hook keywords for accent color formatting.

---

## 2. Descriptive File Naming via JSON Metadata

### Objective
Ensure exported clip files and API JSON metadata share a unified, human-readable file name derived from the curated hook/title instead of arbitrary UUIDs or raw timestamp ranges.

### Technical Specification
1. **JSON Metadata as Single Source of Truth**:
   - In `curation_engine.py`, each candidate clip includes a sanitized slug and output filename:
     ```json
     {
       "clip_id": "clip_01",
       "headline": "How Free Tiers Fund Billions",
       "file_slug": "How_Free_Tiers_Fund_Billions_clip_784_827",
       "output_filename": "How_Free_Tiers_Fund_Billions_clip_784_827.mp4",
       "start": 784.0,
       "end": 827.0
     }
     ```
2. **Slug Sanitization Rules**:
   - Strip emojis, punctuation, and invalid filesystem characters (`!?,.:;'"/\`).
   - Truncate slug title to a maximum of 45 characters to avoid Windows `MAX_PATH` limitations.
   - Convert spaces to underscores `_` or PascalCase.
   - Format: `{sanitized_title}_clip_{int(start)}_{int(end)}.mp4`.
3. **Pipeline & Server Integration**:
   - `src/server.py` and `src/pipeline.py` write directly to `output_dir / output_filename`.
   - Zip exports, SQLite job entries (`pipeline.db`), and API responses reference this exact filename.

---

## 3. Screen-Share + 2 Speakers Hybrid Layout

### Objective
Gracefully handle webinar, podcast, and remote-pitch scenes where a slide/screen-share is active simultaneously while two speakers are talking.

### Technical Specification
1. **Detection**:
   - In `src/engine_vision.py`, detect high-detail static/presentation regions (slides) alongside two detected face regions.
2. **Layout Blueprint (9:16 canvas: 1080×1920)**:
   - **Top Section (Slide Focus: 1080×1080 or 1080×960)**:
     - 1:1 or 16:9 crop centered on the presentation slide so diagram and text elements remain fully readable.
   - **Bottom Section (Dual Speaker Split: 1080×960 or 1080×840)**:
     - Side-by-side split of Host & Guest:
       - Left pane (540×840): Speaker 1 close-up.
       - Right pane (540×840): Speaker 2 close-up.
3. **FFmpeg Filtergraph**:
   - Slide crop: `[0:v]crop=w=...:h=...[slide]`
   - Speaker 1 crop: `[0:v]crop=...[speaker1]`
   - Speaker 2 crop: `[0:v]crop=...[speaker2]`
   - Composite: `[speaker1][speaker2]hstack[speakers]`, followed by `[slide][speakers]vstack[out]`.
4. **Caption Clearance**:
   - Position subtitles in the visual gap between the slide and the speakers to avoid occluding presentation content.

---

## 4. Gaming Videos: With Facecam vs. Without Facecam

### Objective
Optimize 9:16 vertical conversion for gaming streams and gameplay recordings while preserving essential HUD elements.

### Technical Specification
1. **Gaming WITH Facecam**:
   - Refine `detect_gaming_layout()` in `src/engine_vision.py`.
   - Top pane (1080×720): Webcam feed centered on streamer facecam.
   - Bottom pane (1080×1200): Main gameplay focused on action / crosshair.
   - Clean accent divider bar between facecam and gameplay.
2. **Gaming WITHOUT Facecam (Pure Gameplay)**:
   - **Problem**: Naive 9:16 center cropping cuts off minimaps (top corners), killfeeds (top right), and health/ammo HUDs (bottom corners).
   - **Strategy A (HUD-Safe Dynamic Blur Backdrop)**:
     - 9:16 canvas filled with blurred background gameplay.
     - 16:9 main feed placed sharp and centered at 1080px width, keeping 100% of the game HUD intact.
   - **Strategy B (Action Zoom + Mini-Map Picture-in-Picture)**:
     - 9:16 action crop centered on player crosshair.
     - Detected minimap region cropped and overlaid at top-left or top-right of the vertical canvas.
