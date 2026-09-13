#!/usr/bin/env python3
"""
src/engine_watermark.py
-----------------------
Blurred Watermark & Logo Detector for Pre-Render Rejection (Phase 5).

Detects artificial, localized blur patches (Gaussian/box/mosaic blur) placed over
platform watermarks (e.g., TikTok, Instagram Reels, YouTube Shorts logos) in corner
Regions of Interest (ROI).

Rejects offending videos before expensive AI pipeline stages (WhisperX, YOLO, LLM)
to avoid platform shadowbans and wasted compute time.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class WatermarkRejectionError(Exception):
    """Raised when an artificial blurred watermark or logo patch is detected."""
    pass


# Corner ROIs relative to canvas dimensions (ymin, ymax, xmin, xmax as fractions)
WATERMARK_CORNER_ROIS = {
    "top_left": (0.0, 0.25, 0.0, 0.35),
    "top_right": (0.0, 0.25, 0.65, 1.0),
    "bottom_left": (0.75, 1.0, 0.0, 0.35),
    "bottom_right": (0.75, 1.0, 0.65, 1.0),
}


def _compute_laplacian_variance(image_gray: np.ndarray) -> float:
    """Computes Laplacian variance as a proxy for high-frequency edge energy."""
    if image_gray.size == 0 or image_gray.shape[0] < 3 or image_gray.shape[1] < 3:
        return 0.0
    return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())


def _is_suspicious_blur_patch(
    roi_gray: np.ndarray,
    frame_variance: float,
    block_size: int = 40,
    stride: int = 20,
    max_patch_var: float = 22.0,
    min_surrounding_var: float = 65.0,
    min_contrast_ratio: float = 3.5,
) -> bool:
    """
    Analyzes an ROI corner for an artificial, localized blur patch.

    An artificial watermark blur is characterized by:
    1. A localized block or cluster of blocks with very low edge variance (< max_patch_var).
    2. The surrounding ROI context having sharp, high-frequency content (> min_surrounding_var).
    3. A steep sharpness contrast ratio between the surrounding area and the patch.
    4. Not being a solid border/letterbox bar (excludes pure black/white flat blocks).
    """
    h, w = roi_gray.shape
    if h < block_size or w < block_size:
        return False

    roi_variance = _compute_laplacian_variance(roi_gray)
    # If the entire ROI is already flat/blurry (natural bokeh, uniform sky, flat wall),
    # there is no localized artificial blur island.
    if roi_variance < min_surrounding_var:
        return False

    grid_h = (h - block_size) // stride + 1
    grid_w = (w - block_size) // stride + 1
    if grid_h <= 0 or grid_w <= 0:
        return False

    block_vars = []
    block_means = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            y = gy * stride
            x = gx * stride
            block = roi_gray[y : y + block_size, x : x + block_size]
            mean_val = float(np.mean(block))
            # Ignore solid letterbox / pillarbox bars (near-black or near-white)
            if mean_val < 15.0 or mean_val > 242.0:
                continue

            b_var = _compute_laplacian_variance(block)
            block_vars.append(b_var)
            block_means.append(mean_val)

    if not block_vars:
        return False

    min_bvar = min(block_vars)
    max_bvar = max(block_vars)
    median_bvar = float(np.median(block_vars))

    # Count how many blocks qualify as dead-blur zones
    blur_blocks = [v for v in block_vars if v <= max_patch_var]
    total_valid_blocks = len(block_vars)

    if not blur_blocks:
        return False

    blur_fraction = len(blur_blocks) / total_valid_blocks

    # An artificial watermark blur typically occupies 5% to 45% of the corner ROI.
    # If blur_fraction > 0.75, the entire corner is naturally out-of-focus (bokeh).
    if 0.03 <= blur_fraction <= 0.60:
        # Check contrast against the sharpest or median context in the same ROI
        surrounding_energy = max(median_bvar, max_bvar * 0.5)
        if surrounding_energy >= min_surrounding_var:
            ratio = surrounding_energy / max(1e-5, min_bvar)
            if ratio >= min_contrast_ratio:
                return True

    return False


def detect_blurred_logos(
    video_path: str,
    sample_interval_sec: float = 2.0,
    max_frames: int = 15,
    min_persistent_frames: int = 2,
    min_frame_variance: float = 40.0,
) -> bool:
    """
    Scans video_path for suspicious, artificial blur patches in standard watermark zones.

    Samples frames every `sample_interval_sec` up to `max_frames`.
    Evaluates each corner ROI (top-left, top-right, bottom-left, bottom-right).
    Requires temporal persistence (patch detected in the same corner across >= min_persistent_frames)
    to prevent false positives from transient camera movement or moving textures.

    Returns:
        True if an artificial blurred watermark is detected, False otherwise.
    """
    if not video_path or not os.path.isfile(video_path):
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(sample_interval_sec * fps)))

    corner_hit_counts: Dict[str, int] = {corner: 0 for corner in WATERMARK_CORNER_ROIS}
    analyzed_frames = 0
    current_frame_idx = 0

    try:
        while analyzed_frames < max_frames and (total_frames <= 0 or current_frame_idx < total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            if h < 100 or w < 100:
                current_frame_idx += frame_step
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_var = _compute_laplacian_variance(gray)

            # Skip whole-frame blurry, dark, or transition frames
            if frame_var < min_frame_variance:
                current_frame_idx += frame_step
                continue

            analyzed_frames += 1

            for corner_name, (ymin_f, ymax_f, xmin_f, xmax_f) in WATERMARK_CORNER_ROIS.items():
                y1 = int(round(ymin_f * h))
                y2 = int(round(ymax_f * h))
                x1 = int(round(xmin_f * w))
                x2 = int(round(xmax_f * w))

                roi = gray[y1:y2, x1:x2]
                if _is_suspicious_blur_patch(roi, frame_variance=frame_var):
                    corner_hit_counts[corner_name] += 1
                    # If this corner hits the required persistence threshold, return True early
                    if corner_hit_counts[corner_name] >= min_persistent_frames:
                        print(
                            f"[engine_watermark] Detected persistent blurred watermark in {corner_name} "
                            f"(hits={corner_hit_counts[corner_name]} / frames={analyzed_frames})."
                        )
                        return True

            current_frame_idx += frame_step

    finally:
        cap.release()

    # Final check across corners
    for corner, hits in corner_hit_counts.items():
        if hits >= min_persistent_frames:
            print(f"[engine_watermark] Confirmed blurred watermark in {corner} (hits={hits}).")
            return True

    return False
