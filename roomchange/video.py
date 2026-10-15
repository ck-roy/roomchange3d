from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Settings


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int
    sharpness: float
    sampled_frames: list[np.ndarray]
    semantic_frames: list[np.ndarray]
    sampled_times: list[float]
    sampled_sharpness: list[float]
    warnings: list[str]

    def public_dict(self) -> dict:
        return {"path": Path(self.path).name, "width": self.width, "height": self.height,
                "fps": self.fps, "duration_seconds": self.duration_seconds,
                "frame_count": self.frame_count, "sharpness": self.sharpness,
                "sampled_times": list(self.sampled_times),
                "sampled_sharpness": list(self.sampled_sharpness),
                "warnings": list(self.warnings)}


def inspect_video(path: str | Path, settings: Settings = Settings()) -> VideoInfo:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Video is missing or empty: {path.name}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not read {path.name}. Try exporting it as MP4 (H.264).")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 160 or height < 120 or count < 2 or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video metadata is invalid or the video is too short to inspect.")
    duration = count / fps
    # Inspect up to twice as many temporal candidates as we will reconstruct,
    # then keep the sharpest frame in each time bin. This often avoids a blurred
    # frame without over-representing a single clear moment in the clip.
    n = min(settings.max_frames, count)
    candidate_count = min(count, max(n, n * 2))
    indices = np.unique(np.linspace(0, count - 1, candidate_count, dtype=int))
    candidates: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        semantic_scale = min(1.0, 1024 / max(h, w))
        semantic_frame = (cv2.resize(bgr, (round(w * semantic_scale), round(h * semantic_scale)), interpolation=cv2.INTER_AREA)
                          if semantic_scale < 1 else bgr.copy())
        scale = min(1.0, settings.max_side / max(h, w))
        if scale < 1:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        candidates.append((int(i), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(semantic_frame, cv2.COLOR_BGR2RGB), score, float(i / fps)))
    cap.release()
    if not candidates:
        raise ValueError(f"No decodable frames found in {path.name}.")

    # Divide the whole clip into n bins and choose the sharpest decoded frame
    # from each. Empty bins (for damaged or unusual variable-frame-rate videos)
    # are skipped safely.
    chosen = []
    for bin_id in range(n):
        lo = bin_id * count / n
        hi = (bin_id + 1) * count / n
        bin_frames = [x for x in candidates if lo <= x[0] < hi or (bin_id == n - 1 and x[0] == count - 1)]
        if bin_frames:
            chosen.append(max(bin_frames, key=lambda x: x[3]))
    frames = [x[1] for x in chosen]
    semantic_frames = [x[2] for x in chosen]
    sharpness = [x[3] for x in chosen]
    times = [x[4] for x in chosen]
    warnings: list[str] = []
    if duration < 4:
        warnings.append("Short clip: 6–20 seconds with a slow walk-around usually gives better coverage.")
    blurry_fraction = float(np.mean(np.asarray(sharpness) < 35))
    if float(np.median(sharpness)) < 35 or blurry_fraction >= 0.5:
        warnings.append("Many sampled frames still look soft or blurred after selecting the sharpest nearby frames. Record more slowly with steadier movement and brighter light; image sharpening cannot recover missing detail.")
    if min(width, height) < 360:
        warnings.append("Low resolution can reduce detail and reconstruction quality.")
    if len(frames) < settings.min_frames:
        warnings.append("Too few usable frames; record a longer clip with overlapping viewpoints.")
    return VideoInfo(str(path), width, height, fps, duration, count,
                     float(np.median(sharpness)), frames, semantic_frames, times, sharpness, warnings)

