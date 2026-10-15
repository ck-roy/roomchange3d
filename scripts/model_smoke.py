"""Load the official VGGT checkpoint and run synthetic point-map extraction."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roomchange.reconstruction import reconstruct_vggt
from roomchange.video import VideoInfo


def main() -> None:
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 255, (518, 518, 3), dtype=np.uint8) for _ in range(2)]
    video = VideoInfo("synthetic-only", 518, 518, 30.0, 2.0, 60, 100.0,
                      frames, frames, [0.0, 1.0], [100.0, 100.0], [])
    points, colors, metadata, maps, confidence, depths = reconstruct_vggt(video)
    print(f"Synthetic point-map smoke succeeded: {len(points):,} filtered points, map={maps.shape}, confidence={confidence.shape}, depth={depths.shape}, metadata={metadata}", flush=True)


if __name__ == "__main__":
    main()
