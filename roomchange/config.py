from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # More temporal samples give small objects several fully-in-frame
    # observations even when the camera approaches them gradually.
    max_frames: int = 32
    min_frames: int = 5
    max_side: int = 518
    voxel_size: float = 0.025
    change_threshold: float = 0.08
    output_dir: Path = Path("outputs")

