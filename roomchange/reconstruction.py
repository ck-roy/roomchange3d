from __future__ import annotations

import logging
import os
import numpy as np

from .video import VideoInfo

log = logging.getLogger(__name__)


def reconstruct_vggt(video: VideoInfo, progress=None) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Run official VGGT point-map branch. Returns camera/world point-map samples and RGB."""
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as exc:
        raise RuntimeError("VGGT is not installed. Run START_HERE.bat to prepare the local models.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.environ.get("ROOMCHANGE_VGGT_MODEL", "facebook/VGGT-1B").strip()
    if not model_id:
        model_id = "facebook/VGGT-1B"
    if progress:
        progress(f"Loading {model_id} weights (first run downloads the checkpoint)…")
    # checkpoint has a non-commercial license; details in THIRD_PARTY_NOTICES.md
    model = VGGT.from_pretrained(model_id).to(device).eval()
    frames = video.sampled_frames
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    # Official VGGT preprocessing pads/resizes to its patch multiple.
    h, w = tensor.shape[-2:]
    scale = min(1.0, 518 / max(h, w))
    nh, nw = int(h * scale) // 14 * 14, int(w * scale) // 14 * 14
    tensor = torch.nn.functional.interpolate(tensor, (nh, nw), mode="bilinear", align_corners=False)
    tensor = tensor.to(device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    try:
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda", dtype=dtype):
                    pred = model(tensor)
            else:
                pred = model(tensor)
    except torch.cuda.OutOfMemoryError as exc:
        if device != "cuda":
            raise RuntimeError("VGGT ran out of memory. Reduce the frame sampling limit or close GPU-heavy applications.") from exc
        log.warning("VGGT CUDA memory limit reached; retrying this recording on CPU.")
        torch.cuda.empty_cache()
        model = model.to("cpu")
        tensor = tensor.cpu()
        device = "cpu"
        with torch.inference_mode():
            pred = model(tensor)
    # VGGT's point_map is in a common predicted world frame for input images.
    points = pred["world_points"].float().cpu().numpy() if "world_points" in pred else pred["world_points"]
    if points.ndim == 5:
        points = points[0]
    conf = pred.get("world_points_conf")
    if conf is not None:
        conf = conf.float().cpu().numpy()
        if conf.ndim == 4:
            conf = conf[0]
        mask = conf > np.quantile(conf, 0.25)
    else:
        conf = np.ones(points.shape[:-1], dtype=np.float32)
        mask = np.isfinite(points).all(-1)
    point_maps = points.copy()
    confidence_maps = conf.copy()
    depth_maps = pred["depth"].float().cpu().numpy()[0, ..., 0]
    color = np.stack(frames)
    if color.shape[1:3] != points.shape[1:3]:
        import cv2
        color = np.stack([cv2.resize(x, (points.shape[2], points.shape[1])) for x in color])
    p = points[mask]
    c = color[mask]
    finite = np.isfinite(p).all(1) & (np.linalg.norm(p, axis=1) < 100)
    p, c = p[finite], c[finite]
    # VGGT expects (height, width). Passing these in reverse swaps the focal
    # lengths and principal point, so visibility projections land off-image.
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(points.shape[1], points.shape[2]))
    extrinsics = extrinsics[0].float().cpu().numpy()
    intrinsics = intrinsics[0].float().cpu().numpy()
    camera_positions = np.stack([-pose[:, :3].T @ pose[:, 3] for pose in extrinsics])
    camera_rotations = extrinsics[:, :, :3]
    camera_translations = extrinsics[:, :, 3]
    del model, pred, tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    if len(p) < 300:
        raise RuntimeError("VGGT produced too few confident 3D points to compare safely.")
    license_note = ("CC-BY-NC-4.0 checkpoint; research/non-commercial use only" if model_id == "facebook/VGGT-1B"
                    else "Use subject to the selected checkpoint's upstream license and access terms")
    return p.astype(np.float32), c.astype(np.uint8), {"model": model_id, "device": device,
              "license": license_note, "point_count": len(p),
              "camera_positions": camera_positions.tolist(), "camera_rotations": camera_rotations.tolist(),
              "camera_translations": camera_translations.tolist(), "camera_intrinsics": intrinsics.tolist()}, point_maps.astype(np.float32), confidence_maps.astype(np.float32), depth_maps.astype(np.float32)

