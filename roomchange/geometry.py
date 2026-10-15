from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Registration:
    rotation: np.ndarray
    translation: np.ndarray
    inlier_fraction: float
    rmse: float
    success: bool


def transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return p @ rotation.T + translation


def transform_camera_extrinsics(rotations: np.ndarray, translations: np.ndarray,
                                scale: float, world_rotation: np.ndarray,
                                world_translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map world-to-camera extrinsics through x' = scale * Q*x + u."""
    camera_rotations = np.asarray(rotations, dtype=float) @ np.asarray(world_rotation, dtype=float).T
    camera_translations = (np.asarray(translations, dtype=float) * scale
                           - camera_rotations @ np.asarray(world_translation, dtype=float))
    return camera_rotations, camera_translations


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform mapping source to target (no scale)."""
    if len(source) < 3 or len(target) != len(source):
        raise ValueError("Rigid fitting requires at least three paired points.")
    a, b = np.asarray(source, float), np.asarray(target, float)
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    # Column-vector rotation compatible with transform(points) = points @ R.T.
    r = vt.T @ correction @ u.T
    return r, bc - ac @ r.T


def icp_register(source: np.ndarray, target: np.ndarray, max_iterations: int = 30,
                 max_distance: float = 0.15, initial_rotation: np.ndarray | None = None,
                 initial_translation: np.ndarray | None = None) -> Registration:
    """Robust trimmed point-to-point ICP; returns an explicit low-confidence failure."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    eye = np.eye(3)
    if len(a) < 30 or len(b) < 30:
        return Registration(eye, np.zeros(3), 0.0, float("inf"), False)
    # Centering is only an initialization; ICP resolves residual rigid motion.
    r = eye if initial_rotation is None else np.asarray(initial_rotation, float)
    t = b.mean(0) - a.mean(0) if initial_translation is None else np.asarray(initial_translation, float)
    tree = cKDTree(b)
    fraction = 0.0
    rmse = float("inf")
    for _ in range(max_iterations):
        moved = transform(a, r, t)
        d, ix = tree.query(moved, k=1, workers=-1)
        valid = d < max_distance
        if valid.sum() < 30:
            break
        cutoff = np.quantile(d[valid], 0.8)
        valid &= d <= cutoff
        dr, dt = rigid_fit(moved[valid], b[ix[valid]])
        r = dr @ r
        t = dr @ t + dt
        fraction = float(valid.mean())
        rmse = float(np.sqrt(np.mean(d[valid] ** 2)))
    success = fraction >= 0.12 and np.isfinite(rmse) and rmse < max_distance * 0.65
    return Registration(r, t, fraction, rmse, success)


def pca_multistart_register(source: np.ndarray, target: np.ndarray,
                            max_distance: float = 0.15) -> Registration:
    """PCA-initialized multi-start ICP for unknown camera orientation; scale must be normalized first."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if len(a) < 30 or len(b) < 30:
        return Registration(np.eye(3), np.zeros(3), 0.0, float("inf"), False)
    def basis(x):
        _, vec = np.linalg.eigh(np.cov(x.T))
        return vec[:, ::-1]
    qa, qb = basis(a), basis(b)
    best = None
    for perm in permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            r = qb @ p @ np.diag(signs) @ qa.T
            if np.linalg.det(r) < 0.999:
                continue
            t = b.mean(0) - r @ a.mean(0)
            candidate = icp_register(a, b, max_iterations=18, max_distance=max_distance,
                                      initial_rotation=r, initial_translation=t)
            score = (0 if candidate.success else 1, candidate.rmse - 0.05 * candidate.inlier_fraction)
            if best is None or score < best[0]:
                best = (score, candidate)
    return best[1] if best else Registration(np.eye(3), np.zeros(3), 0.0, float("inf"), False)


def voxel_downsample(points: np.ndarray, colors: np.ndarray | None = None,
                     voxel: float = 0.025) -> tuple[np.ndarray, np.ndarray | None]:
    p = np.asarray(points, float)
    if len(p) == 0:
        return p.reshape((-1, 3)), None if colors is None else np.asarray(colors)[:0]
    keys = np.floor(p / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    out = np.column_stack([np.bincount(inverse, weights=p[:, k]) / count for k in range(3)])
    c_out = None
    if colors is not None:
        c = np.asarray(colors, float)
        c_out = np.column_stack([np.bincount(inverse, weights=c[:, k]) / count for k in range(c.shape[1])])
    return out, c_out


def synthetic_scene_pair(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic unit-test fixture: static cube room, one moved box."""
    rng = np.random.default_rng(seed)
    # Room plane samples are stable shared support; changed object clouds are distinct.
    room = rng.uniform(-1, 1, (1200, 3))
    room[:, 2] = np.round(room[:, 2] * 2) / 2
    box_a = rng.normal([-0.35, 0, 0.15], [0.07, 0.07, 0.07], (180, 3))
    box_b = rng.normal([0.45, 0.1, 0.15], [0.07, 0.07, 0.07], (180, 3))
    return np.vstack([room, box_a]), np.vstack([room, box_b])


def classify_unmatched(unmatched_count: int, observed_both: bool, confidence: float) -> str:
    if not observed_both or confidence < 0.45 or unmatched_count < 8:
        return "uncertain"
    return "added_or_removed"


def surface_unmatched(source: np.ndarray, target: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return source points with no target surface within threshold and their nearest distances."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if len(a) == 0:
        return a.reshape((-1, 3)), np.empty((0,), dtype=float)
    if len(b) == 0:
        return a, np.full(len(a), np.inf)
    distance = cKDTree(b).query(a, k=1, workers=-1)[0]
    mask = distance > threshold
    return a[mask], distance[mask]


def project_points_to_views(points: np.ndarray, point_maps: np.ndarray,
                           confidence_maps: np.ndarray, camera_rotations: np.ndarray,
                           camera_translations: np.ndarray, intrinsics: np.ndarray,
                           max_views: int = 12,
                           confidence_quantile: float = 0.25,
                           pixel_radius: int = 2) -> dict:
    """Measure whether world points are supported/visible in a set of VGGT views.

    VGGT world_points are generated by lifting every pixel through its predicted
    depth into a shared world frame. Nearest point-map projection acts as an
    approximate forward camera model without assuming a metric focal length.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    maps = np.asarray(point_maps, dtype=np.float64)
    conf = np.asarray(confidence_maps, dtype=np.float64)
    rotations = np.asarray(camera_rotations, dtype=np.float64)
    translations = np.asarray(camera_translations, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    # Normalize singleton batch dimensions emitted by some model/version combos.
    if maps.ndim == 5 and maps.shape[0] == 1:
        maps = maps[0]
    if conf.ndim == 4 and conf.shape[0] == 1:
        conf = conf[0]
    if rotations.ndim == 4 and rotations.shape[0] == 1:
        rotations = rotations[0]
    if translations.ndim == 3 and translations.shape[0] == 1:
        translations = translations[0]
    if intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics[0]
    if maps.ndim != 4 or maps.shape[-1] != 3 or conf.shape != maps.shape[:3]:
        raise ValueError("Point maps/confidence must have shapes (views,H,W,3) and (views,H,W).")
    if rotations.shape != (maps.shape[0], 3, 3) or translations.shape != (maps.shape[0], 3) or intrinsics.shape != (maps.shape[0], 3, 3):
        raise ValueError("Camera calibration must contain one rotation, translation, and intrinsic matrix per view.")
    n = maps.shape[0]
    ids = np.unique(np.linspace(0, n - 1, min(max_views, n), dtype=int)) if n else np.array([], int)
    # Bounds use confident finite point support, not the recording-specific scene origin.
    flat_conf = conf.reshape(-1)
    finite_conf = flat_conf[np.isfinite(flat_conf)]
    cutoff = np.quantile(finite_conf, confidence_quantile) if len(finite_conf) else np.inf
    view_clouds = []
    for i in ids:
        m = maps[i].reshape(-1, 3)
        c = conf[i].reshape(-1)
        valid = np.isfinite(m).all(1) & np.isfinite(c) & (c >= cutoff)
        view_clouds.append(m[valid])
    observed = np.zeros(len(pts), dtype=bool)
    observations = np.zeros(len(pts), dtype=np.int16)
    edge_exposure = np.zeros(len(pts), dtype=bool)
    from scipy.spatial import cKDTree
    # Vectorized nearest pixel-to-point-map reprojection for up to 12 views.
    for local, view in enumerate(ids):
        cloud = view_clouds[local]
        if len(cloud) < 10 or not len(pts):
            continue
        r, t = rotations[view], translations[view]
        # VGGT pose encoding's extrinsic is world-to-camera: x_cam = R*x_world+t.
        cam_pts = pts @ r.T + t
        z = cam_pts[:, 2]
        front = np.isfinite(cam_pts).all(1) & (z > 1e-5)
        if not front.any():
            continue
        # Use the predicted intrinsics to project into this frame, then compare
        # candidate depth against the z-buffer formed by the confident point map.
        k = intrinsics[view]
        uvw = cam_pts @ k.T
        uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
        height, width = maps.shape[1:3]
        visible_here = np.zeros(len(pts), dtype=bool)
        # Restrict to coordinates whose rounded nearest pixel remains in bounds.
        in_frame = (front & (uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5) &
                    (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5))
        edge_margin = max(1, min(8, int(round(min(height, width) * 0.04))))
        interior = (in_frame & (uv[:, 0] >= edge_margin) & (uv[:, 0] < width - edge_margin) &
                    (uv[:, 1] >= edge_margin) & (uv[:, 1] < height - edge_margin))
        edge_exposure |= in_frame & ~interior
        ys, xs = np.where(np.isfinite(conf[view]) & (conf[view] >= cutoff) & np.isfinite(maps[view]).all(2))
        if not len(xs):
            continue
        zbuffer = np.full((height, width), np.inf, dtype=np.float64)
        map_cam = maps[view, ys, xs] @ r.T + t
        good = np.isfinite(map_cam).all(1) & (map_cam[:, 2] > 1e-5)
        zbuffer[ys[good], xs[good]] = map_cam[good, 2]
        # Fill tiny holes only; a large absent patch is not visibility evidence.
        if pixel_radius > 0:
            from scipy.ndimage import minimum_filter
            zbuffer = minimum_filter(zbuffer, size=2 * pixel_radius + 1, mode="constant", cval=np.inf)
        candidate_indices = np.flatnonzero(interior)
        px = np.rint(uv[candidate_indices, 0]).astype(int)
        py = np.rint(uv[candidate_indices, 1]).astype(int)
        scene_scale = np.maximum(0.02, np.abs(cam_pts[candidate_indices, 2]) * 0.025)
        # Empty pixel means observed free space only when no nearer surface occludes it.
        depths = zbuffer[py, px]
        visible_here[candidate_indices] = (~np.isfinite(depths) |
            (cam_pts[candidate_indices, 2] <= depths + scene_scale))
        observed |= visible_here
        observations += visible_here.astype(np.int16)
    return {"observed": observed, "observation_count": observations,
            "edge_exposure": edge_exposure,
            "sampled_view_indices": ids.tolist()}

