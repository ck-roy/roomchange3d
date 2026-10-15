import numpy as np

from roomchange.geometry import classify_unmatched, icp_register, pca_multistart_register, project_points_to_views, rigid_fit, surface_unmatched, synthetic_scene_pair, transform, transform_camera_extrinsics, voxel_downsample


def test_rigid_fit_round_trip():
    rng = np.random.default_rng(3)
    p = rng.normal(size=(100, 3))
    angle = 0.3
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    t = np.array([0.4, -0.7, 0.2])
    rr, tt = rigid_fit(p, transform(p, r, t))
    assert np.allclose(transform(p, rr, tt), transform(p, r, t), atol=1e-6)


def test_camera_extrinsics_follow_similarity_transform():
    theta = 0.4
    q = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    camera_r = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=float)
    camera_t = np.array([[0.3, -0.2, 0.5]])
    u, scale = np.array([0.8, -0.1, 0.2]), 1.7
    new_r, new_t = transform_camera_extrinsics(camera_r, camera_t, scale, q, u)
    world_point = np.array([[0.2, 0.4, 2.0]])
    mapped_world = transform(world_point * scale, q, u)
    camera_from_mapped = mapped_world @ new_r[0].T + new_t[0]
    expected = (world_point @ camera_r[0].T + camera_t[0]) * scale
    assert np.allclose(camera_from_mapped, expected, atol=1e-7)


def test_icp_register_static_cloud():
    rng = np.random.default_rng(5)
    source = rng.uniform(-1, 1, (800, 3))
    theta = 0.12
    r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    target = transform(source, r, [0.2, 0.1, -0.15])
    result = icp_register(source, target, max_distance=0.4)
    assert result.success
    assert result.rmse < 0.15


def test_pca_multistart_unknown_orientation():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(250, 3)) * [1.0, 0.65, 0.3]
    angle = 0.7
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    target = transform(source, r, [0.1, -0.2, 0.15])
    result = pca_multistart_register(source, target, max_distance=0.35)
    assert result.success
    assert result.rmse < 0.12


def test_synthetic_fixture_has_moved_object():
    before, after = synthetic_scene_pair()
    assert before.shape[1] == after.shape[1] == 3
    assert len(before) > 1000 and len(after) > 1000
    removed, _ = surface_unmatched(before, after, threshold=0.15)
    added, _ = surface_unmatched(after, before, threshold=0.15)
    assert len(removed) >= 8 and len(added) >= 8


def test_empty_target_is_all_unmatched_but_empty_source_is_safe():
    p = np.ones((4, 3))
    unmatched, distances = surface_unmatched(p, np.empty((0, 3)), 0.1)
    assert unmatched.shape == (4, 3) and np.isinf(distances).all()
    unmatched, distances = surface_unmatched(np.empty((0, 3)), p, 0.1)
    assert unmatched.shape == (0, 3) and distances.shape == (0,)


def test_voxel_downsample_empty_and_colors():
    points, colors = voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3)))
    assert points.shape == (0, 3) and colors.shape == (0, 3)


def test_low_evidence_is_uncertain():
    assert classify_unmatched(100, False, 0.9) == "uncertain"
    assert classify_unmatched(100, True, 0.2) == "uncertain"
    assert classify_unmatched(3, True, 0.9) == "uncertain"


def test_visibility_projection_detects_observed_empty_and_offscreen_unknown():
    # Simple orthographic-like camera, point maps cover a surface at z=2.
    h, w = 40, 40
    yy, xx = np.mgrid[:h, :w]
    maps = np.stack([np.stack([(xx - 20) / 10, (yy - 20) / 10, np.full_like(xx, 2)], axis=-1)] * 3).astype(float)
    confidence = np.ones((3, h, w), dtype=float)
    rotations = np.repeat(np.eye(3)[None], 3, axis=0)
    translations = np.zeros((3, 3))
    intrinsics = np.repeat(np.array([[[10, 0, 20], [0, 10, 20], [0, 0, 1]]], dtype=float), 3, axis=0)
    result = project_points_to_views(np.array([[0.2, 0.2, 2.0], [10, 0, 2.0]]), maps, confidence,
                                    rotations, translations, intrinsics)
    assert result["observed"].tolist() == [True, False]
    assert result["observation_count"][0] == 3


def test_visibility_near_frame_edge_is_not_empty_space_evidence():
    h, w = 40, 40
    yy, xx = np.mgrid[:h, :w]
    maps = np.stack([np.stack([(xx - 20) / 10, (yy - 20) / 10, np.full_like(xx, 2)], axis=-1)])
    result = project_points_to_views(np.array([[-3.8, 0, 2.0]]), maps, np.ones((1, h, w)),
        np.eye(3)[None], np.zeros((1, 3)),
        np.array([[[10, 0, 20], [0, 10, 20], [0, 0, 1]]], dtype=float))
    assert result["edge_exposure"].tolist() == [True]
    assert result["observed"].tolist() == [False]


def test_visibility_occlusion_keeps_candidate_unknown():
    from roomchange.pipeline import _clusters

    hidden = np.random.default_rng(12).normal([0.3, 0, 0], [0.01, 0.01, 0.01], (40, 3))
    no_visibility = {"observed": np.zeros(40, dtype=bool), "observation_count": np.zeros(40, dtype=int)}
    findings = _clusters(hidden, "added", 0.6, no_visibility)
    assert findings
    assert all(f["change_type"] == "uncertain" for f in findings)
    assert all(f["candidate_side"] == "added" for f in findings)


def test_visibility_projection_accepts_singleton_batch_camera_calibration():
    h, w, n = 8, 8, 3
    yy, xx = np.mgrid[:h, :w]
    one_map = np.stack([(xx - 4) / 4, (yy - 4) / 4, np.full_like(xx, 2)], axis=-1)
    maps = np.repeat(one_map[None, None], n, axis=1)  # (batch=1, views, H, W, 3)
    confidence = np.ones((1, n, h, w))
    rotations = np.repeat(np.eye(3)[None, None], n, axis=1)
    translations = np.zeros((1, n, 3))
    intrinsics = np.repeat(np.array([[[[4, 0, 4], [0, 4, 4], [0, 0, 1]]]], dtype=float), n, axis=1)
    result = project_points_to_views(np.array([[0, 0, 2.0]]), maps, confidence,
                                    rotations, translations, intrinsics)
    assert result["observed"].tolist() == [True]
    assert result["observation_count"].tolist() == [3]

