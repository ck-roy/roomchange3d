import numpy as np
import pytest

from roomchange.video import inspect_video
from roomchange.config import Settings
from roomchange.semantics import geolocate_detections
from roomchange.pipeline import (_apply_blur_quality_gate, _confirm_one_sided_object_events,
                                _confirmed_object_events,
                                _label_findings_from_persistent_tracks,
                                _reconcile_moved_object_surfaces,
                                _withhold_untracked_surface_events,
                                _write_html, _write_ply, write_results_bundle)


def test_streamlit_arrow_conversion_with_pinned_stack():
    # Reproduce the table conversion path that previously crashed in the app.
    import streamlit.dataframe_util as dataframe_util
    payload = [{"change": "added", "position": [0.1, 0.2, 0.3], "distance": None}]
    encoded = dataframe_util.convert_anything_to_arrow_bytes(payload)
    assert encoded


def test_previous_run_alignment_metrics_pass_legacy_ui_gate():
    import json
    from pathlib import Path

    report_path = Path("outputs/run_20261013_205946/report.json")
    if not report_path.exists():
        pytest.skip("Local user footage report is not present in clean checkouts.")
    alignment = json.loads(report_path.read_text(encoding="utf-8"))["alignment"]
    max_rmse = alignment.get("maximum_rmse_scene_units", max(
        2.0 * alignment.get("change_threshold_scene_units", 0.08), 0.1))
    assert alignment.get("status") == "ok" or (
        alignment.get("success", False)
        and alignment.get("inlier_fraction", 0) >= alignment.get("minimum_overlap_fraction", 0.25)
        and alignment.get("rmse_scene_units", float("inf")) <= max_rmse)


def test_corrupt_video_fails_actionably(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a movie")
    with pytest.raises(ValueError):
        inspect_video(path)


def test_blur_quality_gate_withholds_event_classifications():
    from types import SimpleNamespace
    findings = [{"change_type": "added", "event_status": "observed", "evidence": "visible",
                 "candidate_side": "added"}]
    comparisons = [{"label": "chair", "status": "moved", "evidence": {}}]
    videos = [SimpleNamespace(warnings=[]), SimpleNamespace(warnings=["Frames remain soft or blurred."])]
    assert _apply_blur_quality_gate(findings, comparisons, videos)
    assert findings[0]["change_type"] == "uncertain"
    assert findings[0]["candidate_change_type"] == "added"
    assert comparisons[0]["status"] == "uncertain"
    assert comparisons[0]["candidate_status"] == "moved"


def test_added_object_requires_consecutive_track_and_visible_empty_geometry():
    comparison = {"label": "chair", "status": "uncertain", "candidate_side": "B",
                  "position_before": None, "position_after": [1, 2, 3],
                  "evidence": {"after_consecutive_frames": 4, "after_interior_consecutive_frames": 4}}
    finding = {"semantic_label": "chair", "candidate_side": "added", "change_type": "added",
               "event_status": "observed empty in opposite recording", "position_before": None,
               "position_after": [1.01, 2, 3]}
    _confirm_one_sided_object_events([comparison], [finding])
    assert comparison["status"] == "added"
    assert "consecutive" in comparison["evidence"]["reason"]

    comparison["status"] = "uncertain"
    _confirm_one_sided_object_events([comparison], [])
    assert comparison["status"] == "uncertain"


def test_same_category_on_both_sides_cannot_be_confirmed_as_removed():
    comparison = {"label": "bottle", "status": "uncertain", "candidate_side": "A",
                  "position_before": [0.45, -0.32, 0.90], "position_after": None,
                  "evidence": {"before_consecutive_frames": 6, "before_interior_consecutive_frames": 4}}
    finding = {"semantic_label": "bottle", "candidate_side": "removed", "change_type": "removed",
               "event_status": "observed empty in opposite recording", "position_before": [0.45, -0.32, 0.90]}
    detections_b = [{"label": "bottle", "frame_index": i, "score": 0.9} for i in range(10, 16)]
    _confirm_one_sided_object_events([comparison], [finding], [], detections_b)
    assert comparison["status"] == "uncertain"


def test_unconfirmed_detector_tracks_are_hidden_from_results():
    rows = [{"label": "banana", "status": "uncertain"},
            {"label": "bottle", "status": "moved"},
            {"label": "book", "status": "removed"}]
    assert [x["label"] for x in _confirmed_object_events(rows)] == ["bottle", "book"]


def test_moved_object_suppresses_false_removed_and_added_surface_fragments():
    findings = [
        {"semantic_label": "bottle", "candidate_side": "removed", "change_type": "removed",
         "position_before": [0.40, -0.34, 0.87], "position_after": None},
        {"semantic_label": "bottle", "candidate_side": "added", "change_type": "uncertain",
         "position_before": None, "position_after": [-0.15, -0.16, 1.10]},
        {"semantic_label": "chair", "candidate_side": "removed", "change_type": "removed",
         "position_before": [2, 2, 2], "position_after": None},
    ]
    comparison = {"label": "bottle", "status": "moved",
                  "position_before": [0.45, -0.32, 0.90], "position_after": [-0.12, -0.17, 0.88]}
    visible, reconciled = _reconcile_moved_object_surfaces(findings, [comparison])
    assert [x["semantic_label"] for x in visible] == ["chair"]
    assert len(reconciled) == 2
    assert all(x["reconciled_as"] == "moved" for x in reconciled)


def test_one_frame_semantic_detection_cannot_label_surface_change():
    findings = [{"semantic_label": "backpack", "candidate_side": "added", "change_type": "added",
                 "event_status": "observed empty in opposite recording", "position_before": None,
                 "position_after": [1.0, 2.0, 3.0], "evidence": "visible"}]
    _label_findings_from_persistent_tracks(findings, [])
    _withhold_untracked_surface_events(findings, [])
    assert findings[0]["semantic_label"] == "unidentified object/region"
    assert findings[0]["change_type"] == "uncertain"


def test_persistent_track_can_name_nearby_surface_and_support_event():
    findings = [{"candidate_side": "added", "change_type": "added", "event_status": "observed empty in opposite recording",
                 "position_before": None, "position_after": [1.0, 2.0, 3.0], "evidence": "visible"}]
    tracks = [{"label": "bag", "status": "added", "candidate_side": "B",
               "position_before": None, "position_after": [1.01, 2.0, 3.0]}]
    _label_findings_from_persistent_tracks(findings, tracks)
    _withhold_untracked_surface_events(findings, tracks)
    assert findings[0]["semantic_label"] == "bag"
    assert findings[0]["change_type"] == "added"


def test_missing_video_fails(tmp_path):
    with pytest.raises(ValueError):
        inspect_video(tmp_path / "missing.mov")


def test_video_inspection_keeps_higher_resolution_semantic_frames(tmp_path):
    import cv2
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360))
    assert writer.isOpened()
    yy, xx = np.mgrid[:360, :640]
    for index in range(40):
        checker = (((xx // 8 + yy // 8 + index) % 2) * 220 + 20).astype(np.uint8)
        writer.write(np.dstack([checker, np.roll(checker, 3, axis=1), checker]))
    writer.release()
    info = inspect_video(path, Settings(max_frames=4, max_side=300))
    assert len(info.sampled_frames) == len(info.semantic_frames) == 4
    assert max(info.sampled_frames[0].shape[:2]) <= 300
    assert max(info.semantic_frames[0].shape[:2]) == 640
    assert info.sampled_times == sorted(info.sampled_times)
    assert len(info.sampled_sharpness) == 4


def test_detector_box_is_geolocated_from_confident_point_map():
    maps = np.zeros((1, 12, 16, 3), dtype=np.float32)
    yy, xx = np.mgrid[:12, :16]
    maps[0, :, :, 0] = xx
    maps[0, :, :, 1] = yy
    maps[0, :, :, 2] = 2
    conf = np.ones((1, 12, 16), dtype=np.float32)
    detection = {"frame_index": 0, "label": "chair", "score": 0.9, "box_xyxy": [2, 2, 8, 10]}
    located = geolocate_detections([detection], maps, conf, [np.zeros((12, 16, 3), dtype=np.uint8)])
    assert located[0]["label"] == "chair"
    assert np.allclose(located[0]["position_scene_units"], [4.5, 5.5, 2])


def test_ply_and_results_bundle_export(tmp_path):
    cloud = tmp_path / "cloud.ply"
    _write_ply(cloud, np.array([[1, 2, 3]], dtype=np.float32), np.array([[20, 40, 60]], dtype=np.uint8))
    text = cloud.read_text(encoding="ascii")
    assert "element vertex 1" in text and "1.000000 2.000000 3.000000 20 40 60" in text
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "private_input.mp4").write_bytes(b"private")
    bundle = write_results_bundle(tmp_path)
    import zipfile
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"cloud.ply", "report.json"}


def test_scene_view_includes_labeled_object_event_locations(tmp_path):
    path = tmp_path / "scene.html"
    _write_html(path,
                np.array([[0, 0, 0]], dtype=float), np.array([[10, 20, 30]], dtype=np.uint8),
                np.array([[1, 1, 1]], dtype=float), np.array([[40, 50, 60]], dtype=np.uint8),
                [], [{"label": "laptop", "status": "moved",
                      "position_before": [0.1, 0.2, 0.3], "position_after": [0.5, 0.2, 0.3]}],
                np.zeros((0, 3)), np.zeros((0, 3)))
    html = path.read_text(encoding="utf-8")
    assert "Laptop \\u00b7 moved" in html
    assert "laptop \\u00b7 before" in html and "laptop \\u00b7 after" in html

