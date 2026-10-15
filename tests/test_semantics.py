from roomchange.semantics import (DEFAULT_OPEN_VOCAB_LABELS, _box_iou, _canonical_label,
                                  _imagenet_label_support, _suppress_nested_component_detections,
                                  compare_detected_objects)


def item(frame, position, edge=False):
    return {"frame_index": frame, "label": "chair", "score": 0.9,
            "position_scene_units": position, "touches_frame_edge": edge}


def test_repeated_nearby_category_tracks_are_unchanged():
    before = [item(0, [0, 0, 0]), item(1, [0.02, 0, 0]), item(2, [0.01, 0, 0])]
    after = [item(0, [0.04, 0, 0]), item(1, [0.03, 0, 0]), item(2, [0.05, 0, 0])]
    result = compare_detected_objects(before, after)
    assert len(result) == 1
    assert result[0]["status"] == "unchanged"


def test_single_well_supported_category_track_can_be_reported_moved():
    before = [item(0, [0, 0, 0]), item(1, [0.01, 0, 0]), item(2, [0.02, 0, 0])]
    after = [item(0, [0.8, 0, 0]), item(1, [0.81, 0, 0]), item(2, [0.79, 0, 0])]
    result = compare_detected_objects(before, after)
    assert len(result) == 1
    assert result[0]["status"] == "moved"


def test_edge_clipped_track_cannot_support_moved_or_unchanged_claim():
    before = [item(i, [0.45, -0.32, 0.90], edge=True) for i in range(1, 7)]
    after = [item(i, [-0.12, -0.17, 0.88]) for i in range(14, 20)]
    result = compare_detected_objects(before, after)
    assert result
    assert all(x["status"] == "uncertain" for x in result)


def test_two_frame_edge_only_detections_are_omitted():
    before = [item(0, [0, 0, 0], edge=True), item(1, [0.01, 0, 0], edge=True)]
    after = [item(0, [0.8, 0, 0], edge=True), item(1, [0.81, 0, 0], edge=True)]
    result = compare_detected_objects(before, after)
    assert result == []


def test_one_sided_detector_track_is_retained_as_uncertain():
    before = [item(0, [0, 0, 0]), item(1, [0.01, 0, 0]), item(2, [0.02, 0, 0])]
    result = compare_detected_objects(before, [])
    assert len(result) == 1
    assert result[0]["label"] == "chair"
    assert result[0]["status"] == "uncertain"
    assert result[0]["candidate_side"] == "A"
    assert result[0]["evidence"]["before_consecutive_frames"] == 3


def test_multiple_same_category_tracks_are_compared_individually():
    before = [item(i, [0.005 * i, 0, 0]) for i in range(3)] + [item(i, [1 + 0.005 * i, 0, 0]) for i in range(3)]
    after = [item(i, [0.02 + 0.005 * i, 0, 0]) for i in range(3)] + [item(i, [1.8 + 0.005 * i, 0, 0]) for i in range(3)]
    result = compare_detected_objects(before, after)
    assert [x["status"] for x in result] == ["unchanged", "moved"]
    assert all(x["label"] == "chair" for x in result)


def test_nonconsecutive_detections_do_not_count_as_a_persistent_track():
    before = [item(0, [0, 0, 0]), item(2, [0.01, 0, 0]), item(4, [0.02, 0, 0])]
    after = [item(0, [0.01, 0, 0]), item(2, [0.02, 0, 0]), item(4, [0.03, 0, 0])]
    result = compare_detected_objects(before, after)
    assert result == []


def test_imagenet_verifier_rejects_laptop_like_book_guess():
    assert not _imagenet_label_support("book", ["hard disc", "notebook", "laptop", "binder"])
    assert _imagenet_label_support("book", ["comic book", "book jacket"])
    assert _imagenet_label_support("banana", ["banana", "lemon"])
    assert not _imagenet_label_support("banana", ["lemon", "orange"])


def test_open_vocabulary_aliases_and_box_overlap():
    assert _canonical_label("backpack") == "bag"
    assert _canonical_label("handbag") == "bag"
    assert "bag" in DEFAULT_OPEN_VOCAB_LABELS
    assert not any("crocs" in label.lower() or "clog" in label.lower() for label in DEFAULT_OPEN_VOCAB_LABELS)
    assert "shoe" in DEFAULT_OPEN_VOCAB_LABELS
    assert "remote control" in DEFAULT_OPEN_VOCAB_LABELS
    detections = [
        {"label": "laptop", "frame_index": 0, "box_xyxy": [0, 0, 100, 100]},
        {"label": "keyboard", "frame_index": 0, "box_xyxy": [10, 60, 90, 95]},
        {"label": "keyboard", "frame_index": 1, "box_xyxy": [110, 60, 190, 95]},
    ]
    assert [x["label"] for x in _suppress_nested_component_detections(detections)] == ["laptop", "keyboard"]
    assert _box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    assert _box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0


def test_open_vocabulary_track_still_needs_three_consecutive_frames():
    def owl_item(frame):
        return {**item(frame, [0, 0, 0]), "label": "bag", "score": 0.31,
                "detector_source": "OWLv2 zero-shot"}
    result = compare_detected_objects([owl_item(0), owl_item(1)], [])
    assert result == []
    result = compare_detected_objects([owl_item(0), owl_item(1), owl_item(2)], [])
    assert len(result) == 1
    assert result[0]["label"] == "bag"
    assert result[0]["status"] == "uncertain"
    low_score = [{**owl_item(i), "score": 0.24} for i in (0, 1, 2)]
    assert compare_detected_objects(low_score, []) == []


def test_temporal_association_keeps_one_object_track_through_smooth_3d_drift():
    observations = [
        {**item(frame, [0.1 * frame, 0, 0]), "label": "bag", "score": 0.45,
         "detector_source": "OWLv2 zero-shot"}
        for frame in range(7)
    ]
    result = compare_detected_objects([], observations)
    assert len(result) == 1
    assert result[0]["label"] == "bag"
    assert result[0]["evidence"]["after_frames"] == list(range(7))
    assert result[0]["evidence"]["after_consecutive_frames"] == 7
