from __future__ import annotations

import csv
import json
import logging
import platform
import zipfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import Settings
from .geometry import pca_multistart_register, project_points_to_views, surface_unmatched, transform, transform_camera_extrinsics, voxel_downsample
from .reconstruction import reconstruct_vggt
from .semantics import (DEFAULT_OPEN_VOCAB_LABELS, _canonical_label,
                        compare_detected_objects, detect_objects, geolocate_detections,
                        open_vocabulary_status)
from .video import inspect_video

log = logging.getLogger(__name__)


def _clusters(points: np.ndarray, side: str, confidence: float,
              visibility: dict | None = None, min_observations: int = 2) -> list[dict]:
    """Cluster unmatched geometry into regions; conservatively label without 3D instance IDs."""
    if len(points) == 0:
        return []
    # Voxel sampling keeps hierarchical clustering bounded.
    raw = np.asarray(points, dtype=float)
    voxel_keys, inverse = np.unique(np.floor(raw / 0.06).astype(np.int64), axis=0, return_inverse=True)
    p, _ = voxel_downsample(raw, voxel=0.06)
    if len(p) < 5:
        return []
    pairs = cKDTree(p).query_pairs(r=0.14, output_type="ndarray")
    if len(pairs):
        rows = np.r_[pairs[:, 0], pairs[:, 1], np.arange(len(p))]
        cols = np.r_[pairs[:, 1], pairs[:, 0], np.arange(len(p))]
        graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(p), len(p))).tocsr()
        _, groups = connected_components(graph, directed=False)
    else:
        groups = np.arange(len(p))
    findings = []
    visibility = visibility or {"observed": np.zeros(len(raw), dtype=bool), "observation_count": np.zeros(len(raw), dtype=int)}
    raw_observed = np.asarray(visibility.get("observed", np.zeros(len(raw), bool)), dtype=bool)
    raw_edges = np.asarray(visibility.get("edge_exposure", np.zeros(len(raw), bool)), dtype=bool)
    # Carry visibility evidence through voxel reduction. Previously cluster indices
    # referred to the reduced cloud but were applied to the unreduced evidence array.
    voxel_count = np.bincount(inverse, minlength=len(voxel_keys))
    voxel_visible = np.bincount(inverse, weights=raw_observed.astype(float), minlength=len(voxel_keys))
    voxel_edge = np.bincount(inverse, weights=raw_edges.astype(float), minlength=len(voxel_keys))
    visible_fraction_by_voxel = voxel_visible / np.maximum(voxel_count, 1)
    edge_fraction_by_voxel = voxel_edge / np.maximum(voxel_count, 1)
    voxel_observed = visible_fraction_by_voxel >= 0.5
    voxel_edges = edge_fraction_by_voxel >= 0.5
    for group in np.unique(groups):
        pts = p[groups == group]
        if len(pts) < 8:
            continue
        center = pts.mean(0)
        indices = np.flatnonzero(groups == group)
        observed_count = int(np.count_nonzero(voxel_observed[indices]))
        visible_fraction = observed_count / len(indices)
        edge_fraction = float(np.mean(voxel_edges[indices]))
        confirmed = visible_fraction >= 0.5 and observed_count >= min_observations and edge_fraction < 0.5
        event = side if confirmed else "uncertain"
        findings.append({"object_or_region": "unidentified changed region", "change_type": event,
                         "candidate_side": side, "event_status": "observed empty in opposite recording" if confirmed else "uncertain; opposite recording did not provide sufficient visibility evidence",
                         "position_before": center.tolist() if side == "removed" else None,
                         "position_after": center.tolist() if side == "added" else None,
                         "displacement": None, "units": "relative scene units (monocular scale ambiguous)",
                         "confidence": round(float(confidence * (0.7 + 0.3 * visible_fraction)), 3), "point_count": int(len(pts)),
                         "visibility_evidence": {"observed_fraction": round(visible_fraction, 3), "supporting_points": observed_count,
                                                 "frame_edge_fraction": round(edge_fraction, 3)},
                         "evidence": (f"Surface is absent from aligned opposite cloud and {observed_count}/{len(indices)} candidate voxels have interior-frame visibility support; visibility is approximate." if confirmed else f"Surface is absent from aligned opposite cloud, but visibility is inconclusive ({observed_count}/{len(indices)} visible voxels; {edge_fraction:.0%} of candidate voxels project near a frame edge). Viewpoint, crop edges or occlusion may explain the difference."),
                         "identity": "not established"})
    return findings


def _apply_blur_quality_gate(findings: list[dict], comparisons: list[dict], videos) -> bool:
    """Downgrade event claims when either recording remains materially blurred."""
    blurred = any(any("soft or blurred" in warning.lower() for warning in video.warnings)
                  for video in videos)
    if not blurred:
        return False
    for item in comparisons:
        if item.get("status") != "uncertain":
            item["candidate_status"] = item["status"]
            item["status"] = "uncertain"
        evidence = item.setdefault("evidence", {})
        evidence["quality_gate"] = "At least one recording remained blurred; category event classification withheld."
    for finding in findings:
        if finding.get("change_type") in {"added", "removed"}:
            finding["candidate_change_type"] = finding["change_type"]
            finding["change_type"] = "uncertain"
            finding["event_status"] = "uncertain; at least one recording remained blurred"
            finding["evidence"] += " Event status was downgraded because at least one recording remained blurred."
    return True


def _longest_consecutive_frame_run(frame_ids: list[int]) -> int:
    longest = current = previous = 0
    for frame in sorted(set(int(x) for x in frame_ids)):
        current = current + 1 if frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


def _confirmed_object_events(comparisons: list[dict]) -> list[dict]:
    """Remove unmatched detector hypotheses from the user-facing results."""
    return [x for x in comparisons
            if x.get("status") in {"moved", "unchanged", "added", "removed"}]


def _reconcile_moved_object_surfaces(findings: list[dict], comparisons: list[dict],
                                     distance_threshold: float = 0.28) -> tuple[list[dict], list[dict]]:
    """Remove add/remove surface fragments explained by a matched moved track."""
    moved = [x for x in comparisons if x.get("status") in {"moved", "unchanged"}]
    kept, reconciled = [], []
    for finding in findings:
        side = finding.get("candidate_side")
        point = finding.get("position_after") if side == "added" else finding.get("position_before")
        match = None
        if point is not None:
            for comparison in moved:
                if finding.get("semantic_label") != comparison.get("label"):
                    continue
                reference = comparison.get("position_after") if side == "added" else comparison.get("position_before")
                if reference is not None and np.linalg.norm(np.asarray(point) - np.asarray(reference)) <= distance_threshold:
                    match = comparison
                    break
        if match:
            reconciled.append({**finding, "reconciled_as": match["status"],
                "reconciliation_reason": f"Nearby {match['label']} track was matched across both recordings as {match['status']}; this surface fragment is not a separate {finding.get('change_type')} event."})
        else:
            kept.append(finding)
    return kept, reconciled


def _confirm_one_sided_object_events(comparisons: list[dict], findings: list[dict],
                                     detections_a: list[dict] | None = None,
                                     detections_b: list[dict] | None = None,
                                     visibility_context: dict | None = None) -> None:
    """Require repeated interior recognition and visible-empty opposite views."""
    for comparison in comparisons:
        if comparison.get("status") != "uncertain" or not comparison.get("candidate_side"):
            continue
        side = comparison["candidate_side"]
        event = "added" if side == "B" else "removed"
        # If the same category is also repeatedly detected in the opposite
        # clip, a one-sided track is likely a moved/fragmented track. Never call
        # it added/removed based only on the unmatched 3D cluster.
        opposite = detections_a if side == "B" else detections_b
        other_frames = [d["frame_index"] for d in (opposite or [])
                        if _canonical_label(d.get("label", "")) == comparison["label"] and
                        float(d.get("score", 0)) >= (0.15 if d.get("detector_source") == "OWLv2 zero-shot" else 0.55)]
        if _longest_consecutive_frame_run(other_frames) >= 3:
            continue
        location = comparison.get("position_after") if side == "B" else comparison.get("position_before")
        evidence = comparison.get("evidence", {})
        repeated = evidence.get("after_consecutive_frames", 0) if side == "B" else evidence.get("before_consecutive_frames", 0)
        interior_repeated = (evidence.get("after_interior_consecutive_frames", 0) if side == "B"
                             else evidence.get("before_interior_consecutive_frames", 0))
        if repeated < 3 or interior_repeated < 3 or location is None:
            continue
        for finding in findings:
            candidate_location = finding.get("position_after") if event == "added" else finding.get("position_before")
            if (finding.get("semantic_label") == comparison["label"] and
                    finding.get("candidate_side") == event and finding.get("change_type") == event and
                    finding.get("event_status") == "observed empty in opposite recording" and
                    candidate_location is not None and
                    np.linalg.norm(np.asarray(candidate_location) - np.asarray(location)) <= 0.25):
                comparison["status"] = event
                evidence["reason"] = "same-label track persists across consecutive sampled frames and a nearby surface region is confirmed visible and absent in the opposite recording"
                break
        else:
            # Object geometry can merge into a support surface (for example a
            # dark backpack resting on a dark chair), leaving no isolated
            # changed-surface cluster. In that case test the persistent track's
            # 3D samples directly against the opposite clip's projected depth
            # maps. A track alone is never enough to claim an add or removal.
            context = (visibility_context or {}).get(side)
            track_frames = set(evidence.get("after_frames" if side == "B" else "before_frames", []))
            track_detections = [d for d in (detections_b if side == "B" else detections_a) or []
                                if _canonical_label(d.get("label", "")) == comparison["label"]
                                and int(d.get("frame_index", -1)) in track_frames
                                and "position_scene_units" in d]
            if context and len(track_detections) >= 3:
                from .geometry import project_points_to_views
                result = project_points_to_views(
                    np.asarray([d["position_scene_units"] for d in track_detections]),
                    context["point_maps"], context["confidence_maps"],
                    context["camera_rotations"], context["camera_translations"],
                    context["intrinsics"])
                observations = np.asarray(result["observation_count"])
                visible_samples = int(np.count_nonzero(observations >= 2))
                visible_fraction = visible_samples / len(track_detections)
                if visible_samples >= 3 and visible_fraction >= 0.5:
                    comparison["status"] = event
                    evidence["opposite_visible_empty_track_samples"] = visible_samples
                    evidence["opposite_visible_empty_fraction"] = round(visible_fraction, 3)
                    evidence["reason"] = ("persistent same-label object track has at least three consecutive interior detections; "
                                           "its 3D samples project into multiple confident interior views of the opposite recording "
                                           "without a nearer surface")


def _label_findings_from_persistent_tracks(findings: list[dict], comparisons: list[dict],
                                           distance_threshold: float = 0.30) -> None:
    """Attach semantic names only when a persistent, cross-frame category track supports them."""
    supported = [x for x in comparisons if x.get("status") in {"moved", "unchanged", "added", "removed", "uncertain"}]
    for finding in findings:
        if finding.get("semantic_label") == "person":
            finding["semantic_label"] = "unidentified object/region"
        side = finding.get("candidate_side")
        position = finding.get("position_after") if side == "added" else finding.get("position_before")
        if position is None:
            continue
        choices = []
        for track in supported:
            if track.get("label") == "person":
                continue
            track_side = track.get("candidate_side")
            if track_side and track_side != ("B" if side == "added" else "A"):
                continue
            ref = track.get("position_after") if side == "added" else track.get("position_before")
            if ref is None:
                continue
            distance = float(np.linalg.norm(np.asarray(position) - np.asarray(ref)))
            if distance <= distance_threshold:
                choices.append((distance, track))
        if choices:
            _, track = min(choices, key=lambda x: x[0])
            finding["semantic_label"] = track["label"]
            finding["object_or_region"] = track["label"]
            finding["semantic_evidence"] = "Category supported by a persistent multi-frame 2D/3D object track; per-frame raw detections are listed in the report."
            finding["semantic_track_status"] = track["status"]
        else:
            finding["semantic_label"] = "unidentified object/region"
            finding["object_or_region"] = "unidentified changed region"
            finding["semantic_evidence"] = "No persistent, spatially matched semantic track; label and object event withheld."


def _withhold_untracked_surface_events(findings: list[dict], comparisons: list[dict]) -> None:
    """A visibility-only surface difference is a candidate, never a named object event."""
    for finding in findings:
        event = finding.get("change_type")
        if event not in {"added", "removed"}:
            continue
        side = "B" if event == "added" else "A"
        position = finding.get("position_after") if event == "added" else finding.get("position_before")
        matching = [x for x in comparisons if x.get("status") == event and x.get("candidate_side") == side
                    and x.get("label") == finding.get("semantic_label")]
        if position is not None and any(
                (p := (x.get("position_after") if event == "added" else x.get("position_before"))) is not None
                and np.linalg.norm(np.asarray(position) - np.asarray(p)) <= 0.35 for x in matching):
            continue
        finding["candidate_change_type"] = event
        finding["change_type"] = "uncertain"
        finding["event_status"] = "uncertain; geometry differs, but no persistent semantic track confirms an object event"
        finding["evidence"] += " The surface difference is kept as a review candidate; no moved/added/removed object label is asserted without persistent recognition."


def analyze(video_a_path: str, video_b_path: str, progress=None,
            settings: Settings = Settings(),
            object_labels: tuple[str, ...] | list[str] = DEFAULT_OPEN_VOCAB_LABELS) -> Path:
    start = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = settings.output_dir / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    logs = []
    def stage(message: str):
        logs.append(f"{time.strftime('%H:%M:%S')} {message}")
        if progress:
            progress(message)
    stage("Validating recordings and selecting frames…")
    a, b = inspect_video(video_a_path, settings), inspect_video(video_b_path, settings)
    for name, info in (("A", a), ("B", b)):
        for i, frame in enumerate(info.semantic_frames):
            from PIL import Image
            Image.fromarray(frame).save(out / f"recording_{name}_frame_{i:02d}.jpg", quality=90)
    stage("Reconstructing Recording A with VGGT…")
    pa, ca, meta_a, maps_a, conf_a, depths_a = reconstruct_vggt(a, progress)
    stage("Reconstructing Recording B with VGGT…")
    pb, cb, meta_b, maps_b, conf_b, depths_b = reconstruct_vggt(b, progress)
    stage("Aligning scale-ambiguous reconstructions and checking overlap…")
    # Normalize initial median radius to address independent monocular scale before rigid ICP.
    ra = np.median(np.linalg.norm(pa - np.median(pa, axis=0), axis=1))
    rb = np.median(np.linalg.norm(pb - np.median(pb, axis=0), axis=1))
    scale = ra / max(rb, 1e-8)
    pb_scaled = pb * scale
    aa, _ = voxel_downsample(pa, voxel=max(settings.voxel_size, 0.08))
    bb, _ = voxel_downsample(pb_scaled, voxel=max(settings.voxel_size, 0.08))
    # PCA multistart handles differing first-frame orientations; ICP refines each candidate.
    reg = pca_multistart_register(bb, aa, max_distance=max(0.1, settings.change_threshold * 2))
    min_alignment_overlap = 0.25
    max_alignment_rmse = max(settings.change_threshold * 2.0, 0.1)
    if not reg.success or reg.inlier_fraction < min_alignment_overlap or reg.rmse > max_alignment_rmse:
        raise RuntimeError(f"Scene alignment failed quality checks (overlap={reg.inlier_fraction:.2f}, RMSE={reg.rmse:.3f}). Record the same room with more shared views and texture.")
    pb_aligned = transform(pb_scaled, reg.rotation, reg.translation)
    camera_b = transform(np.asarray(meta_b["camera_positions"]) * scale, reg.rotation, reg.translation)
    meta_b["camera_positions"] = camera_b.tolist()
    # Derive tolerance from alignment residuals; never force an arbitrary fraction of the
    # cloud to become a "change" when the two scenes are actually similar.
    threshold = max(settings.change_threshold, float(reg.rmse * 3.0))
    # Candidate B geometry is tested for visibility from A's predicted camera views,
    # and vice versa. Only confident, repeated projections support an event label.
    unmatched_b, _ = surface_unmatched(pb_aligned, pa, threshold)
    unmatched_a, _ = surface_unmatched(pa, pb_aligned, threshold)
    # Bring A's cameras into A coordinates (already native) and B point maps/cameras
    # into A coordinates using the same scale+rigid transform as B's point cloud.
    rotations_a = np.asarray(meta_a["camera_rotations"], dtype=float)
    translations_a = np.asarray(meta_a["camera_translations"], dtype=float)
    # If x' = scale * Q * x + u, transform the full world-to-camera extrinsics.
    rotations_b, translations_b = transform_camera_extrinsics(
        meta_b["camera_rotations"], meta_b["camera_translations"], scale,
        reg.rotation, reg.translation)
    maps_b_aligned = transform(maps_b.reshape(-1, 3) * scale, reg.rotation, reg.translation).reshape(maps_b.shape)
    vis_b_from_a = project_points_to_views(unmatched_b, maps_a, conf_a, rotations_a, translations_a,
                                           np.asarray(meta_a["camera_intrinsics"], dtype=float))
    # Similarity transform x'=s*R*x+t scales camera-space translation and depth,
    # but leaves pixel intrinsics unchanged.
    vis_a_from_b = project_points_to_views(unmatched_a, maps_b_aligned, conf_b, rotations_b, translations_b,
                                           np.asarray(meta_b["camera_intrinsics"], dtype=float))
    overlap_confidence = min(0.65, max(0.25, reg.inlier_fraction * (1 - min(1, reg.rmse / max(threshold, 1e-6)))))
    findings = (_clusters(unmatched_b, "added", overlap_confidence, vis_b_from_a) +
                _clusters(unmatched_a, "removed", overlap_confidence, vis_a_from_b))
    stage("Checking local semantic evidence…")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    sem_a = geolocate_detections(detect_objects(a.semantic_frames, device, open_vocab_labels=object_labels), maps_a, conf_a, a.semantic_frames)
    sem_b = geolocate_detections(detect_objects(b.semantic_frames, device, open_vocab_labels=object_labels), maps_b, conf_b, b.semantic_frames)
    sem_b = [dict(x, position_scene_units=transform(np.asarray([x["position_scene_units"]]) * scale,
                                                     reg.rotation, reg.translation)[0].tolist()) for x in sem_b]
    object_comparisons = compare_detected_objects(sem_a, sem_b)
    _label_findings_from_persistent_tracks(findings, object_comparisons)
    _confirm_one_sided_object_events(
        object_comparisons, findings, sem_a, sem_b,
        visibility_context={
            "B": {"point_maps": maps_a, "confidence_maps": conf_a,
                 "camera_rotations": rotations_a, "camera_translations": translations_a,
                 "intrinsics": np.asarray(meta_a["camera_intrinsics"], dtype=float)},
            "A": {"point_maps": maps_b_aligned, "confidence_maps": conf_b,
                 "camera_rotations": rotations_b, "camera_translations": translations_b,
                 "intrinsics": np.asarray(meta_b["camera_intrinsics"], dtype=float)},
        })
    _withhold_untracked_surface_events(findings, object_comparisons)
    blur_quality_limited = _apply_blur_quality_gate(findings, object_comparisons, (a, b))
    # Unmatched, unconfirmed detector tracks are diagnostic noise, not results.
    # Keep only events supported by consecutive recognition plus cross-clip or
    # visibility evidence. A blur gate withholds the event entirely.
    suppressed_object_track_count = sum(x.get("status") == "uncertain" for x in object_comparisons)
    object_comparisons = _confirmed_object_events(object_comparisons)
    findings, reconciled_surface_findings = _reconcile_moved_object_surfaces(findings, object_comparisons)
    # The end-user report is an event list, not a collection of speculative
    # geometric differences. Keep only regions backed by a confirmed object
    # event; inconclusive regions remain absent from the presented outputs.
    findings = [x for x in findings if x.get("change_type") in {"moved", "added", "removed"}]
    stage("Exporting point clouds, visualization, and report…")
    np.savez_compressed(out / "reconstruction.npz", points_a=pa.astype(np.float32), colors_a=ca,
                        points_b_aligned=pb_aligned.astype(np.float32), colors_b=cb,
                        cameras_a=np.asarray(meta_a["camera_positions"], dtype=np.float32),
                        cameras_b_aligned=camera_b.astype(np.float32), depth_maps_a=depths_a,
                        depth_maps_b=depths_b)
    _write_ply(out / "recording_a.ply", pa, ca)
    _write_ply(out / "recording_b_aligned.ply", pb_aligned, cb)
    report = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
              "recordings": {"A": a.public_dict(), "B": b.public_dict()},
              "reconstruction": {"A": meta_a, "B": meta_b, "coordinate_units": "relative scene units; each independently reconstructed monocular scene has unknown metric scale", "initial_scale_ratio_b_to_a": float(scale)},
              "alignment": {"method": "PCA multi-start trimmed point-to-point ICP after robust global-radius scale normalization", "success": reg.success, "status": "ok", "minimum_overlap_fraction": min_alignment_overlap, "maximum_rmse_scene_units": max_alignment_rmse, "inlier_fraction": reg.inlier_fraction, "rmse_scene_units": reg.rmse, "change_threshold_scene_units": threshold},
              "visibility_analysis": {"method": "camera-intrinsics projection into per-frame confidence-filtered point maps with a local depth z-buffer; frame-edge projections do not confirm empty space", "views_sampled_per_recording": 12, "minimum_candidate_points": 2, "minimum_visible_fraction": 0.5, "limitations": "Visibility is approximate; VGGT pose/depth and inter-clip registration errors can still cause false visibility or occlusion decisions."},
              "findings": findings, "object_comparisons": object_comparisons,
              "reconciled_surface_findings": reconciled_surface_findings,
              "suppressed_object_track_count": suppressed_object_track_count,
              "quality_gates": {"blur_limited_events": blur_quality_limited,
                                "sharpness_threshold": 35,
                                "action": "Sharpest nearby frames are selected. Event claims are downgraded to uncertain when the selected frames still trigger the blur warning."},
              "object_recognition": {"detectors": ["COCO Faster R-CNN", "OWLv2 zero-shot"],
                                     "open_vocabulary_labels": list(object_labels),
                                     "open_vocabulary": open_vocabulary_status(),
                                     "track_policy": "object events require at least three consecutive sampled-frame detections and either median COCO score >=0.70 or median OWLv2 score >=0.25; one-frame surface labels are withheld"},
              "semantic_detections_a": sem_a, "semantic_detections_b": sem_b,
              "timings_seconds": {"total": round(time.perf_counter() - start, 2)},
              "runtime": {"python": platform.python_version(), "platform": platform.platform(), "device": device},
              "limitations": ["Visibility has not been calibrated against independent ground truth; different viewpoints can leave objects hidden or unobserved.", "Visibility is approximated by projecting candidate points into confident VGGT point maps; it is not exact ray casting and can still be wrong when poses/depth are noisy.", "Object labels use closed-set COCO and open-vocabulary OWLv2; these are hypotheses, not calibrated probabilities or proof of exact instance identity.", "A moved label is a category-track match, not proof that the exact same physical instance was followed when multiple objects share that category.", "Object events require detections in at least three consecutive sampled frames; sparse or occluded views may suppress true events.", "Monocular scale is ambiguous; all coordinates are relative scene units.", "2D box-to-3D association uses confident VGGT points in a detector box and spatial proximity; detector and pose errors can still mislabel geometry."],
              "visibility": "Unknown outside reconstructed surface support; absence alone is not treated as a confirmed removal."}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "changes.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["object_or_region", "semantic_label", "change_type", "candidate_change_type", "candidate_side", "event_status", "position_before", "position_after", "displacement", "units", "confidence", "visibility_evidence", "semantic_confidence", "evidence"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: x.get(k) for k in fields} for x in findings)
    with (out / "object_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["label", "status", "candidate_status", "candidate_side", "position_before", "position_after", "displacement_scene_units", "evidence"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**x, "position_before": json.dumps(x.get("position_before")),
                          "position_after": json.dumps(x.get("position_after")),
                          "evidence": json.dumps(x.get("evidence", {}))} for x in object_comparisons)
    summary = ["# RoomChange3D results", "", f"Alignment quality: {'passed' if reg.success else 'failed'}; ICP RMSE {reg.rmse:.4f} relative units, overlap fraction {reg.inlier_fraction:.2f}.", "", f"Reported {len(object_comparisons)} verified object track(s). Unsupported detector guesses and inconclusive surface regions were omitted.", "", "## Verified object events"]
    summary.extend(f"- {x['label']}: {x['status']}" + (f" (present in Recording {x['candidate_side']} only)." if x.get("candidate_side") else ".") for x in object_comparisons)
    summary.extend(["", "## Limitations", *[f"- {x}" for x in report["limitations"]]])
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    logs.append(json.dumps({"settings": {"max_frames": settings.max_frames, "max_side": settings.max_side,
                            "voxel_size": settings.voxel_size, "change_threshold": settings.change_threshold},
                           "alignment": report["alignment"], "models": [meta_a, meta_b], "runtime": report["runtime"]}, default=str))
    (out / "processing.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    _write_html(out / "scene.html", pa, ca, pb_aligned, cb, findings, object_comparisons,
                np.asarray(meta_a["camera_positions"]), camera_b)
    write_results_bundle(out)
    stage(f"Done. Results saved to {out.resolve()}")
    return out


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(points))
        for p, c in zip(points, colors):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (*p, *c[:3]))


def write_results_bundle(out: Path) -> Path:
    """Zip generated output files without including input videos or the zip itself."""
    bundle = out / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for file in out.iterdir():
            if file.is_file() and file != bundle and file.suffix.lower() not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
                archive.write(file, arcname=file.name)
    return bundle


def _write_html(path: Path, pa: np.ndarray, ca: np.ndarray, pb: np.ndarray, cb: np.ndarray,
                findings: list[dict], object_comparisons: list[dict],
                cameras_a: np.ndarray, cameras_b: np.ndarray) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    for points, colors, name in ((pa, ca, "Recording A"), (pb, cb, "Recording B aligned")):
        step = max(1, len(points) // 12000)
        pts, col = points[::step], colors[::step]
        rgb = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in col]
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name,
                                   marker={"size": 2, "color": rgb, "opacity": 0.75}))
    for change_type, color in (("uncertain", "#f6c453"),):
        positions = [(x.get("position_after") if x.get("candidate_side") == "added" else x.get("position_before"))
                     for x in findings if x["change_type"] == change_type]
        positions = [p for p in positions if p is not None]
        if positions:
            p = np.asarray(positions)
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers+text",
                                       text=[change_type.upper()] * len(p), textposition="top center",
                                       name=f"{change_type.title()} candidate regions",
                                       marker={"size": 8, "color": color, "symbol": "diamond"}))
    event_colors = {"moved": "#ffd166", "unchanged": "#67b7ff", "added": "#48e0a4", "removed": "#ff6b75"}
    for item in object_comparisons:
        status = item.get("status")
        if status not in event_colors:
            continue
        before, after = item.get("position_before"), item.get("position_after")
        points = [p for p in (before, after) if p is not None]
        if not points:
            continue
        labels = [f"{item['label']} · before" if before is not None else None,
                  f"{item['label']} · after" if after is not None else None]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
            mode="markers+text" if len(points) == 1 else "lines+markers+text",
            text=[x for x in labels if x is not None], textposition="top center",
            name=f"{item['label'].title()} · {status}",
            line={"width": 5, "color": event_colors[status]},
            marker={"size": 8, "color": event_colors[status], "symbol": "diamond" if status in {"added", "removed"} else "circle"}))
    for positions, name, color in ((cameras_a, "Recording A cameras", "#f6c453"),
                                   (cameras_b, "Recording B cameras aligned", "#67b7ff")):
        p = np.asarray(positions)
        if len(p):
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers", name=name,
                                       marker={"size": 4, "color": color}, line={"width": 3, "color": color}))
    fig.update_layout(title="Aligned room reconstructions · labeled event locations · relative scene units", template="plotly_dark", scene_aspectmode="data")
    fig.write_html(path, include_plotlyjs=True, full_html=True)

