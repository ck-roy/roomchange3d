"""Optional local COCO object detector. All detections remain frame-grounded evidence."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_IMAGENET_MODEL = None
_IMAGENET_WEIGHTS = None
_OWLV2_MODEL = None
_OWLV2_PROCESSOR = None
_OWLV2_NAME = "google/owlv2-base-patch16-ensemble"
_OWLV2_ERROR: str | None = None


DEFAULT_OPEN_VOCAB_LABELS = (
    "bag", "bottle", "laptop", "chair", "table", "sofa", "bed", "desk",
    "book", "cup", "remote control", "cell phone", "keyboard", "mouse",
    "shoe", "lamp", "potted plant", "basket", "box", "suitcase",
)


def _canonical_label(label: str) -> str:
    """Map common detector names to the user's broad requested category."""
    value = " ".join(label.lower().replace("_", " ").split())
    aliases = {"backpack": "bag", "handbag": "bag", "rucksack": "bag",
               "water bottle": "bottle", "rubber shoe": "shoe", "footwear": "shoe"}
    return aliases.get(value, value)


def detect_open_vocabulary_objects(frames: list[np.ndarray], device: str,
                                   labels: tuple[str, ...] | list[str] = DEFAULT_OPEN_VOCAB_LABELS,
                                   score_threshold: float = 0.15) -> list[dict]:
    """Detect built-in and optional object categories with OWLv2."""
    global _OWLV2_MODEL, _OWLV2_PROCESSOR, _OWLV2_ERROR
    if not frames or not labels:
        return []
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        if _OWLV2_MODEL is None:
            _OWLV2_PROCESSOR = AutoProcessor.from_pretrained(_OWLV2_NAME)
            _OWLV2_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(_OWLV2_NAME).to(device).eval()
        elif next(_OWLV2_MODEL.parameters()).device.type != device:
            _OWLV2_MODEL = _OWLV2_MODEL.to(device)

        prompts = []
        prompt_to_label = {}
        for label in labels:
            canonical = _canonical_label(label)
            variants = (["a photo of a bag", "a photo of a backpack", "a photo of a handbag",
                         "a photo of a rucksack"] if canonical == "bag" else
                        [f"a photo of {label}" if label.lower().startswith(("a ", "an "))
                         else f"a photo of a {label}"])
            for prompt in variants:
                if prompt not in prompt_to_label:
                    prompts.append(prompt)
                    prompt_to_label[prompt] = canonical
        out: list[dict] = []
        with torch.inference_mode():
            for frame_id, frame in enumerate(frames):
                image = Image.fromarray(frame)
                inputs = _OWLV2_PROCESSOR(text=[prompts], images=image, return_tensors="pt").to(device)
                outputs = _OWLV2_MODEL(**inputs)
                result = _OWLV2_PROCESSOR.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=torch.tensor([(image.height, image.width)]),
                    threshold=score_threshold,
                    text_labels=[prompts],
                )[0]
                for box, score, prompt in zip(result["boxes"].cpu().numpy(),
                                              result["scores"].cpu().numpy(), result["text_labels"]):
                    prompt = str(prompt)
                    if prompt not in prompt_to_label:
                        continue
                    out.append({"frame_index": frame_id, "label": prompt_to_label[prompt],
                                "score": float(score), "box_xyxy": box.tolist(),
                                "detector_source": "OWLv2 zero-shot", "query": prompt})
        _OWLV2_ERROR = None
        return out
    except Exception as exc:
        _OWLV2_ERROR = str(exc)
        log.warning("Open-vocabulary detector unavailable; continuing with COCO detections: %s", exc)
        return []


def open_vocabulary_status() -> dict:
    return {"model": _OWLV2_NAME, "loaded": _OWLV2_MODEL is not None,
            "error": _OWLV2_ERROR}


def _imagenet_label_support(label: str, top_labels: list[str]) -> bool:
    """Require an independent ImageNet classifier to agree on ambiguous classes."""
    top = {x.lower().replace("_", " ") for x in top_labels}
    if label == "book":
        # "notebook" is deliberately excluded: the detector often labels a
        # laptop/tablet as a book and ImageNet uses notebook for computers too.
        return bool(top & {"comic book", "book jacket"})
    if label == "banana":
        return "banana" in top
    if label == "backpack":
        return "backpack" in top
    if label == "remote":
        return "remote control" in top
    if label == "bottle":
        return any("bottle" in x and x != "bottlecap" for x in top)
    if label == "chair":
        return any("chair" in x for x in top)
    if label == "cup":
        return bool(top & {"cup", "measuring cup", "coffee mug", "goblet"})
    if label == "sports ball":
        return any("ball" in x and x not in {"balloon", "ballpoint", "ballplayer"} for x in top)
    if label == "bird":
        return any("bird" in x and "birdhouse" not in x for x in top)
    if label == "couch":
        return "studio couch" in top
    if label == "dining table":
        return "dining table" in top
    if label == "laptop":
        return "laptop" in top
    if label == "cell phone":
        return "cellular telephone" in top
    if label == "tv":
        return "television" in top
    if label == "keyboard":
        return bool(top & {"computer keyboard", "typewriter keyboard"})
    if label == "mouse":
        return "computer mouse" in top
    return True


def _verify_detection_categories(frames: list[np.ndarray], detections: list[dict], device: str) -> list[dict]:
    """Use ImageNet crop classification to reject known COCO category confusions."""
    global _IMAGENET_MODEL, _IMAGENET_WEIGHTS
    verify_labels = {"book", "banana", "backpack", "remote", "bottle", "chair", "cup", "sports ball", "bird",
                     "couch", "dining table", "laptop", "cell phone", "tv", "keyboard", "mouse"}
    candidates = [d for d in detections if d["label"] in verify_labels]
    if not candidates:
        return detections
    try:
        import torch
        from PIL import Image
        from torchvision.models import ResNet50_Weights, resnet50
        if _IMAGENET_MODEL is None:
            _IMAGENET_WEIGHTS = ResNet50_Weights.DEFAULT
            _IMAGENET_MODEL = resnet50(weights=_IMAGENET_WEIGHTS).to(device).eval()
        elif next(_IMAGENET_MODEL.parameters()).device.type != device:
            _IMAGENET_MODEL = _IMAGENET_MODEL.to(device)
        transform = _IMAGENET_WEIGHTS.transforms()
        keep = {id(d): True for d in detections}
        categories = _IMAGENET_WEIGHTS.meta["categories"]
        with torch.inference_mode():
            for detection in candidates:
                frame = frames[int(detection["frame_index"])]
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = detection["box_xyxy"]
                crop = frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
                if crop.size == 0:
                    keep[id(detection)] = False
                    continue
                tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
                probabilities = _IMAGENET_MODEL(tensor).softmax(1)[0]
                values, indices = probabilities.topk(5)
                labels = [categories[int(i)] for i in indices]
                supported = _imagenet_label_support(detection["label"], labels)
                detection["recognition_check"] = {"model": "ResNet50 ImageNet-1K", "supporting_top5": labels,
                                                  "agreed": bool(supported)}
                keep[id(detection)] = bool(supported)
        return [d for d in detections if keep[id(d)]]
    except Exception as exc:
        # Fail closed for categories where the second model is required. Other
        # COCO categories remain available through the existing detector.
        log.warning("ImageNet category verification unavailable; suppressing ambiguous category guesses: %s", exc)
        return [d for d in detections if d["label"] not in verify_labels]


def detect_objects(frames: list[np.ndarray], device: str = "cpu", score_threshold: float = 0.55,
                   open_vocab_labels: tuple[str, ...] | list[str] = DEFAULT_OPEN_VOCAB_LABELS) -> list[dict]:
    """Combine COCO detections with open-vocabulary target detections where available."""
    coco: list[dict] = []
    try:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    except Exception as exc:
        log.info("Optional torchvision detector unavailable: %s", exc)
    else:
        labels = weights.meta["categories"]
        with torch.inference_mode():
            for frame_id, frame in enumerate(frames):
                image = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(device)
                result = model([image])[0]
                for box, label, score in zip(result["boxes"].cpu().numpy(), result["labels"].cpu().numpy(), result["scores"].cpu().numpy()):
                    if float(score) >= score_threshold:
                        coco.append({"frame_index": frame_id, "label": labels[int(label)], "score": float(score),
                                     "box_xyxy": box.tolist(), "detector_source": "COCO Faster R-CNN"})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        coco = _verify_detection_categories(frames, coco, device)

    open_vocab = detect_open_vocabulary_objects(frames, device, open_vocab_labels)
    # Merge same-category boxes from both detectors so one object in one frame
    # cannot inflate temporal evidence. Prefer COCO confidence when boxes overlap.
    combined = list(coco)
    for detection in open_vocab:
        duplicate = False
        for existing in combined:
            if (existing["frame_index"] != detection["frame_index"] or
                    _canonical_label(existing["label"]) != detection["label"]):
                continue
            if _box_iou(existing["box_xyxy"], detection["box_xyxy"]) >= 0.45:
                duplicate = True
                break
        if not duplicate:
            combined.append(detection)
    return _suppress_nested_component_detections(combined)


def _box_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _suppress_nested_component_detections(detections: list[dict]) -> list[dict]:
    """Avoid reporting a laptop's built-in keyboard as a separate room object."""
    laptops = [d for d in detections if _canonical_label(d.get("label", "")) == "laptop"]
    keep = []
    for detection in detections:
        label = _canonical_label(detection.get("label", ""))
        if label in {"keyboard", "mouse"}:
            x1, y1, x2, y2 = map(float, detection["box_xyxy"])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            for laptop in laptops:
                if laptop["frame_index"] != detection["frame_index"] or area <= 0:
                    continue
                lx1, ly1, lx2, ly2 = map(float, laptop["box_xyxy"])
                overlap = max(0, min(x2, lx2) - max(x1, lx1)) * max(0, min(y2, ly2) - max(y1, ly1))
                if overlap / area >= 0.85:
                    break
            else:
                keep.append(detection)
        else:
            keep.append(detection)
    return keep


def geolocate_detections(detections: list[dict], point_maps: np.ndarray,
                         confidence_maps: np.ndarray, frames: list[np.ndarray]) -> list[dict]:
    """Associate each 2D detector box with confident VGGT point-map samples in that box."""
    located = []
    for detection in detections:
        i = int(detection["frame_index"])
        if i >= len(point_maps):
            continue
        ph, pw = point_maps[i].shape[:2]
        fh, fw = frames[i].shape[:2]
        x1, y1, x2, y2 = detection["box_xyxy"]
        x1, x2 = int(np.clip(x1 * pw / fw, 0, pw)), int(np.clip(x2 * pw / fw, 0, pw))
        y1, y2 = int(np.clip(y1 * ph / fh, 0, ph)), int(np.clip(y2 * ph / fh, 0, ph))
        if x2 <= x1 or y2 <= y1:
            continue
        pts = point_maps[i, y1:y2, x1:x2].reshape(-1, 3)
        conf = confidence_maps[i, y1:y2, x1:x2].reshape(-1)
        valid = np.isfinite(pts).all(1) & np.isfinite(conf)
        if valid.sum() < 30:
            continue
        threshold = np.quantile(conf[valid], 0.25)
        pts = pts[valid & (conf >= threshold)]
        if len(pts) < 20:
            continue
        item = dict(detection)
        item["position_scene_units"] = np.median(pts, axis=0).tolist()
        item["projected_point_count"] = int(len(pts))
        # Detector boxes touching the crop boundary are often truncated. Keep
        # these observations visible in the report, but never use them alone to
        # confirm an object event.
        margin_x, margin_y = fw * 0.04, fh * 0.04
        item["touches_frame_edge"] = bool(x1 * fw / pw <= margin_x or y1 * fh / ph <= margin_y or
                                           x2 * fw / pw >= fw - margin_x or y2 * fh / ph >= fh - margin_y)
        located.append(item)
    return located


def compare_detected_objects(detections_a: list[dict], detections_b: list[dict],
                             match_threshold: float = 0.30) -> list[dict]:
    """Compare every geolocated category track and preserve weak cases as uncertain.

    COCO labels describe categories, not exact instances. Repeated, confident
    interior observations can support a tentative move or unchanged result.
    A one-sided track is always retained with an uncertain status: lack of a
    detection is not proof of an addition or removal because the object may be
    hidden, outside the sampled views, or missed by the detector.
    """
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.optimize import linear_sum_assignment

    def tracks(detections):
        result = {}
        # People are normally the phone operator or a transient visitor, not a
        # room object whose placement should be compared.
        labels = sorted({_canonical_label(d["label"]) for d in detections if d["label"] != "person"})
        for label in labels:
            items = [d for d in detections if _canonical_label(d["label"]) == label and d["label"] != "person" and "position_scene_units" in d]
            if not items:
                continue
            if label == "bag":
                # The bag detector boxes include varying amounts of chair and
                # floor as the camera approaches. Whole-track spatial
                # clustering splits this one object as its estimated 3D center
                # drifts. Link adjacent samples first for the bag category.
                tracks_by_frame = []
                for frame_id in sorted({int(x["frame_index"]) for x in items}):
                    current = [x for x in items if int(x["frame_index"]) == frame_id]
                    active = [track for track in tracks_by_frame
                              if int(track[-1]["frame_index"]) == frame_id - 1]
                    if active and current:
                        distances = np.linalg.norm(
                            np.asarray([track[-1]["position_scene_units"] for track in active])[:, None, :] -
                            np.asarray([x["position_scene_units"] for x in current])[None, :, :], axis=2)
                        rows, cols = linear_sum_assignment(distances)
                        assigned = set()
                        for row, col in zip(rows, cols):
                            if distances[row, col] <= max(match_threshold, 0.20):
                                active[int(row)].append(current[int(col)])
                                assigned.add(int(col))
                        tracks_by_frame.extend([[x] for i, x in enumerate(current) if i not in assigned])
                    else:
                        tracks_by_frame.extend([[x] for x in current])
            else:
                xyz = np.asarray([d["position_scene_units"] for d in items], dtype=float)
                groups = fclusterdata(xyz, t=match_threshold, criterion="distance", method="average") if len(items) > 1 else np.ones(1, int)
                tracks_by_frame = [[items[i] for i in np.flatnonzero(groups == group)]
                                   for group in np.unique(groups)]
            clusters = []
            for members in tracks_by_frame:
                observed_frames = sorted({int(x["frame_index"]) for x in members})
                observed_frames = sorted({int(x["frame_index"]) for x in members})
                consecutive = current = 0
                interior_consecutive = interior_current = 0
                previous = None
                for frame_id in observed_frames:
                    current = current + 1 if previous is not None and frame_id == previous + 1 else 1
                    consecutive = max(consecutive, current)
                    interior_here = any(int(x["frame_index"]) == frame_id and not x.get("touches_frame_edge", False)
                                        for x in members)
                    interior_current = (interior_current + 1 if interior_here else 0) if previous is not None and frame_id == previous + 1 else (1 if interior_here else 0)
                    interior_consecutive = max(interior_consecutive, interior_current)
                    previous = frame_id
                coco_scores = [float(x["score"]) for x in members
                               if x.get("detector_source", "COCO Faster R-CNN") == "COCO Faster R-CNN"]
                owl_scores = [float(x["score"]) for x in members if x.get("detector_source") == "OWLv2 zero-shot"]
                clusters.append({"position": np.median([x["position_scene_units"] for x in members], axis=0),
                                 "frame_count": len(observed_frames),
                                 "consecutive_count": consecutive,
                                 "interior_consecutive_count": interior_consecutive,
                                 "frames": observed_frames,
                                 "score": float(np.median([x["score"] for x in members])),
                                 "coco_median_score": float(np.median(coco_scores)) if coco_scores else None,
                                 "open_vocab_median_score": float(np.median(owl_scores)) if owl_scores else None,
                                 "detector_sources": sorted({x.get("detector_source", "COCO Faster R-CNN") for x in members}),
                                 "edge_fraction": float(np.mean([x.get("touches_frame_edge", False) for x in members])),
                                 "observation_count": len(members)})
            result[label] = clusters
        return result

    ta, tb = tracks(detections_a), tracks(detections_b)
    comparisons = []
    # Edge-touching boxes can help establish that a category appeared, but they
    # are weak evidence for matching its 3D location. Require interior views in
    # both clips before making a moved/unchanged claim; this avoids classifying
    # a partial, cropped track from frame-edge detections alone.
    def supported(track):
        score_ok = ((track["coco_median_score"] is not None and track["coco_median_score"] >= 0.70) or
                    (track["open_vocab_median_score"] is not None and track["open_vocab_median_score"] >= 0.25))
        return track["frame_count"] >= 3 and track["consecutive_count"] >= 3 and score_ok

    def record(label, status, before=None, after=None, reason=""):
        comparisons.append({"label": label, "status": status,
            "candidate_side": "A" if after is None else ("B" if before is None else None),
            "position_before": before["position"].tolist() if before is not None else None,
            "position_after": after["position"].tolist() if after is not None else None,
            "displacement_scene_units": (round(float(np.linalg.norm(before["position"] - after["position"])), 4)
                                         if before is not None and after is not None else None),
            "evidence": {"before_frames": before["frames"] if before is not None else [],
                         "after_frames": after["frames"] if after is not None else [],
                         "before_median_score": round(before["score"], 3) if before is not None else None,
                         "after_median_score": round(after["score"], 3) if after is not None else None,
                         "before_coco_median_score": round(before["coco_median_score"], 3) if before is not None and before["coco_median_score"] is not None else None,
                         "after_coco_median_score": round(after["coco_median_score"], 3) if after is not None and after["coco_median_score"] is not None else None,
                         "before_open_vocab_median_score": round(before["open_vocab_median_score"], 3) if before is not None and before["open_vocab_median_score"] is not None else None,
                         "after_open_vocab_median_score": round(after["open_vocab_median_score"], 3) if after is not None and after["open_vocab_median_score"] is not None else None,
                         "before_detector_sources": before["detector_sources"] if before is not None else [],
                         "after_detector_sources": after["detector_sources"] if after is not None else [],
                         "frame_edge_fraction": round(max(before["edge_fraction"] if before else 0,
                                                           after["edge_fraction"] if after else 0), 3),
                         "before_observations": before["observation_count"] if before is not None else 0,
                         "after_observations": after["observation_count"] if after is not None else 0,
                         "before_consecutive_frames": before["consecutive_count"] if before is not None else 0,
                         "after_consecutive_frames": after["consecutive_count"] if after is not None else 0,
                         "before_interior_consecutive_frames": before["interior_consecutive_count"] if before is not None else 0,
                         "after_interior_consecutive_frames": after["interior_consecutive_count"] if after is not None else 0,
                         "reason": reason}})

    for label in sorted(set(ta) | set(tb)):
        left, right = ta.get(label, []), tb.get(label, [])
        good_left = [i for i, x in enumerate(left) if supported(x)]
        good_right = [i for i, x in enumerate(right) if supported(x)]
        used_left, used_right = set(), set()
        if good_left and good_right:
            distances = np.linalg.norm(np.asarray([left[i]["position"] for i in good_left])[:, None, :] -
                                       np.asarray([right[j]["position"] for j in good_right])[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(distances)
            for row, col in zip(rows, cols):
                i, j = good_left[int(row)], good_right[int(col)]
                a, b = left[i], right[j]
                if (a["interior_consecutive_count"] < 3 or
                        b["interior_consecutive_count"] < 3):
                    continue
                used_left.add(i); used_right.add(j)
                distance = float(distances[row, col])
                status = "unchanged" if distance <= match_threshold else "moved"
                record(label, status, a, b,
                       "matched repeated category tracks by 3D proximity; category labels do not prove exact instance identity")
        for i, track in enumerate(left):
            if i not in used_left and supported(track):
                reason = "no matched track in Recording B; event requires opposite-view visibility evidence"
                record(label, "uncertain", track, None, reason)
        for j, track in enumerate(right):
            if j not in used_right and supported(track):
                reason = "no matched track in Recording A; event requires opposite-view visibility evidence"
                record(label, "uncertain", None, track, reason)
    return comparisons

