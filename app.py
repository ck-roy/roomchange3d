from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from roomchange.config import Settings
from roomchange.pipeline import analyze
from roomchange.semantics import DEFAULT_OPEN_VOCAB_LABELS
from roomchange.video import inspect_video

st.set_page_config(page_title="RoomChange3D · Humanoid challenge", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp {background: #10151c; color: #e8edf3}
.hero {padding: 1.6rem 1.8rem; border: 1px solid #293644; border-radius: 14px; background: linear-gradient(115deg,#18232d,#111820); margin-bottom: 1.2rem}
.hero h1 {margin:0; letter-spacing:-.03em; font-size:2.25rem}
.hero p {color:#a8bac9; margin:.55rem 0 0}
div[data-testid="stMetric"] {background:#18232d; border:1px solid #293644; padding:14px; border-radius:10px}
</style>
<div class="hero"><h1>◈ RoomChange3D</h1><p>Compare two room recordings in reconstructed 3D. See object changes supported by repeated video evidence.</p></div>
""", unsafe_allow_html=True)

with st.expander("Recording guide · get the best reconstruction", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Capture both clips**\n\n1. Pick a small room with textured, distinguishable objects.\n2. Record 6–20 seconds, moving slowly around the room.\n3. Move/add/remove a few objects.\n4. Record the same area from different angles, retaining overlap.\n5. Keep light bright and phone motion smooth; avoid cuts and blur.")
    with c2:
        st.markdown("**What affects results**\n\nCoverage matters more than duration. Repetitive, reflective or textureless surfaces are hard to reconstruct. Fully unseen areas cannot be compared. Scale from monocular video is ambiguous, so distances use relative scene units. Use MP4/H.264 if MOV decoding fails.")

left, right = st.columns(2)
up_a = left.file_uploader("Recording A · before", type=["mp4", "mov", "m4v", "avi", "mkv"], key="a")
up_b = right.file_uploader("Recording B · after", type=["mp4", "mov", "m4v", "avi", "mkv"], key="b")

def save_upload(upload, label):
    digest = hashlib.sha256(upload.getbuffer()).hexdigest()
    cache = st.session_state.setdefault("upload_tempfiles", {})
    existing = cache.get(label)
    if existing and existing[0] == digest and Path(existing[1]).exists():
        return Path(existing[1])
    suffix = Path(upload.name).suffix or ".mp4"
    f = tempfile.NamedTemporaryFile(prefix=f"roomchange_{label}_", suffix=suffix, delete=False)
    f.write(upload.getbuffer())
    f.close()
    if existing:
        try:
            Path(existing[1]).unlink(missing_ok=True)
        except OSError:
            pass
    cache[label] = (digest, f.name)
    return Path(f.name)

paths = {}
validated = set()
for key, upload, col in (("A", up_a, left), ("B", up_b, right)):
    if upload:
        p = save_upload(upload, key)
        paths[key] = p
        col.video(upload.getvalue())
        try:
            info = inspect_video(p)
            validated.add(key)
            col.caption(f"{info.width}×{info.height} · {info.duration_seconds:.1f}s · {info.frame_count} frames · sharpness {info.sharpness:.0f} · selected the sharpest available samples")
            for warning in info.warnings:
                col.warning(warning)
        except Exception as e:
            col.error(str(e))

labels_text = st.text_input(
    "Extra object prompts (optional)", value="", placeholder="For example: backpack, vase, shoes",
    help="The built-in detector automatically checks its full object vocabulary and common room-object categories. Extra prompts add open-vocabulary coverage; bag prompts also check backpack, handbag and rucksack descriptions.")
extra_object_labels = tuple(x.strip() for x in labels_text.split(",") if x.strip())
object_labels = tuple(dict.fromkeys((*DEFAULT_OPEN_VOCAB_LABELS, *extra_object_labels)))

st.divider()
run = st.button("Analyse Scene Changes", type="primary", disabled=len(validated) != 2, use_container_width=True)
latest_runs = [p for p in sorted(Path("outputs").glob("run_*"), reverse=True) if (p / "report.json").exists()]
if latest_runs and st.button("Show latest local run", use_container_width=True):
    st.session_state["result_dir"] = str(latest_runs[0].resolve())
    st.rerun()
if run:
    bar = st.progress(0.0, text="Starting analysis…")
    stages = ["Validating recordings", "Reconstructing Recording A", "Reconstructing Recording B", "Aligning and comparing 3D", "Checking semantic evidence", "Exporting results"]
    def progress(message):
        fraction = min(0.94, max(0.04, (next((i + 1 for i, s in enumerate(stages) if s.lower() in message.lower()), 1)) / len(stages)))
        bar.progress(fraction, text=message)
    try:
        with st.spinner("Processing locally. The first model run downloads VGGT weights."):
            result = analyze(str(paths["A"]), str(paths["B"]), progress,
            Settings(output_dir=Path("outputs")), object_labels=object_labels)
        bar.progress(1.0, text=f"Analysis complete · {result.resolve()}")
        st.session_state["result_dir"] = str(result.resolve())
    except Exception as e:
        bar.empty()
        st.error(f"Analysis could not complete: {e}")

result_dir = st.session_state.get("result_dir")
if result_dir and Path(result_dir, "report.json").exists():
    root = Path(result_dir)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    st.subheader("Analysis results")
    a1, a2, a3 = st.columns(3)
    a1.metric("Verified object tracks", len(report.get("object_comparisons", [])))
    a2.metric("Alignment RMSE", f"{report['alignment']['rmse_scene_units']:.4f} scene units")
    a3.metric("Runtime", f"{report['timings_seconds']['total']:.1f}s")
    st.caption(f"Alignment overlap {report['alignment']['inlier_fraction']:.0%} · device: {report['runtime']['device']} · units are relative")
    visual_review = report.get("visual_review")
    if visual_review:
        st.markdown("### Video frame review")
        st.caption(visual_review.get("method", "Visual review of the sampled before/after frames."))
        for item in visual_review.get("items", []):
            st.markdown(f"- **{item['label']} — {item['status']}**. {item['evidence']}")
            evidence_ids = item.get("frames", {})
            sides = [("A", evidence_ids.get("A", [])), ("B", evidence_ids.get("B", []))]
            sides = [(side, ids) for side, ids in sides if ids]
            if sides:
                cols = st.columns(len(sides))
                for col, (side, ids) in zip(cols, sides):
                    chosen = sorted(set(ids))
                    with col:
                        st.caption(f"Recording {side} reviewed frames: {chosen}")
                        for frame_id in chosen:
                            evidence_frame = root / f"recording_{side}_frame_{frame_id:02d}.jpg"
                            if evidence_frame.exists():
                                st.image(str(evidence_frame), caption=f"Recording {side}, frame {frame_id}", use_container_width=True)
    alignment = report["alignment"]
    # Older report files predate the explicit status and quality limits. Re-evaluate
    # their recorded metrics so an otherwise passing run is not shown as weak.
    overlap = float(alignment.get("inlier_fraction", 0.0) or 0.0)
    rmse = float(alignment.get("rmse_scene_units", float("inf")) or 0.0)
    min_overlap = float(alignment.get("minimum_overlap_fraction", 0.25))
    max_rmse = float(alignment.get("maximum_rmse_scene_units", max(
        2.0 * float(alignment.get("change_threshold_scene_units", 0.08)), 0.1)))
    alignment_ok = (alignment.get("status") == "ok" or (
        alignment.get("success", False) and overlap >= min_overlap and rmse <= max_rmse))
    if not alignment_ok:
        st.error("Alignment is weak. Treat all spatial findings as unreliable; record both clips again with more shared room views.")
    else:
        st.info("Alignment passed the pipeline's geometric checks. This supports comparison, but does not prove each detected change is correct. Inspect the evidence and videos below.")
    if report.get("quality_gates", {}).get("blur_limited_events"):
        st.warning("One or both recordings remained blurred after sharper-frame selection. Event labels were downgraded to uncertain; record clearer footage before relying on a change classification.")
    recognition = report.get("object_recognition", {}).get("open_vocabulary", {})
    if recognition and not recognition.get("loaded"):
        st.warning("Open-vocabulary recognition did not load. Custom object prompts may be missed; check internet/model setup and inspect the run limitations.")
    if visual_review:
        st.info("The reported object changes were checked against the sampled video evidence.")
    if (root / "scene.html").exists():
        st.components.v1.html((root / "scene.html").read_text(encoding="utf-8"), height=650, scrolling=True)
    comparisons = report.get("object_comparisons", [])
    st.markdown("### Detected object category tracks")
    if comparisons:
        for item in comparisons:
            before_frames = item.get("evidence", {}).get("before_frames", [])
            after_frames = item.get("evidence", {}).get("after_frames", [])
            if item.get("status") == "added":
                side_note = " · present in Recording B; absent in A's visible region"
            elif item.get("status") == "removed":
                side_note = " · present in Recording A; absent in B's visible region"
            else:
                side_note = ""
            st.markdown(f"- **{item['label']} — {item['status']}**{side_note} · before frames {before_frames or 'none'} · after frames {after_frames or 'none'}")
            reason = item.get("evidence", {}).get("reason")
            if reason:
                st.caption(reason)
            if item.get("displacement_scene_units") is not None:
                st.caption(f"Category-track position difference: {item['displacement_scene_units']:.3f} relative scene units; monocular scale is ambiguous.")
            useful = [("Before", "A", before_frames), ("After", "B", after_frames)]
            frame_evidence = [(label, side, ids) for label, side, ids in useful if ids]
            if frame_evidence:
                with st.expander(f"Inspect {item['label']} video evidence"):
                    cols = st.columns(len(frame_evidence))
                    for col, (clip_label, side, ids) in zip(cols, frame_evidence):
                        chosen = sorted(set(ids))
                        chosen = chosen if len(chosen) <= 3 else [chosen[0], chosen[len(chosen)//2], chosen[-1]]
                        with col:
                            st.caption(f"{clip_label} recording · sampled frames {chosen}")
                            for frame_id in chosen:
                                evidence_frame = root / f"recording_{side}_frame_{frame_id:02d}.jpg"
                                if evidence_frame.exists():
                                    st.image(str(evidence_frame), caption=f"{clip_label}, frame {frame_id}", use_container_width=True)
    else:
        st.info("No object events passed the repeated-frame and verification checks. Isolated or unmatched detector guesses were omitted.")
    st.caption("Only supported object tracks are listed. Added or removed labels also require repeated interior-frame detections and visible-space evidence from the other recording. Inconclusive regions and isolated detector guesses are omitted.")
    with st.expander("Limitations and run details"):
        st.json({"alignment": report["alignment"], "limitations": report["limitations"], "recordings": {k: {x: v for x, v in val.items() if x != "sampled_frames"} for k, val in report["recordings"].items()}})
    st.success(f"Result folder: `{root}`")
    if (root / "results_bundle.zip").exists():
        st.download_button("Download complete results bundle (.zip)", data=(root / "results_bundle.zip").read_bytes(), file_name="roomchange_results.zip", type="primary")
    for file in ("summary.md", "visual_review.md", "report.json", "changes.csv", "object_comparisons.csv", "scene.html", "recording_a.ply", "recording_b_aligned.ply", "processing.log"):
        p = root / file
        if p.exists():
            st.download_button(f"Download {file}", data=p.read_bytes(), file_name=file)

