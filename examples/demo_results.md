# Example output: local room-pair replay

This example summarizes the replay of [before.mp4](before.mp4) and [after.mp4](after.mp4). The clips are included as example inputs. Extracted image previews are not committed. The values below are model outputs from that pair, not ground-truth accuracy measurements.

## Reported object tracks

| Category | Result | Evidence summary |
|---|---|---|
| Bag | Added | Recording B detections span sampled frames 15–31 (17 consecutive observations). Three consecutive detections are interior to the frame; 17 projected track samples had visible-empty support in A. |
| Bottle | Removed | Recording A detections span frames 0–4 and 20–31 (17 observations total, with a 12-frame consecutive run and 8 consecutive interior detections); 17 projected track samples had visible-empty support in B. |
| Chair | Unchanged | Persistent category tracks in both clips; 3D centers differ by 0.2677 relative scene units, within the configured 0.30 matching threshold. |
| Laptop | Moved | Persistent tracks in both clips; 3D centers differ by 0.373 relative scene units, beyond the 0.30 matching threshold. |

Alignment passed with 0.786 inlier overlap and 0.0322 relative-unit RMSE. Analysis took 77.6 seconds on CUDA in this local run. The scene scale is arbitrary; these distances are not metres.

The current code requires at least three consecutive interior detections in both recordings before reporting a moved or unchanged match. A desk track in this replay was omitted because its detections touched the frame edge throughout; it was not turned into an event based on that incomplete view.

## Interpretation limits

- These statuses are model-supported hypotheses. The pair has no independent frame-by-frame ground-truth annotations, so it cannot establish precision or recall.
- Bag and bottle additions/removals also use approximate VGGT-based visibility projections. Pose, depth, or registration errors can still affect those decisions.
- Moved and unchanged are based on category-track proximity, not guaranteed exact physical-instance identity.
- The example does not test every supported object category or room layout.

## Reproduce

On Windows, run `START_HERE.bat`, upload `examples/before.mp4` as Recording A and `examples/after.mp4` as Recording B, leave the optional extra-object prompt field blank for built-in categories, and choose **Analyse Scene Changes**. Review the event evidence and alignment status. The run creates an interactive 3D scene and results bundle under `outputs/`.
