# SAM 3D Body benchmark

Run 2026-11-13 on the Mac (Apple M5 Pro, PyTorch 2.15, MPS). Rally at 267 s (frames 16,036 on), 900 frames (every second frame of 30 s), four players. Page: https://claude.ai/artifact/VbDLv6rLHUdAxH8ErtJrj8

## Setup

- Code: `vendor/sam-3d-body` (Meta, commit b5c765a), patched locally: hard-coded `cuda` replaced by the model's device, and the scripted MHR rig kept on the CPU (it uses float64, which MPS rejects). `git diff` in that folder shows the patch.
- Weights: `models/sam3d_body/` (DINOv3-H+ checkpoint and `assets/mhr_model.pt`), downloaded by Trevor from the gated Hugging Face repo and copied over; checksums match Meta's. SAM License.
- Environment: `bodies/.venv` (separate from the project's).
- Boxes come from the existing YOLO keypoints, focal length from the court calibration. Meta's detector and field-of-view estimator are not used.

## Commands

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 bodies/.venv/bin/python bodies/run_sam3d.py data/video/IpUxCoBHg_g.mp4 players/out/rally_267_tracks.npz calib/out/t270_calibration.json bodies/out/rally_267_body --frames 1800 --step 2 --mode body
bodies/.venv/bin/python bodies/render_bench.py data/video/IpUxCoBHg_g.mp4 bodies/out/rally_267_body players/out/rally_267_tracks.npz calib/out/t270_calibration.json bodies/out/rally_267
```

## Results

| Setting | Per player-frame | Per video minute, 4 players, 50 fps |
|---|---|---|
| Body only, MPS (900 frames) | 0.184 s | 37 min |
| Body + hand refinement, MPS (30 frames) | 0.675 s | 135 min |
| Body only, CPU (6 frames) | 20.6 s | not usable |

- A body came back for 3,599 of 3,600 player-frames.
- Poses match the video on stills for near and far players (padded boxes 300-315 px and 215-245 px tall).
- Position: across court agrees with the court foot tracks to 0.05-0.13 ft (median). Along court the median gap is 1.1-4.3 ft and the worst tenth exceed 3-7 ft. Use the court tracks for position.
- Shape and scale parameters vary frame to frame (std about 0.1), so they need fixing per player.

## Not done

- GVHMR control: needs SMPL-X files from a personal registration.
- No ground truth; no smoothing; no 3D court view; no paddle.
- Far players in this rally are larger than the 80-150 px case the research report worried about.

Full numbers: `out/rally_267_stats.json`.

## GVHMR comparison (added 2026-11-13)

- Code: `vendor/GVHMR` (commit ee960bb), unpatched. `bodies/run_gvhmr.py` redirects every `.cuda()` to MPS, stubs `wis3d`, skips the tracker/SLAM imports, and uses `bodies/shims/pytorch3d` (rotation conversions only) in place of pytorch3d.
- Weights in `vendor/GVHMR/inputs/checkpoints/`: GVHMR, HMR2 and ViTPose checkpoints from the project's Google Drive, SMPL-X v1.1 and SMPL v1.1.0 from Trevor's registrations. SMPL/SMPL-X may not be redistributed, so GVHMR output must not go on a shared page as a mesh file.
- Same frames, boxes and focal length as the SAM 3D Body run; static-camera mode.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 bodies/.venv/bin/python bodies/run_gvhmr.py data/video/IpUxCoBHg_g.mp4 players/out/rally_267_tracks.npz calib/out/t270_calibration.json bodies/out/rally_267_gvhmr --frames 1800
```

| Measure (range over four players) | GVHMR | SAM 3D Body |
|---|---|---|
| Time per player-frame | 0.144 s | 0.184 s |
| Wrist/ankle acceleration rms, hip-relative, 30 fps | 10-17 m/s² | 22-48 m/s² |
| Departure from 5-frame average | 0.9-1.5 cm | 1.4-2.6 cm |
| Wrist speed, 99th percentile | 2.9-4.2 m/s | 3.9-6.0 m/s |
| Along-court gap to court tracks, median | 8.6-16.3 ft | 1.1-4.3 ft |

GVHMR's 517 s split: ViTPose 332 s, HMR2 features 177 s, GVHMR itself 4.6 s. The two models' hip-relative wrists differ by 5-6 cm and ankles by 8-11 cm (median). No ground truth, so the lower GVHMR wrist speed is either less jitter or damped swings; not resolved.

## Steadying and court placement (added 2026-11-13)

`bodies/ground.py` turns the raw SAM 3D Body run into bodies on the court; `bodies/eval_ground.py` measures it; `bodies/export_view.py` packs it for the browser. 3D page: https://claude.ai/artifact/9igWnu9pQp8yLM9s96UtJi

```bash
bodies/.venv/bin/python bodies/ground.py bodies/out/rally_267_body players/out/rally_267_tracks.npz calib/out/t270_calibration.json bodies/out/rally_267_grounded
bodies/.venv/bin/python bodies/eval_ground.py bodies/out/rally_267_body bodies/out/rally_267_grounded players/out/rally_267_tracks.npz calib/out/t270_calibration.json bodies/out/rally_267_gvhmr
bodies/.venv/bin/python bodies/export_view.py bodies/out/rally_267_grounded fit3d/out/rally_267_path.npz bodies/out/view
```

Method, per player: (1) one shape for the shot, the median of the per-frame estimates; (2) joint angles converted to the model's wrap-free form, median of 3 then zero-phase 6 Hz low-pass, body rebuilt with the fixed shape (`bodies/mhr.py`, exact round trip); (3) the body keeps the image direction the model gave it and is slid along that ray until its lowest point touches the floor, with one size factor per player chosen so this agrees with the model's own distance on the median frame (1.039, 1.004, 0.978, 1.034); the ray distance is then low-passed at 3 Hz.

| Measure (range over four players) | Raw | After | GVHMR |
|---|---|---|---|
| Wrist/ankle acceleration rms, hip-relative | 31-46 m/s² | 11-17 m/s² | 10-17 m/s² |
| Wrist/ankle speed, 99th percentile | 3.5-5.4 m/s | 2.9-4.5 m/s | 2.7-3.8 m/s |
| Gap to YOLO 2D keypoints, % of box height (median) | 1.3-1.5 | 1.4-1.8 | - |
| Along court vs court foot tracks (median) | 1.1-4.3 ft | 0.2-0.4 ft | 8.6-16.3 ft |
| Across court vs court foot tracks (median) | 0.05-0.13 ft | 0.02-0.03 ft | 0.2-0.6 ft |

Filter sweep (no filter, 4, 6, 10 Hz): the 2D-keypoint gap stayed at 1.37-1.39% for the first player and similarly flat for the others, so the small rise from raw comes from fixing the shape and placing on the floor, not from the filter. 6 Hz was kept because it matches GVHMR's steadiness.

Known gaps: planted feet slide 0.5-0.9 ft/s (no foot pinning); airborne players are pushed away from the camera (the method assumes floor contact); 99th-percentile speeds fell about 15% and it is not known how much was real; tuned and checked on one rally; the browser mesh is reduced to 1,500 vertices at 15 key frames a second.

## Paddle v1 and whole-shot ball (added 2026-11-13)

`bodies/paddle.py` merges the ball from existing 3D fits and puts a rigid paddle in each player's hand. Shown on the 3D page.

```bash
bodies/.venv/bin/python bodies/paddle.py bodies/out/rally_267_grounded bodies/out/rally_267_paddle fit3d/out/rally_267 fit3d/out/rally_267_auto_pointA fit3d/out/rally_267_auto_pointB
bodies/.venv/bin/python bodies/export_view.py bodies/out/rally_267_grounded - bodies/out/view --paddle bodies/out/rally_267_paddle
```

- **Ball:** the shot holds two points (0.9-7.6 s and 15.0-33.0 s) with dead time between. Merging the hand-labelled fit (20.3-31.2 s), the automatic point A fit and the automatic point B fit gives a ball for 21.8 of the 30 s. Point B's automatic fit is poor in places (rms 38 px, 95th percentile 106 px) and supplies 15-20 s.
- **Hand:** right for all four (the wrist nearer the ball at each player's hits; player 1 was 6 to 2).
- **At hits:** face normal = direction of the ball's velocity change (frictionless flat face); 23 of 26 hits used (3 had the ball over 3 ft from the grip or no velocity). Blended over ±0.25 s. The handle is aimed at the contact point only when ball and hand agree in depth (4 hits); otherwise it keeps the default handle, squared to the face.
- **Between hits (default, not measured):** handle along the forearm with the tip raised, face towards the net and slightly open. At the 23 hits this default was 17.5° (median) from the ball-derived face; near players 19.7°, far 16.8°. Not independent: the default was chosen after seeing these hits.
- **Tried and dropped:** setting the paddle from the body model's finger and wrist keypoints (handshake grip). The ball-derived face was a median 60° from it and scattered in the hand frame, so the body-only hand pose carries no usable paddle information on this footage.
- **Ball and hand disagree in depth:** at hits the ball is a median 1.39 ft from the grip and 1.06 ft off the plane through the grip square to the face, i.e. mostly along the court. The hand position (good to a few inches along court) is a better contact anchor than the current "within 4.5 ft of the feet" rule in `fit3d/fit.py`; feeding it back is not done.

Not done: any check of the paddle against the image (keypoints or masks); spin; backhand/forehand distinction; a 3D ball for 7.6-15 s (dead time) and before 0.9 s.

## Paddle v2: shape and motion fixes (added 2026-11-13)

After Trevor's review of v1 (boxy shape; clipping, flipping, odd motion). Same commands as v1; `bodies/paddle.py` was rewritten and this supersedes the v1 method notes above where they differ.

- **Shape (page only):** rounded outline with throat, edge guard and grip, 15.75 x 7.9 in, 16 mm thick.
- **Grip point:** in the palm (mean of wrist and the index and pinky knuckles), not offset from the wrist.
- **Handle:** continues the forearm with the wrist cocked 30° towards up, filtered at 6 Hz. The handle has priority, so the paddle moves with the arm.
- **Roll about the handle:** one angle per frame, measured from a reference carried along the handle's path without twist. It is pulled weakly towards net-facing (weight scaled by how well defined that is), pinned at hits to the ball-derived normal's projection, treated modulo 180° (two identical faces), and solved for the whole rally with first- and second-difference penalties.
- **At hits:** the remaining tilt takes the face to the ball-derived normal by the shortest turn, carrying the handle; held ±0.05 s, faded by ±0.30 s. Final 8 Hz low-pass.

| Check (range over four players) | v1 | v2 |
|---|---|---|
| Largest turn speed in any frame | 3,200-5,000 °/s | 770-1,270 °/s |
| Turn speed, 99th percentile | 740-1,190 °/s | 540-870 °/s |
| Forearm turn speed, 99th percentile | 410-800 °/s | same |
| Handle reversals within 0.1 s | 2-9 | 0-1 |
| Frames with the face inside the body (capsule model) | not measured | 0-0.6% |
| Largest forearm-to-handle angle | up to 170° | 87-94° |
| Face error at hits after filtering | - | median 0.1°, max 0.3° |

Two designs were tried and dropped on the way: face-priority default (net-facing face, handle squared to it) folded the paddle back along a hanging forearm and put the face inside the body in up to 41% of frames; handle-priority without the roll solve still flipped the face 1-3 times per player when the arm pointed at the net.

Still true: nothing measures the paddle in the image; between hits the pose is a default; the ball is about 1.4 ft from the grip at hits. All numbers are from this one rally and the checks are about plausibility of motion, not accuracy.
