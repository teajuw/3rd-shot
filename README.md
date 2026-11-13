# 3rd-shot

Turning single-camera broadcast video of a pickleball match into a 3D digital twin: the court, four moving bodies, their paddles and the ball, in real units. Runs offline on an Apple Silicon Mac. The longer aim is a simulator that learned agents can play in.

This is a research log as much as a codebase. Every stage has a `RESULTS.md` that says what was measured, including what did not work.

## What works today

| Stage | Folder | State |
|---|---|---|
| Camera calibration and shot cuts | `calib/`, `cuts/` | Works on the one broadcast angle used |
| Player positions on the court | `players/` | Works |
| 2D ball tracking | `ball/` | Works; gaps where the ball is hidden |
| Hits and bounces | `fit3d/decode_events.py`, `eval/` | 0.95 on the clips it was tuned on, 0.50 on the one held-out clip |
| 3D ball path | `fit3d/` | Works on hand-checked stretches; automatic fits are uneven |
| 3D bodies (SAM 3D Body, grounded and smoothed) | `bodies/` | Works; 0.18 s per player per frame |
| Paddles | `bodies/paddle.py` | Plausible, inferred from the arm and the ball; not measured from pixels |
| Ball simulator | `sim/` | Built; its landing-prediction gate is not passed |
| Shot-target model | `shots/` | Scored on 659 shots; labels are automatic and unchecked |

Speed: about 6 hours for 10 minutes of footage with bodies on every second frame (`optimize.md`).

## Layout

- `calib/` `cuts/` `players/` `ball/` `audio/`: the front stages
- `fit3d/`: segmentation into flights, event decoding, physics-based 3D ball fit
- `bodies/`: SAM 3D Body and GVHMR runners, grounding, paddle, web export
- `sim/`: ball flight, bounce and paddle-impact model, calibration against tracked flights
- `shots/`: shot table, target-distribution models, court geometry
- `batch/`: whole-match runs
- `eval/`: hand labels, scorer and the labelling page
- `research_notes/`, `reports/`: literature reviews behind the design choices
- `open-items.md`, `optimize.md`: what is parked and where time could be saved

## Not in this repository

- **Match video** and anything cut from it. Outputs are regenerated from your own copy under `data/video/`.
- **Model weights.** SAM 3D Body, SMPL and SMPL-X are licensed to whoever downloads them. Put them under `models/` and `vendor/GVHMR/inputs/checkpoints/`.
- **Vendored code.** Clone into `vendor/`:
  - `facebookresearch/sam-3d-body` at `b5c765a`, then apply `bodies/patches/sam-3d-body.patch` (runs it on Apple GPUs)
  - `zju3dv/GVHMR` at `ee960bb` (unpatched; `bodies/run_gvhmr.py` adapts it at run time)

`bodies/patches/sam-3d-body.patch` modifies Meta's SAM 3D Body code and quotes a few lines of it; that code remains under Meta's SAM License (see `LICENSE` in their repository). SMPL, SMPL-X and GVHMR have their own licences, some non-commercial.

## Setup

```bash
uv sync
```

The body models use a second environment at `bodies/.venv` (torch, lightning, smplx); see `bodies/RESULTS.md`.

## Limits

One match, one venue, one camera angle. Little of this has been checked against hand labels, and the write-ups say where.
