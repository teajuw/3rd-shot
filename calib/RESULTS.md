# Experiment 1: calibrate one match from the court

Run 2026-11-11 on the PPA Brisbane 2026 men's doubles final (YouTube `IpUxCoBHg_g`, 1080p60, 46 min).

## What was done

1. `calibrate.py`: 12 rough court marks on one frame (`points_t270.json`) seed a PnP solve; the fit is then refined against the painted lines (about 520 sub-pixel samples). Unknowns are focal length, rotation and position; principal point is fixed at the image centre.
2. The net tape and post tops were **not** used in the fit. They check the height mapping.
3. `track.py`: one frame every 5 s across the match (553 frames). Each is matched to the reference camera by a coarse pan/tilt/zoom search, then refit to the lines with the camera position free and with it fixed.

## Results

Reference frame (t = 270 s):

| Quantity | Value |
|---|---|
| Focal length | 5,490 px (about 20° horizontal field of view) |
| Camera position | 91 ft behind the near baseline, 29 ft up, on the centre line |
| Court line fit | 0.64 px rms, 1.3 px at the 95th percentile |
| Net tape (held out) | within 0.1 in of regulation along its whole length |
| Hand-marked net centre / posts | 33.8 in / 35.9 and 35.8 in (regulation 34 / 36) |
| Height resolution at the net | 3.85 px per inch |
| Radial distortion (k1) | not needed: the fit is unchanged with it |

Across the match:

| Quantity | Value |
|---|---|
| Frames matched to the baseline camera | 281 of 553 (about 51%) |
| Pan / tilt spread | std 0.002° / 0.007° |
| Focal length spread | median 5,491 px, max 5,513 px (0.4%) |
| Line fit, camera position fixed | median 0.65 px, worst 0.87 px |
| Net tape error (held out) | median 0.01 in, std 0.07 in, worst 0.47 in |
| Camera position when left free | std 0.03 / 0.18 / 0.06 ft in x / y / z |

## What this means

- The baseline camera is locked off. One calibration serves the whole match; no per-frame tracking or smoothing is needed for this broadcast.
- The planar court alone pins the focal length here: 281 independent refits agree to about 0.2%, and the held-out net confirms the height scale.
- Only about half the sampled frames come from the baseline camera. The rest are mostly a low side camera at the net post, plus replays and close-ups. Covering whole rallies needs either a second calibration for the side camera (same method, new marks) or accepting baseline-camera rallies only.

## Not established

- The focal-length sweep in `track.py` (refit at fixed f from 3,000 to 10,000 px) was inconclusive: fits at wrong focal lengths lost most of their line samples, so their errors are not comparable. The repeat-fit agreement above is the evidence for f, not that sweep.
- Player height checks were rough (67 in for two players in an athletic stance, from hand-marked head and feet). They show the scale is sane, nothing more.
- One match, one venue. Other venues may use a moving or zooming main camera.
- How the 49% non-baseline frames split between side camera, replays and graphics was not measured; one inspected frame was the side camera.

## Reproduce

```bash
uv run python calib/calibrate.py data/frames/calib/t270.png calib/points_t270.json calib/out
```

```bash
cd calib && uv run python track.py ../data/video/IpUxCoBHg_g.mp4 out/t270_calibration.json out/track 5
```
