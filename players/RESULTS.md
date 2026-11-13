# Player dots

Run 2026-11-11 on one baseline-camera shot of the PPA Brisbane 2026 men's doubles final: frames 16,036–18,025 (267.3–300.4 s, 1,990 frames at 60 fps).

## Method

1. Pose model (`yolo11m-pose`, Ultralytics, AGPL-3.0) on every frame at 1280 px, on the Mac's GPU (MPS).
2. Each player's ankle pixels are intersected with a plane 0.3 ft above the floor using the camera from `calib/`. Across-court position is the mean of both ankles; depth uses the ankle nearer the camera plus half the depth gap, capped at 1.5 ft, so a lifted foot cannot drag the point away.
3. Four tracks (two per side) are followed by nearest-neighbour assignment with a distance gate; players are not allowed to cross the net.
4. Samples more than 1.5 ft from a quarter-second rolling median are dropped, gaps are interpolated, and each track is low-pass filtered at 2.5 Hz, forward and backward.

## Results

| Player (side at start) | Found | Raw minus smooth (rms) | Peak speed | Distance |
|---|---|---|---|---|
| near-left | 100% | 0.10 ft | 9.4 ft/s | 75 ft |
| near-right | 100% | 0.12 ft | 18.9 ft/s | 120 ft |
| far-left | 99.8% | 0.20 ft | 26.1 ft/s | 122 ft |
| far-right | 100% | 0.32 ft | 17.9 ft/s | 129 ft |

- Detection took 76 s for 1,990 frames (38 ms per frame), so about 2.3× slower than real time at 60 fps.
- The model found exactly four people in 1,883 frames and five in 107; the tracker ignored the fifth.
- Both ankles were confidently located in over 99% of frames for every player.
- Smoothing cut frame-to-frame acceleration from 450–1,260 ft/s² to 18–37 ft/s².
- The two far players swapped sides early in the rally (stacking). Closest approach between them was 3.4 ft and the tracks did not swap identities.

## Known problems

- **Airborne feet.** Position comes from ankles on an assumed floor plane. With both feet off the ground mid-stride, the point lands too far from the camera. The far-left player shows 20 frames above 20 ft/s (peak 26.1), which is not believable and is most likely this effect. Far players are affected most because one pixel there is about 0.1 ft of depth.
- **No ground truth.** Nothing here measures position error. The jitter figures show noise, not accuracy.
- **Names are positions at the first frame**, not identities. Players who start on the left are called "left" for the whole shot.
- **One shot only.** Tracking restarts per shot; nothing links a player across cuts.
- **Licence.** Ultralytics models are AGPL-3.0, which matters if this is ever distributed.

## Files

- `out/rally_267_dots.mp4`: broadcast with ground rings, beside a top-down court with dots and trails.
- `out/rally_267_tracks.npz`: raw and smoothed positions, plus all 17 pose keypoints per player per frame.
- `out/rally_267_stats.json`: the table above with more columns.

## Reproduce

```bash
uv run python players/track_players.py data/video/IpUxCoBHg_g.mp4 calib/out/t270_calibration.json 16036 1990 players/out/rally_267
```
