# Cut detection and baseline-camera labelling

Run 2026-11-11 on the PPA Brisbane 2026 men's doubles final (`IpUxCoBHg_g`, 165,762 frames at 60 fps). Deterministic only; no learned model.

## Method

1. `signal.py`: decode the match at 96 × 54 and record each frame's mean colour change from the previous frame, plus a colour-histogram distance. 59 s for the 46-minute match (about 2,800 frames/s).
2. `detect.py`: frames with a large change are grouped into events. A 1–2 frame event is a hard cut; a longer one is a transition (the replay wipe). Transitions are grown outward while change stays elevated, so the gentle first and last wipe frames are included. Shots are the stretches between events.
3. Court score, 10 times a second at half resolution: the fraction of points along the calibrated court lines that sit on a thin bright line. It needs the calibration from `calib/` and nothing else. About 1 min 45 s for the match.

## Results

| | Count | Time | Share of match |
|---|---|---|---|
| Hard cuts | 183 | | |
| Transitions (wipes, 14–18 frames each) | 168 | | |
| Baseline-camera shots | 131 | 1,385 s | 50.1% |
| Other shots (side camera, close-ups, replays) | 132 | 1,246 s | 45.1% |
| Fragments under 1 s (wipe graphics, flash cutaways) | 89 | 74 s | 2.7% |
| Shots where the court score flips partway | 0 | | |

Baseline-camera shots of 3 s or longer: 119, totalling 1,362 s. Median baseline shot is 7.5 s, longest 60.3 s.

The court score is cleanly two-sided: of 27,627 samples, 13,652 fall below 0.1 and 13,850 above 0.7, with 125 in between.

## Checks

- **Precision, by eye:** 16 hard cuts picked at random (seed 7) all show a real change of camera or content (`out/check_cuts.jpg`).
- **Missed cuts, by eye:** 8 "other" shots of 3 s or longer picked at random (seed 11) all start and end on the same camera (`out/check_misses.jpg`).
- **Cross-check from the court score:** no shot mixes baseline and non-baseline samples, and no two consecutive shots are both baseline. So the change signal missed no boundary into or out of the baseline camera at 10 Hz resolution.
- The 30-second demo clip (262–292 s) contains a 4-second cutaway at 263.3–267.2 s, found as three hard cuts.

## Not established

- No full hand-labelled ground truth exists, so there is no measured recall or precision beyond the samples above (16 cuts, 8 shots).
- A cut between two non-baseline cameras that the change signal missed would not be caught by the court-score cross-check. None was seen in the 8-shot sample.
- 31 frames inside shots still show a change above 4 (the hard-cut threshold is 8). The eight largest were inspected before the transition-growth fix and were all wipe edges; the remainder were not looked at.
- "Other" is not broken down into side camera, close-up and replay. Replays show a "REPLAY" badge that a template match could find; that was not built.
- Thresholds were set on this one match. Slow dissolves, which this broadcast does not use, would need a different rule.
- The baseline label depends on this match's calibration; a new match needs its own.

## Files

- `out/IpUxCoBHg_g_shots.csv` and `.json`: every shot with start, end, label, court score and how it ends.
- `out/IpUxCoBHg_g_signal.npz`, `out/IpUxCoBHg_g_court.npz`: cached per-frame signals.

## Reproduce

```bash
uv run python cuts/signal.py data/video/IpUxCoBHg_g.mp4 cuts/out/IpUxCoBHg_g_signal.npz
```

```bash
uv run python cuts/detect.py data/video/IpUxCoBHg_g.mp4 cuts/out/IpUxCoBHg_g_signal.npz calib/out/t270_calibration.json cuts/out/IpUxCoBHg_g
```
