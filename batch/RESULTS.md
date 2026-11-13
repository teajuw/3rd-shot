# Batch run: 12 random rallies, no hand steps

Run 2026-11-11. `run_many.py 12 2026` drew 12 baseline-camera shots at random (seed 2026) from the 47 shots of 8–35 s in the PPA Brisbane 2026 final, excluding the two shots used for development. Every stage ran with the code and thresholds as they stood, on every stretch of ball visibility of 3 s or more. Nothing was chosen or adjusted by hand.

## What ran

- 12 shots, 167 s of footage, 44 minutes of compute (about 16× slower than real time).
- All three front stages (players, ball, segmentation) completed on all 12. The player tracker started on time in all 12.
- Ball detected in a median 68% of frames per shot.
- 15 stretches of 3 s or more; 10 produced at least 3 events and a completed 3D fit. The other 5 produced 0 or 1 hit. Whether those were dead time or missed play was not checked.

## Flight quality

73 flights were fitted.

| Measure | This batch | The two development rallies |
|---|---|---|
| Flights flagged suspect | 32 of 73 (44%) | 12 of 49 (24%) |
| Median reprojection per flight | 4.6 px | 2.0–3.0 px |
| Net crossings below the tape | 10 of 50 (7 by more than 4 in) | 1 by more than 4 in |
| Bounces in court | 21 of 25 | 17 of 18 |
| Hits with contact at floor level (under 0.3 ft) | 14 of 58 | not counted |

Per stretch, results range from good to useless:

| Shot | Window (s) | Hits | Bounces | Flights | Suspect | Median px |
|---|---|---|---|---|---|---|
| b_2133 | 0.0–8.0 | 8 | 3 | 10 | 1 | 2.0 |
| b_1882 | 0.0–8.8 | 8 | 3 | 10 | 3 | 3.3 |
| b_2459 | 0.0–6.7 | 4 | 0 | 3 | 1 | 4.5 |
| b_1422 | 8.9–15.4 | 6 | 4 | 9 | 3 | 4.6 |
| b_1095 | 0.0–8.2 | 7 | 3 | 9 | 3 | 5.5 |
| b_692 | 14.8–19.4 | 4 | 1 | 4 | 3 | 7.2 |
| b_404 | 5.9–21.0 | 11 | 6 | 16 | 9 | 11.8 |
| b_1844 | 0.0–7.5 | 4 | 3 | 6 | 4 | 12.0 |
| b_1844 | 8.2–14.0 | 2 | 2 | 3 | 2 | 153.3 |
| b_1422 | 16.0–19.3 | 4 | 0 | 3 | 3 | 182.0 |

Suspect rate by flight type: hit to bounce 13 of 24, hit to hit 10 of 27, bounce to hit 8 of 21. No type is clean.

## Reading

- **The pipeline runs end to end unattended**, which it did not at the start of the day.
- **Quality on random rallies is clearly worse than on the rallies it was developed on.** Suspect flights are 44% against 24%, and physically impossible results appear (10 crossings under the net tape, 14 contacts at floor level). The development numbers were optimistic.
- **A few stretches work well** (b_2133 and b_1882: 20 flights, 4 suspect), so the approach is sound when its inputs are right.
- **Two stretches are garbage** (median error over 150 px). They are the second stretch in their shot, so they are probably after the point ended, but that was not checked. There is still no test for live play versus dead time.

## What this run cannot tell

- There are no hand labels for these rallies. "Suspect" means the fit disagrees with itself or with physics. A flight can pass and still be wrong, and nothing here measures how many hits or bounces were missed or invented.
- Why the bad stretches are bad was not investigated.
- All of this is one match, one venue, one camera.

## Files

- `out/results.json`: per-shot record of every stage.
- `out/b_*_s*_fit.json`: fits. `out/b_*.log`: full logs.

## Reproduce

```bash
uv run python batch/run_many.py 12 2026
```

```bash
uv run python batch/aggregate.py
```
