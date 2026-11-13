# Shot-target model: first results on the whole match

Written 2026-11-13. Stage 2 of the simulation plan: given the court at the moment of a hit, predict where the ball goes, as a distribution.

## Data

- `batch/run_front.py` ran the front stages and the event decoder on every baseline-camera shot of 4 s or longer: 108 of 109 finished (`b_2414` failed; not investigated).
- `shots/build.py` turned them into 780 hits from 102 clips. 659 have a target (402 the next bounce, 257 where the opponent volleyed it); 121 have none or fall in clips with too few hits.
- Everything is flipped so the hitter is at the near end.

## Scores

Five folds split by clip, so no clip is in both training and test. Three random fold splits (seeds 7, 8, 9); ranges are across seeds. Gain is log-likelihood per shot over the frequency map; the interval is from resampling whole clips.

| Model | Log-likelihood per shot | Gain over frequency (90% interval, seed 7) | Top zone right (of 16) | Top 3 zones right |
|---|---|---|---|---|
| Uniform over the far half | −6.59 | −0.74 | 9.1% | 32.3% |
| Frequency map (ignores the situation) | −5.83 to −5.85 | 0 | 20.8% | 41.2 to 42.2% |
| Hitter position | −5.72 to −5.74 | +0.13 (0.09 to 0.18) | 20.1 to 21.1% | 49.5 to 50.8% |
| + contact height, after a bounce or not | −5.68 to −5.71 | +0.14 (0.09 to 0.18) | 20.1 to 21.4% | 46.6 to 47.7% |
| + opponent positions | −5.34 to −5.37 | +0.50 (0.46 to 0.54) | 25.8 to 28.4% | 60.2 to 61.4% |

Calibration error of the top-zone confidence is 0.01 to 0.05 for every model.

## What it shows

- Where the opponents stand is by far the most useful input: it adds about three times what hitter position adds.
- The opponent gain is not just "volleys happen where an opponent is standing". Split by target kind it is +0.54 to +0.57 on bounce targets and +0.38 to +0.39 on volley targets.
- Contact height and bounce-or-volley add nothing measurable over position (intervals overlap fully).
- The best model still puts the true zone first only about 27% of the time. It narrows the court; it does not call the shot.

## Limits

- **Labels come from the automatic detector**, which scored 0.50 on the one held-out hand-labelled clip. The median clip has 5 detected hits, which is low for rallies of this length, so hits are being missed. A missed hit turns the next detected event into the wrong target.
- **Dead time is mixed in.** Clips are whole camera shots, not serve-to-point-over.
- **One match, four players.** Nothing here says the model transfers to other players.
- **No hand-checked test.** The numbers measure agreement with the detector's own labels on unseen clips, not with the truth.

## Reproduce

```
uv run python batch/run_front.py 4
uv run python shots/build.py
uv run python shots/model.py 7
```

## Court geometry (added 2026-11-13)

`shots/geometry.py` computes, for every hit and every direction from the contact point, the distance to the net, the net height on that line (34 in at the middle, 36 in at the posts) and the room between the net and the far boundary. Output: `shots/out/geometry.json`.

On the 402 bounce targets (10th / median / 90th percentile):

| Measure | 10% | Median | 90% |
|---|---|---|---|
| Direction used, degrees from straight up the court | −31.8 | −3.3 | 20.7 |
| Room on the line used, ft | 12.6 | 22.2 | 23.1 |
| Share of that room used (1 = on the line) | 0.26 | 0.63 | 0.92 |
| Distance from the bounce to the nearest line, ft | 1.1 | 4.4 | 8.3 |

26 of the 402 bounces are past a line.

**As a way to predict: no gain.** `shots/model_geo.py` stores each past target as (direction, share of room used) instead of court feet, with the same situations and folds. Log-likelihood per shot is 0.17 to 0.20 worse than the court-feet model on all three seeds (90% interval −0.22 to −0.13 on seed 7), and an even mix of the two is no better than court feet alone (−0.02, interval spans zero). Geometry is still the right language for describing a shot's margin and for the limits a simulated shot must respect.

**Opponents during the flight.** For bounce targets, the opponent who ends up nearest starts a median 6.6 ft from the bounce point and is 0.6 ft closer when the ball lands (n = 403).

## Hand check (added 2026-11-13)

`shots/geometry.py` draws two samples (`shots/out/samples.json`, seed 11) and `shots/make_check.py` cuts an overlaid clip for each:

- **Random, 40 shots**: drawn uniformly from all 780 rows. This is the one that estimates how much of the table is right.
- **Geometric, 20 shots**: the bounce shots with the least room on the line they used (4.8 to 10.6 ft). A separate hard suite; it says nothing about the average.

They are shuffled together on the check page (https://claude.ai/artifact/Bp3QSpBZpF7a94tFkMxXpP); answers are stored in its `checks` collection. `shots/out/check/items.json` maps each clip back to its row and sample. Not yet checked.
