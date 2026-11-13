# Event detection hill-climb

Target: pooled event F1 (a prediction is right if a label of the same class - near hit, far hit, bounce - is within 5 frames). Train = the three labelled clips (58 events). **No test clip is labelled yet, so every number here is train-only and directional.**

| round | change | event F1 | hit F1 | bounce F1 | matched / false / missed |
|---|---|---|---|---|---|
| baseline | auto_events.py, stretches of 3 s or more | 0.518 | 0.638 | 0.368 | 22 / 5 / 36 |
| v1 | whole-shot sequence decoder (decode_events.py) | 0.800 | 0.839 | 0.750 | 44 / 8 / 14 |
| v2 | order of play: bounces on opposite sides need a hit between; hit into own side is cheap | 0.842 | 0.875 | 0.800 | 48 / 8 / 10 |
| v3 | sound alone is not a hit when no ball is tracked nearby | 0.855 | 0.900 | 0.800 | 47 / 5 / 11 |
| v4 | velocities measured inside one flight (short hops before a hit) | 0.875 | 0.900 | 0.846 | 49 / 5 / 9 |

Best so far: v4. Nearly all the gain came from v1: replacing the hit-first, greedy labeller with one decode of the whole shot, where every track break or audio onset can be a near hit, far hit, bounce or nothing and the order of play is scored jointly. v2-v4 are small and inside what three clips can resolve.

What is left (14 errors): about 10 are dead-ball events (the server bouncing the ball before the serve in b_404, ball handling after the point in b_158), where the ball is small, slow and often untracked. Two are in-rally bounces a few frames before a hit (b_404 frames 900 and 1149) where the 2D track shows no break. 

Stopped here on purpose: more tuning on three clips without a held-out score would be fitting these clips. Next: label the four test clips, score baseline and v4 on them once.

## Live play only (2026-11-12)

Trevor confirmed the first labels include dead-ball events (bouncing before the serve, after the point). The grader now scores only events in play, between a Serve mark and a Point over mark; for the three clips labelled before the Serve mark existed the windows are set in `eval/run_eval.py` (`LIVE_FALLBACK`). `ALL_EVENTS=1` restores the old scoring.

| variant | event F1 | hit F1 | bounce F1 | matched / false / missed |
|---|---|---|---|---|
| baseline_live | 0.620 | 0.682 | 0.519 | 22 / 5 / 22 |
| v4_live | 0.953 | 0.980 | 0.914 | 41 / 1 / 3 |

Same detector code as baseline and v4; only the grading window changed. Still train-only. Left in b_404: the serve itself (far hit at 365), and two bounces a few frames before a hit (900, 1149).

## First held-out clip (2026-11-12)

b_692 labelled (11 marks, Serve mark, no Point over, not ticked finished; 7 events in play). Scored once, misses not read: baseline 0.50 (3/2/4), v4 0.50 (3/2/4; hits 0.29, bounces 0.80). The train gain did not show up on this clip. Seven events is too few to conclude much, but v4 is not shown to generalise. Rounds baseline_t1 and v4_t1 hold all four clips.
