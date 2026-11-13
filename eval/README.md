# Event eval: hits, bounces and point ends

Built 2026-11-11. An answer key for the automatic hit/bounce labelling, so changes can be scored instead of judged by feel.

## The set

- 12 baseline-camera shots from the PPA Brisbane 2026 final, 166 s in total. They are the 12 shots the unattended batch run drew at random (seed 2026) from the 47 eligible shots; none was chosen by hand.
- Split at random (seed 7) into dev and test, six each. b_158 and b_404 were then labelled first and read during development, so they moved to dev. **Dev** is now 8 clips (b_158, b_404, b_1095 labelled; b_1422, b_1882, b_1994, b_2133, b_2459 not). **Test** is 4 clips, none labelled or read: b_692, b_1759, b_1844, b_2039. `clips.json` records the split and each clip's first video frame.
- The two rallies the pipeline was developed on (267 s and 1021 s) are not in this set.

Tune only against dev. Run test rarely, to check that a dev gain is real.

## Labelling

Labels are made by a person on the Rally Labeller page: https://claude.ai/artifact/EeJj5XBXwczarBiHpEMZ3e

Four marks: near hit, far hit, bounce, point over. Each is a clip frame at 60 fps. The page does not show the pipeline's guesses, so the labels are independent of it. Labels are stored in the page's database, one document per clip under `labels/`.

**Status (2026-11-12): three clips labelled by Trevor: b_158, b_404, b_1095 (58 events plus 2 point-over marks). How they were made, and how exact they are, is in `labelling-method.md`.**

## Scoring

Export the labels into `eval/labels/` (ask Claude to read the page's `labels` collection with `out_dir` set to `eval/labels`), then:

```bash
uv run python eval/score.py eval/labels dev
```

- A predicted event matches a labelled one of the same type within ±5 frames (83 ms); each label matches at most once. Only clips marked finished are scored.
- Reported: precision, recall and F1 for hits and for bounces, whether the hitter's side is right on matched hits, and timing error.
- **Score** = mean of hit F1 and bounce F1.

The scorer was self-tested on fabricated labels with one known miss and one known false event per clip and counted them correctly.

## Not covered yet

- Point-over marks are collected but nothing is scored against them: the pipeline has no point-end detector.
- Which player hit (as opposed to which side) is not labelled.
- 3D accuracy. This scores events only.
- One match, one venue, one camera, one labeller.

## Hill-climb runner (added 2026-11-12)

```bash
uv run python eval/run_eval.py <round_name> decode "<what changed>"
```

Scores a detector on every labelled clip and writes a round under `.claude/hillclimb/events/` (state in `_state.json`, running summary in `narrative.md`). Its target is pooled **event F1**: a prediction is right if a label of the same class (near hit, far hit, bounce) is within 5 frames. `score.py` is the older scorer and only reads clips ticked finished.

Current detector: `fit3d/decode_events.py`. Baseline 0.518, now 0.875 on the three labelled clips. That is a train-only number; there is no held-out score until test clips are labelled.

## Live play only (2026-11-12)

The first labels also marked dead-ball events. The labeller now has a **Serve** mark (key S) beside Point over, and `run_eval.py` scores only what lies between them. On live play the three labelled clips score 0.620 (baseline) and 0.953 (`decode_events.py`), train-only. The labeller also shows the sound level around the current frame; the level data is `eval/audio_levels.json`.
