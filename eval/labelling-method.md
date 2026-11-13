# How the first labels were made (human method)

Recorded 2026-11-12 from Trevor's description of labelling the first three clips (b_158, b_404, b_1095) on the Rally Labeller page.

**This is a record, not a specification.** It describes what a person did with a human tool. It is not to be hard-coded into the pipeline or turned into a skill, and the pipeline is free to use any method that scores best. Its main value is as evidence that a person can identify every action in a rally from this footage.

## The method

1. Watch the whole clip once at speed to understand the structure of the point.
2. **Bounces:** step frame by frame and mark the frame where the ball changes direction.
3. **Hits, ball visible:** same thing, frame by frame, mark the change of direction.
4. **Hits, ball hidden** by a body or paddle: combine sound and inference. Skip ahead a few frames to where the ball is visible again, work backwards to when the sound was heard, and use the player's body movement to confirm.

## Known limits, in Trevor's words

- The tool had no audio display in frame-by-frame mode. An audio spike could have been used for timing but was only available by ear.
- How frame-accurate the labels are "is debatable".
- The method was slightly inconsistent. If the true event is the instant of the spike (the audio spike, or the kink in the ball's screen Y), some labels sit on the last frame *before* it (ball still visible, not yet turned) and some on the first frame *after* it (ball has already moved).

## What this means for scoring

- Treat each label as "an event happened here, within a few frames either side". The labels are reliable for *whether* and *in what order* events occurred, and less so for exact timing.
- The scorer's matching tolerance (±5 frames, 83 ms) is wider than this before/after inconsistency, so it should not penalise the pipeline for it.
- Timing-error numbers below about 2 frames are inside the label noise and should not be optimised.
