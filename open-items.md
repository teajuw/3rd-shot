# Open items

## Paddle outlines: something has to know what a paddle looks like (opened 2026-11-13)

**Where it stands.** SAM 2 outlines the paddle cleanly when given a click that lands on it, even on a motion-blurred paddle. Picking the right outline automatically from size and shape failed (shorts, banners and court were chosen instead). Far players' paddles are clearly visible in crops. Details: `paddle_seg/RESULTS.md`.

**What is missing.** A source of the click (or box) on the paddle.

**Options, cheapest first.**

1. **SAM 3 with a text prompt** ("pickleball paddle"). Needs an access request at https://huggingface.co/facebook/sam3 and one file, `sam3.pt` (3.4 GB), downloaded on the laptop and copied to the mini as before. Ultralytics 8.4 (already installed) has SAM 3 predictors. Untested on paddles this small, and untested on the Mac's GPU.
2. **A vision model that clicks on the paddle** (Trevor's idea, 2026-11-13: a computer-use agent or a decision model trained or prompted to click paddles for labelling). A model looks at a hand-centred crop and returns a point or box; SAM 2 turns that into the outline. Claude's read: this makes sense, and a model called directly on crops is a better fit than a computer-use agent, because driving a labelling UI by screenshots is slow and adds nothing here. Sending frames to an outside model would need Trevor's OK; a local model avoids that.
3. **A small trained detector.** Use the outlines from 1 or 2 as training labels for a fast paddle detector that runs on every frame. This is the end state either way; 1 and 2 are ways to get its labels without hand-labelling thousands of frames.

**Whichever is used, check it the same way:** a random sample of player-frames labelled by hand (a click on the paddle, or "not visible"), kept apart from anything used for training or prompt tuning.

**Then.** Fit the known rigid paddle shape to each outline, anchored at the hand, and combine with the hit physics. That is what replaces the current default pose between hits.

## Wanted ability: predict where a player will hit, as a distribution (noted 2026-11-13)

Trevor, looking at Vid2Player3D's "hit an incoming ball to the target" figure: from 2D pickleball video, detect where the player is going to hit to, even when there is more than one possibility; modelling a distribution over targets "would be amazing".

Claude's notes, not yet researched or built:

- Vid2Player3D's figure shows the reverse problem (a simulated player is given a target and hits to it). Predicting the target is a forecasting problem and does not need the physics simulation.
- The pipeline already produces the training examples: each hit has a hitter, the four player positions and poses, the incoming ball, and where the ball then went (the next bounce or the point where an opponent intercepted it).
- A natural form is a probability map over the opponent's court (or a mixture of a few landing zones), conditioned on what is visible up to contact, and optionally on the player.
- It needs many shots, so it depends on the hit and bounce detector working unattended (currently 0.95 on tuned clips, 0.50 on the one held-out clip) and on running the pipeline over whole matches.
- The research run "Racket sport physics simulation and RL" was asked to cover shot-target prediction in its imitation thread.
