# Film review player (alternative product)

Noted 2026-11-11. A simpler product than the 3D twin: watch the real video, but only the points, and jump between them. No 3D.

## What it is

A player for pickleball match video that:

1. Plays the real footage, as clean and smooth as Apple's native video playback.
2. Lets you skip ahead and back by point.
3. Cuts out the dead time between points.

It should not have to be complicated. Everything rests on one capability: knowing where each point starts and ends.

## What was tried before

- Cutting from a transcript of the video.
- Cutting on sounds.

Neither is the plan here. The new basis is the vision work already done in this folder.

## What already exists in this project

- **Cut detection** (`cuts/`): every camera cut and replay wipe in a match, in under three minutes, no learned model.
- **Baseline-camera labelling** (`cuts/`, using `calib/`): which shots are the main behind-the-baseline view. On the Brisbane 2026 final that is 131 shots covering half the match.
- **Shot list** (`cuts/out/IpUxCoBHg_g_shots.csv`): start, end and label for every shot.

On this broadcast, the director cuts to the baseline camera for play and away from it between points, so a baseline shot is already a rough stand-in for "a point".

## What is still missing

A baseline shot is not exactly a point. To get true point boundaries we still need:

- **Point start:** the serve. A baseline shot usually opens with players walking to position and bouncing the ball.
- **Point end:** the moment the rally dies, which can come several seconds before the camera cuts away.
- **Several points in one shot, or one point across shots:** not yet checked how often either happens.
- **Replays:** these re-show a point and should be skippable or marked, not counted as new points.

Candidate signals, none built or tested yet:

- Scoreboard change (the score graphic is on screen in every baseline shot).
- Player movement on the calibrated court: everyone still, then a serve motion.
- Ball in play, once ball tracking exists.
- The "REPLAY" badge for marking replays.

## Playback

Suggestions, not decisions:

- **No re-encoding.** Keep the original file and store points as a list of time ranges. The player seeks; nothing is cut or exported. This keeps quality identical to the source and makes fixes to point boundaries instant.
- **Native on Apple:** AVPlayer (AVKit) plays the list as one continuous timeline using an AVComposition built from the time ranges, which gives system-quality scrubbing and frame stepping.
- **Quick prototype first:** a web page with a `<video>` element and the point list as JSON is enough to test whether the point boundaries feel right, before building anything native.

## Open questions

- Own recordings, pro broadcasts, or both? A fixed tripod video has no camera cuts, so point detection there depends entirely on the missing signals above, not on cut detection.
- How tight should the trims be: a second of lead-in before the serve, or straight to contact?
- Is point-level navigation enough, or should it also filter (by server, by score, by rally length)?
- Mac app, iPhone app, or web?
