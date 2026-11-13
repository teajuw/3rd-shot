# 3D ball fit, one dink exchange

Run 2026-11-11 on 20.3–31.2 s of the 33-second shot used in `players/` and `ball/` (PPA Brisbane 2026 final): 21 events, 20 flights, 440 ball observations.

## What was found on the way

- **The file is 50 fps footage in a 60 fps container.** Every sixth frame is a repeat. `segment.py` removes them and puts the track on a true 50 Hz clock; without this the physics sees the ball stop for a frame six times a second.
- **One arc alone does not determine depth from this camera.** Fitting each flight freely (six unknowns, `probe.py`) gave depth uncertainties of 5 to 70 ft on many arcs and impossible answers on short ones. Only long, clean dinks came out sensible. This matches the research warning about the behind-the-baseline view.
- **Automatic hit/bounce labelling was not reliable.** A piecewise-quadratic split of the image track (`segment.py`) finds the break times well, but labelling them from "is a wrist nearby" (`events.py`) calls far too many hits: the ball passes close to players in the image without being near them in 3D. Hit times from the split also land a few frames late, because the ball is usually hidden at contact.

## Method that worked

`fit.py`, with events from `events_hand_rally_267.json`:

- Break times come from the automatic split. Hit/bounce labels and hitters were assigned by hand from the break table and the rules of play. Hit times were then moved to where the ball's sideways image motion reverses.
- Each flight is gravity plus quadratic drag (Cd 0.33, no spin), integrated at 5 ms.
- Anchors: a bounce is on the floor; a hit is within 4.5 ft of the hitter's tracked floor position at a hittable height.
- Weak priors: flights join at events, bounce restitution about 0.66, small event time offsets.
- All 20 flights are solved together (about 19 s).

## Results

| Check | Result |
|---|---|
| Reprojection error | median 2.1 px, rms 4.7 px, 95th percentile 6.5 px |
| Bounces in the correct kitchen (not constrained) | 10 of 10 |
| Net clearance on 10 crossings (not constrained) | −1.0 to +6.0 in |
| Dink launch speeds | 15–22 mph |
| Apex heights | 3.1–4.8 ft |
| Contact heights | 0.9–1.8 ft |
| Contact distance from hitter's feet | 0.2–2.7 ft |

Net clearance and bounce side were not part of the fit, so they are real checks: a wrong depth would send dinks through the net or into the wrong kitchen.

## Known problems

- **Hidden bounce near 24.5 s.** The bounce is behind a player and the hit follows within about 0.1 s. That stretch fits badly (a 0.06 s hop at 60 mph, the next flight at 10.9 px rms, a 1.5 ft join mismatch) and shows as a zigzag in the video.
- **Three other bounce-to-hit hops are very short** (0.07–0.09 s, 23 mph). They have 4 or fewer observations and are not trustworthy.
- **One crossing clears the net by −1.0 in.** That is within the error of the fit, but it shows height is good to an inch or two at best.
- **Events are hand-labelled.** This is the blocker for running whole matches. The break times are automatic; the labels are not.
- **The rest of the shot is not fitted:** the serve and first exchanges (14.8–20.3 s), the final lob after 31.2 s, and the first point (0–8 s).
- **No 3D ground truth.** Low reprojection error and sensible physics do not measure accuracy.
- **No spin.** Topspin and slice change flight; ignoring them biases height and speed by an unknown amount.

## Files

- `out/rally_267_3d.mp4`: broadcast with the path, orbiting virtual camera, side and top views.
- `out/rally_267_fit.json`: every flight and event with positions, speeds, clearances.
- `out/rally_267_path.npz`: 3D ball position per 50 Hz sample.
- `out/rally_267_segments.png`, `out/rally_267_events.png`: the automatic split and the unreliable automatic labels.

## Reproduce

```bash
uv run python fit3d/segment.py ball/out/rally_267_fused_track.npz players/out/rally_267_tracks.npz cuts/out/IpUxCoBHg_g_signal.npz fit3d/out/rally_267
```

```bash
uv run python fit3d/fit.py fit3d/out/rally_267_segments.json fit3d/events_hand_rally_267.json players/out/rally_267_tracks.npz calib/out/t270_calibration.json fit3d/out/rally_267
```

```bash
uv run python fit3d/render.py data/video/IpUxCoBHg_g.mp4 calib/out/t270_calibration.json fit3d/out/rally_267_path.npz players/out/rally_267_tracks.npz fit3d/out/rally_267_fit.json fit3d/out/rally_267_3d.mp4
```

---

# Automatic hit and bounce labelling (added later on 2026-11-11)

`auto_events.py` replaces the hand labels with sound plus the 2D ball track.

## Does sound help?

Yes, as the trigger for hits. Measured on the 11-second dink exchange:

- Every one of the 11 hits has a sharp onset in the 1.5–9 kHz band. Hits are loud (onset strength 1.9–3.8); bounces are audible but weaker (0.9–2.0).
- Audio lags the picture by about 87 ms on this broadcast (sound travel plus broadcast offset). After removing that, audio hit times agree with the fitted hit times to within about 15 ms, better than the 20 ms video sample spacing and usable when the ball is hidden at contact.
- Sound alone is not enough: it produced 3 false hits (a loud bounce and two unidentified noises) and cannot say who hit the ball.

## Method

- **Hits:** onsets of strength 1.6 or more, kept only if the ball's image velocity changes across them (or the ball is hidden). Hitter = nearest wrist in the image. Then the rule of play: consecutive hits must come from opposite sides; where two in a row are on one side, the quieter is dropped.
- **Bounces:** at most one between consecutive hits, at a visual break where the ball goes from moving down the screen to up, keeps its sideways motion, and lands across the net from the hitter. A nearby onset refines the time.
- `fit.py` now takes event times from audio and flags a flight as suspect if it reprojects worse than 6 px or passes more than 4 in below the net tape.

## Results

On the dink exchange, against the hand labels (this window was used to set the thresholds and the audio delay, so it is not an independent test):

| | Found | Notes |
|---|---|---|
| Hits | 11 of 11 | same hitter in all 11; no false hits after the alternation rule |
| Bounces | 8 of 10 | the two missed are the one before the first hit and the one hidden behind a player |
| 3D fit from automatic events | median 2.0 px | net clearance −0.7 to +6.6 in, 8 of 8 bounces in court, 3 of 18 flights suspect |

On play that was not used for tuning, judged by the 3D fit:

| Stretch | Flights | Suspect | What it contains |
|---|---|---|---|
| First point, 0–8.5 s | 11 | 9 | serve, return, drives, fast volleys |
| Second point, 14.8–33.2 s | 28 | 8 | serve and drives (bad), the dink exchange (good), a final lob (bad) |

In the second point, 7 of the 8 suspect flights are outside the dink exchange.

## What this means

- Automatic labelling works for dinking and does not yet work for fast play. It is not ready to run on whole matches.
- Likely causes in fast play, none confirmed: the nearest-wrist rule picks the wrong hitter when players are close to the ball's line of sight; the 2D ball track has gaps and wrong links on fast balls; bounces are missed, which merges two flights into one; and drives carry spin the flight model ignores.
- The suspect flag is useful in itself: bad stretches announce themselves instead of producing a confident wrong replay.
- Which stretches are live play was chosen by me. Telling play from dead time (a player bouncing the ball) is not automated.

---

# Fast play: diagnosis and partial fix (later on 2026-11-11)

## What was wrong

Looking at the first point (serve, return, drives, volleys):

- **Sound is much noisier there.** The dink exchange had quiet surroundings; this point has many loud onsets that are not hits (strength 2–3, in clusters). Loudness alone picked several of them.
- **The "did the ball's path change" test was too weak.** It compared straight-line velocities before and after, which differ on any curved flight, so noise passed it.
- **The hitter's sound and the hitter's hand disagree in useful ways.** Real hits had a visual break in the ball's path within 0.1 s and a wrist within about 1.5 torso lengths; the false ones usually lacked one of the two.
- **A spurious "hit" at the first frame** came from the audio edge of the clip.
- **The bounce search could pick the hit itself** when a bounce and a hit were close together.

## What changed in `auto_events.py`

- A hit now needs all three: an onset, a break in the piecewise-quadratic ball track within 0.1 s (or the ball hidden at that moment), and a wrist within 1.6 torso lengths. Candidates are scored as loudness + size of the velocity change − wrist distance, and that score also decides which of two same-side hits to drop.
- A loud onset at a break that looks like a bounce, with no wrist near, is treated as a bounce.
- Breaks at the ends of a stretch no longer count.
- A bounce candidate within 75 ms of the next hit is ignored.

## Result

Suspect flights (reprojection over 6 px, or more than 4 in below the net tape):

| Stretch | Before | After |
|---|---|---|
| First point, 0–8.5 s | 9 of 11 | 7 of 10 |
| Second point outside the dinks | 7 of 10 | 3 of 10 |
| Dink exchange, 20.3–31.3 s | 3 of 18 | 4 of 17 |

First-point reprojection went from median 8.7 px (rms 81) to median 3.8 px (rms 7.3). So fast play is better but not fixed, and the dinks got slightly worse: the 75 ms rule drops one real dink bounce that comes 70 ms before its hit.

These thresholds were set by looking at this one rally. None of it has been run on another rally or match.

## Still unexplained

- Five first-point flights sit at 6–8 px, just over the line, several with implausibly high net clearance (+19 to +29 in). Not diagnosed. Candidates: spin, which the flight model ignores and which matters on drives; and hit anchors, since a running or jumping far-side player's floor position can be off by feet (the airborne-feet problem in `players/`).
- The final lob after 31.2 s still fits very badly (110 px). Its events are wrong or missing.
- Bounce-to-hit hops shorter than about 0.1 s have too few observations to fit.

---

# Unseen rally, no changes (later on 2026-11-11)

One baseline-camera shot was drawn at random (seed 20261111) from the 27 shots of 12–45 s not used so far: 1021.2–1038.5 s, frames 61,270–62,309. Every stage ran with the code and thresholds exactly as left after tuning on the first rally, using the same camera calibration.

## What happened at each stage

| Stage | Result |
|---|---|
| Players | Tracker did not start until frame 304 (5 s in): it waits for exactly two people per side on court, and 5–6 people were detected at the start. From there, all four players were found in every frame. |
| Ball, 2D | Detected in 62% of frames, 71% with short gaps filled. Lower than the first rally (79% / 89%) because the first 6 s are dead time and the ball rolls away at the end. |
| Segmentation | Three stretches; the main one, 6.1–17.3 s, holds the point. |
| Events | 8 hits and 4 bounces between 7.2 and 14.0 s. |
| 3D fit | 11 flights, median reprojection 2.7 px (rms 7.2). |

## Checks

- **Events against the ball track, by eye** (`out/rally_1021_events.png`): every sideways reversal of the ball has a hit on it and every low point has a bounce. Hitters alternate near/far on all 8 hits. I checked this against the plot, not against the video frames.
- **Sequence makes sense as a point:** serve from the near side at 38 mph, bounce deep on the far side, return at 39 mph, bounce deep on the near side, then a 45 mph drive and volleys.
- **Physics, not constrained in the fit:** 4 of 4 bounces in court; all 7 net crossings clear the tape, by 7 to 31 in; contact heights 0.8–3.5 ft.
- **Suspect flights:** 4 of 11, all for reprojection (8.5, 17.6, 12.2 and 10.4 px), none for passing under the net.

## Reading

- Automatic labelling carried over to an unseen rally without retuning. That is one rally from the same match and camera, so it shows the rules are not specific to the first rally, not that they hold for other venues.
- The same residual pattern as the first rally's fast play appears: about a third of flights reproject at 8–18 px. Events look right there, so the cause is more likely in the flight model or anchors (spin, player floor position, hit timing) than in labelling. Not diagnosed.
- New problems this rally exposed: the player tracker's start condition, and no way to tell the live point from dead time or a rolling ball. I chose the 6.0–17.4 s window by looking at the stretches.
- The last "hit" at 14.0 s is doubtful: the ball then drifts steadily, which looks like a ball rolling away after the point.

Video: `out/rally_1021_auto_3d.mp4`.

---

# Why some flights fit badly: diagnosis (later on 2026-11-11)

`diagnose.py` and `diagnose2.py` refit each flight on its own under competing explanations, on the unseen rally and on the first rally's fast point (19 flights, 7 of them bad at over 6 px).

| Explanation tested | Bad flights it brings under 4 px | Verdict |
|---|---|---|
| A bounce is missing inside the flight | 0 of 7 | ruled out |
| The hit time is wrong (let the clock shift) | 0 of 7 | ruled out |
| A few bad detections dominate the error | error is no more concentrated in bad flights than good ones | ruled out as the main cause |
| Unmodelled force such as spin (add a constant acceleration) | 3 of 7, but the fitted accelerations are often absurd (130–250 ft/s²) | not supported as spin; it is absorbing something else |
| Anchors are wrong (free the flight's ends) | 2 of 7, and freed ends move 5–30 ft with similar image error | depth is weakly held; anchor errors of a few feet matter |
| Detections right at contact are unreliable (ignore 3 samples at each end) | 2 of 7, and every flight improves (good flights from about 3 px to 1.6 px) | supported |

## Fix applied

`fit.py` now ignores ball detections within 60 ms of a hit or bounce (unless that leaves fewer than 4). At contact the ball's blob merges with the paddle or hand, and just after launch it is a long blur streak.

Widening the allowed hitter reach from 4.5 to 6 ft was also tried and changed nothing, so it was left at 4.5.

## Result

| Stretch | Suspect before | Suspect after | Median px before → after |
|---|---|---|---|
| Unseen rally | 4 of 11 | 3 of 11 | 2.7 → 2.2 |
| First rally, first point | 7 of 10 | 3 of 10 | 3.8 → 3.0 |
| First rally, second point | 7 of 28 | 6 of 28 | 2.5 → 2.0 |

Part of this improvement is by construction: the reprojection figures now leave out the trimmed samples, about 15% of observations. The independent checks did not get worse: bounces in court are unchanged (4/4, 3/3, 10/11) and the worst below-net crossing in the first point went from −5.3 to −3.6 in.

The unseen rally is no longer unseen: this change was chosen while looking at it.

## What is still unexplained

Twelve of 49 flights remain suspect. They fall into three groups:

- **Volley exchanges at 6–10 px** (for example 11.0 and 13.2 s in the second rally). Nothing tested brings these under 4 px, including freeing the ends, so the 2D ball track itself does not look like one clean flight there. Not diagnosed; wrong or biased detections during fast exchanges are the leading candidate.
- **The final lob of the first rally** (110 px). Its events are wrong or missing.
- **Very short bounce-to-hit hops**, which have too few samples to fit.

Depth remains the weak direction throughout: freeing a flight's ends moves them by many feet for little change in image error, so everything rests on the bounce points and hitter positions being right.
