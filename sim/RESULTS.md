# Ball simulator, stage 1

Built 2026-11-13. `sim/ballsim.py` is an event-based ball simulator; `sim/calibrate.py` fits its shared physics to footage and tests it on held-out flights and bounces.

## What the simulator does

- **Flight:** gravity, quadratic drag (Cd 0.33) and lift from spin (lift coefficient 0.195 x spin ratio, from Steyn et al.), spin constant in flight.
- **Bounce:** an impulse at the floor with restitution that can fall with impact speed, and friction that slides or grips (thin-shell inertia), which also changes spin.
- **Net:** a flight crossing the net plane below the tape stops.
- **Paddle:** an impulse against a moving flat face with its own restitution and friction. Written, not calibrated: nothing measures the paddle yet.
- Sanity checks: a 78 in drop rebounds 30.9 in (certification window 30-34 in); a spinless ball that grips keeps exactly 60% of its horizontal speed.

## Data

45 or so flights between trustworthy events: rally 267 (hand-assigned events) and b_158, b_404, b_1095 (Trevor's labels, live play only). b_692 is a held-out clip for the event detector and was not used. That gives 17 bounces with a visible hop afterwards and 25 flights long enough for a landing test. Each test splits its cases at random, half to fit and half to test; three seeds (11, 12, 13) were run. Run: `uv run python sim/calibrate.py <seed>`; outputs in `sim/out/`.

## Test 1: what the ball does after a bounce (held out, 9 bounces per split)

Fit the flight into the bounce, apply the bounce model, fly on, compare with where the ball was seen. Nothing after the bounce is fitted. Error is pixels, per bounce.

| Model | Seed 11 median | Seed 12 median | Seed 13 median | Better than old, of 9 |
|---|---|---|---|---|
| old: keeps 66% vertical, 80% horizontal | 18.5 | 21.9 | 18.5 | - |
| old+: same form, two numbers fitted | 9.5 | 12.7 | 16.5 | 8, 7, 7 |
| new: friction bounce, fitted | 9.1 | 14.5 | 18.3 | 8, 6, 7 |
| new+spin: plus per-flight spin | 16.0 | 8.9 | 9.4 | 8, 8, 9 |

- **The clear finding:** the ball keeps about 55-59% of its horizontal speed through a bounce, not 80%. The fitted values were 0.55, 0.59 and 0.58. Vertical restitution came out at 0.61-0.64 against the assumed 0.66. Impact speeds here are about 15 ft/s (dinks).
- **This matches the physics with no fitting:** a ball that grips the court keeps 3/5 of its horizontal speed. The friction model reproduces the fitted constant; it is not measurably better than the fitted constant on this data.
- **Spin:** better on two splits and worse on one, with fitted parameters that hit their limits. Not adopted. Monocular footage of dinks does not pin spin down.
- Nine test bounces per split is very few, and all come from four clips of one match.

## Test 2: predicting the landing from the first 60% of a flight (held out, 13 flights per split)

| Model | Landing time error, median | Landing pixel error, median |
|---|---|---|
| old: Cd 0.33, no spin | 104-173 ms | 83-172 px |
| drag fitted on the training half | 113-211 ms | 105-186 px |
| with per-flight spin | 94-144 ms | 61-77 px |

- **No model predicts landings well from a partial flight.** With one camera, the first part of an arc does not fix how far away the ball is, so every model is off by a tenth of a second and several feet. This is the depth problem again, not a drag or lift problem.
- The differences between models are inside the noise for 13 flights. The fitted drag coefficient changed with the split (0.27, 0.40, 0.27), so the footage gives no reason to move it from 0.33.
- An earlier version of this test leaked the answer twice (the starting guess used the landing pixel, and the fit was told the ball stays airborne until the true landing time). Both were removed; the numbers above are after the fix.

## What changed in the pipeline

`fit3d/fit.py`: the weak bounce prior on horizontal speed now uses 0.57 (was 0.8; override with `K_TAN`). On rally 267 the reprojection error is unchanged (rms 4.51 px, median 1.61), all 10 bounces still land in the correct kitchen, and the one net crossing that cleared by -0.3 in now clears by +0.2 in.

## Gate

The stage 1 gate was "landing and time-of-flight error on held-out rallies beats the drag-only model". Not passed: the landing test cannot tell the flight models apart with this data. The bounce part did improve on held-out bounces. Lift and spin are implemented but unproven here.

## What would move this forward

- **The paddle as the contact anchor.** The largest error in the ball path is depth at hits. The hand is now placed to a few inches; using it in `fit.py` in place of "within 4.5 ft of the feet" is the next step and should also stop the ball missing the paddle.
- **More flights, and faster ones.** All data here are dinks at about 15 ft/s impact. Drives and serves are where drag and lift matter.
- **Fit against pixels across whole rallies**, sharing parameters, instead of flight by flight.

## Hand as the contact anchor (added 2026-11-13)

`fit3d/fit.py` now accepts `HANDS=<bodies/ground.py output>`. At each hit the ball must lie within `PADDLE_REACH` of the hitter's palm (default 1.1 ft, firmness `PADDLE_SIG` 0.15 ft), in place of "within 4.5 ft of the feet". Without `HANDS`, or outside the bodies' time span, the old rule applies.

Rally 267, events assigned by hand (10 hits). Net clearance and kitchen are not constrained by the fit, so they are real checks.

| Anchor | Reprojection median | Lowest net clearance | Bounces in right kitchen | Ball to palm, median / max |
|---|---|---|---|---|
| Feet, 4.5 ft | 1.61 px | 0.2 in | 10 of 10 | 1.02 / 2.33 ft |
| Hand, 1.1 ft, soft | 1.59 px | 2.0 in | 10 of 10 | 1.04 / 1.81 ft |
| Hand, 0.9 ft, firm (0.05) | 1.68 px | 2.0 in | 10 of 10 | 0.78 / 1.18 ft |

Point A of the same shot, automatic events (8 hits, some wrong): the soft hand anchor took net crossings below the tape from 4 of 8 to 0 of 8; the firm one left 2 of 8 and raised the reprojection error. So firm anchoring needs trustworthy events. Used for the page: firm for the hand-assigned stretch, soft for the two automatic fits.

```bash
HANDS=bodies/out/rally_267_grounded.npz PADDLE_REACH=0.9 PADDLE_SIG=0.05 uv run python fit3d/fit.py fit3d/out/rally_267_segments.json fit3d/events_hand_rally_267.json players/out/rally_267_tracks.npz calib/out/t270_calibration.json fit3d/out/rally_267_hand
```

Paddle side (`bodies/paddle.py`): contact time and place now come from the fit's solved event (the sample index could be 60 ms and a foot or two off), and where the ball is within the paddle's length of the palm and under 0.5 ft from the face plane the handle swings in that plane to point at it. Result over 23 usable hits: ball within 0.25 ft of the face at 7 (6 of the 10 hand-assigned ones); median gap 0.55 ft; median distance off the face plane 0.44 ft (was about 1.0).

What is left: at most hits the ball sits 0.5 to 1.5 ft from the palm along the line of sight, in front of the player (towards the net). Either the arm reaches further forward than the body model shows (reach along the camera's line of sight is what one camera sees worst), or the ball's depth is still off, or both. Not resolved.
