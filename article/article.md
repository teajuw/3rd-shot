# 3rd-shot

A pickleball match, rebuilt in 3D from one broadcast camera.

The pipeline turns ordinary match video into a digital twin: the court, four moving bodies, their paddles and the ball, in real feet and seconds, on a single Mac. It is a step toward a simulator that learned players can play in, and every stage is written up with what was measured, including what failed.

![A ten-second rally rebuilt in 3D, seen from eye height behind the near baseline: four coloured bodies with paddles on a blue court and a ball arcing over the net between them.](./demo-court.mp4)

Ten seconds of one rally, from a spot the broadcast camera never stood. Nothing from the broadcast is drawn here: every pixel comes from the reconstruction.

## Why

A broadcast shows a match from one fixed angle. Everything a coach or a simulator wants is missing from it: how high the ball was, how far a player reached, where the paddle faced. Tennis research has rebuilt matches from video and trained simulated players on the result. Nobody has published the same for pickleball, where the ball is slower and lighter, the court is small, and most of the game happens within a few feet of the net.

The bet is that physics can supply what one camera cannot see. A ball in flight has to obey gravity and drag. A body has to stand on the floor. A ball that changes direction was hit by a paddle facing a particular way.

## What comes out

For a rally, the pipeline produces a scene you can orbit in a browser:

- **The court**, from a camera calibration off the painted lines.
- **Four bodies** as full meshes, standing on the court at the right size.
- **The ball** as a 3D path that obeys physics between each hit and bounce.
- **Paddles** in each player's hand, turned to meet the ball at each hit.

For the whole match it also produces a table of every detected shot: who hit it, where all four players stood, and where the ball went next.

## How it works

![The same ten seconds of the rebuilt rally, seen from high behind one baseline, where the broadcast camera stands.](./demo-broadcast.mp4)

The same rally from where the real camera stands, which is the view the pipeline starts from.

**Court first.** The court lines give the camera's position and focal length, so every pixel becomes a line of sight in court feet. Everything else is measured against that.

**Players and ball in 2D.** A keypoint detector tracks the four players and places their feet on the court. A separate tracker follows the ball. A decoder then combines the ball's changes of direction, how close each player was, and the sound of the hit to find hits and bounces.

**The ball in 3D.** Each flight between two events is fitted as a physical trajectory that has to match the pixels, start at a paddle and end on the floor. Anchoring each hit to the hitter's hand raised the lowest net clearance on a hand-checked rally from 0.2 inches to 2.0, with the pixel error almost unchanged (1.61 to 1.68 pixels).

**Bodies.** Meta's SAM 3D Body gives a mesh per player per frame. On its own that output shakes and floats, so the pipeline fixes one body shape per player, smooths the joint angles, and slides each body along its line of sight until its lowest point touches the floor. Shake at the wrists and ankles drops from 31 to 46 m/s² to 11 to 17, and the gap to the independently tracked court position drops from 1.1 to 4.3 feet to 0.2 to 0.4.

**Paddles.** The paddle is a few blurred pixels, so it is inferred instead of measured: gripped in the palm, following the forearm, and rolled at each hit to the face angle the ball's change of direction requires. The first version flipped and clipped through bodies; the second turns at most 770 to 1,270 degrees per second, never flips, and sits inside a body in under 1% of frames.

## Where shots go

With the whole match processed, the shot table holds 659 shots with a known target. A model predicts where each shot will go as a probability map over the court, and is scored only on rallies it never saw.

![Two court diagrams for one shot. On the left, a prediction that ignores the situation spreads over the far court. On the right, a prediction that knows where all four players stand concentrates in front of the two opponents, around the spot where the ball actually bounced.](./shot-maps.png)

The figure is one typical shot, chosen as the median improvement and not a best case. Splitting the far court into 16 zones:

| Prediction uses | Right zone ranked first | Right zone in top three |
|---|---|---|
| Nothing (every zone equal) | 9% | 32% |
| Where shots usually go | 21% | 41 to 42% |
| Where the hitter stands | 20 to 21% | 50 to 51% |
| Where all four stand | 26 to 28% | 60 to 61% |

Where the opponents stand adds about three times as much as where the hitter stands. The model narrows the court; it does not call the shot.

## What did not work

- **The event detector does not generalise yet.** It scores 0.95 on the three clips it was tuned on and 0.50 on the one held-out clip. Everything downstream of it, including the shot table above, inherits that.
- **The ball simulator misses its target.** Given the first 60% of a flight, every version predicts the landing 61 to 186 pixels off, about a tenth of a second late or early. Refitting drag and adding spin changed the numbers by less than the noise in 13 test flights. One camera does not fix how far away the ball is from the start of an arc.
- **Court geometry did not help prediction.** Describing each target by its direction and how much of the available court it used scored worse than plain court position.
- **Finding paddles by shape failed.** A segmentation model outlines a paddle when shown where it is, but picking the paddle out automatically by size and shape did not work.

## Limits

One match, one venue, one camera angle, four players. Ten minutes of footage takes about six hours to process. The shot table's labels are automatic and have not been checked by hand, so its numbers measure agreement with the detector, not with the truth.
