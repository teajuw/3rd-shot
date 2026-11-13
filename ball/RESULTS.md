# Ball detection test (2D, no pickleball training)

Run 2026-11-11 on one baseline-camera shot of the PPA Brisbane 2026 men's doubles final: frames 16,036–18,025 (1,990 frames at 60 fps).

## Detectors

- **Colour and motion** (`candidates.py`): subtract the shot's median background, keep small blobs where yellow-green has appeared (red and green up, blue down) and that are greener than red, which rejects skin. NumPy/OpenCV only. 18 ms per frame.
- **TrackNetV3, RacketVision weights** (MIT; trained on tennis, badminton, table tennis): background plus 4 frames at 512 × 288 gives a heatmap. Run with frame stride 1 and 2. 33 ms per frame per stride on MPS.
- **Fused** (`track_ball.py`): colour candidates get a bonus when TrackNet agrees within 15 px; TrackNet-only candidates are kept at lower weight. A dynamic programme picks the single best path under a speed limit of 90 px per frame, allowing gaps up to 12 frames and paying to restart.

## Results on this rally

| Mode | Frames with a detection | With gaps ≤ 8 frames filled | Longest remaining gap |
|---|---|---|---|
| Colour and motion only | 67.8% | 81.2% | 52 frames |
| TrackNetV3 only | 57.4% | 65.5% | 103 frames |
| Fused | 79.0% | 88.6% | 52 frames |

In the fused track, 56.0% of frames are confirmed by both detectors, 21.5% by colour only and 1.5% by TrackNet only. When TrackNet's top candidate fires, a colour candidate sits within 15 px of it 97% of the time.

## Spot check by eye

30 frames drawn at random (seed 5) after the last code change, so this sample was not used for tuning:

| | Count |
|---|---|
| Detection is the ball | 21 |
| Detection is not the ball (net post, TrackNet-only) | 1 |
| No detection, and I could not see a ball in the frame | 8 |
| No detection, but a ball was visible | 0 |

The 8 silent frames were dead time between points (ball in hand) or the ball hidden by a player's body.

An earlier 30-frame sample (seed 3) on the first version found 23 correct, 3 false detections on bare legs, 1 miss at paddle contact and 3 correct silences. The leg errors led to the "greener than red" rule; all three of those frames are now silent. That sample was used for tuning, so it does not count as a test.

## What this does and does not show

- Zero-shot ball finding works on this venue: a blue court with a neon ball is close to the easiest case for a colour rule.
- The tennis-trained tracker transfers to pickleball about half the time with no fine-tuning, and almost never disagrees with the colour rule when it fires.
- Each detector covers frames the other misses, so the fusion beats both.

Not established:

- **No labelled ground truth.** The check is 30 frames judged by eye, at reduced scale for the silent ones. True recall and precision are not measured.
- **One rally, one venue.** A different court colour, ball colour or lighting could break the colour rule. The thresholds were set here.
- **Remaining gaps** (12, up to 52 frames) fall where the ball is hidden or held; whether any contain a visible ball was only checked through the random sample.
- **2D only.** Positions are image pixels. No bounce or hit detection, no height, no 3D.
- **Blur.** Fast balls are streaks; the reported position is the blob centre, which is not necessarily the ball's position at a fixed instant.
- The checkpoint is loaded through an allowlisted unpickler because it references `mmengine` classes; only tensors are read from it.

## Files

- `out/rally_267_ball.mp4`: broadcast with the fused track drawn.
- `out/rally_267_fused_track.npz`: per-frame x, y, weight and source, plus the gap-filled track.
- `out/rally_267_cands.npz`: raw candidates from each detector.
- `out/rally_267_check.jpg`, `out/none_a.jpg`, `out/none_b.jpg`: the spot-check images.

## Reproduce

```bash
uv run python ball/candidates.py data/video/IpUxCoBHg_g.mp4 16036 1990 ball/out/rally_267_cands.npz
```

```bash
uv run python ball/track_ball.py data/video/IpUxCoBHg_g.mp4 ball/out/rally_267_cands.npz ball/out/rally_267 fused 5
```
