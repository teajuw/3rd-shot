# Speed: where the time could come from

Written 2026-11-13. Ideas only; none of this is tested on our footage yet.

## Where it stands

For 10 minutes of baseline-camera rally footage on this Mac (Apple M5 Pro), extrapolated from short runs:

| Stage | Measured rate | 10 minutes of footage |
|---|---|---|
| Existing pipeline (player positions, ball tracking, hits and bounces, 3D ball path) | 16× slower than real time | about 2.7 hours |
| SAM 3D Body, four players, every frame (50 a second) | 37 min per video minute | about 6.1 hours |
| SAM 3D Body, four players, every second frame | 18 min per video minute | about 3.1 hours |

Total: about 8.8 hours at every frame, about 5.8 hours at every second frame. Sources: `batch/RESULTS.md` (167 s of footage in 44 minutes) and `bodies/RESULTS.md` (0.184 s per player per frame). Smoothing, the paddle and the web export are not built and not counted.

Guess at what is reachable: 2 to 3 hours for 10 minutes of footage. Not real time. Plan for an overnight batch job.

## Ideas, most promising first

1. **Skip dead time.** Bodies only matter between serve and point over. Once live play is detected reliably, everything else can be skipped; how much that saves depends on the footage.
2. **Fewer frames for bodies.** Body motion is much slower than ball motion, so every second frame with interpolation halves the cost (already in the 5.8-hour figure), and far players could probably go to every fourth.
3. **Bigger batches and half-precision.** The benchmark ran four players per call with default settings. One optimised version of SAM 3D Body (PoisedLabs fork) reports about 1.5× from batching 8 to 16 crops and 1.2 to 1.4× from half-precision on Apple GPUs.
4. **Overlap the stages.** Ball tracking is mostly CPU and bodies are GPU, so they can run at the same time instead of back to back.
5. **Tune the existing pipeline.** It has never been optimised. One sixth of the frames in this file are duplicates that could be dropped up front, and the ball tracker runs its network twice per frame.
6. **A smaller body model.** A distilled version of SAM 3D Body exists (InstantHMR) with about 1/30 the parameters. It is reported about 10 mm less accurate, most variants are non-commercial, and nobody has published its speed on a Mac, so this is the biggest possible win and the least certain.

## Ruled out for now

- **A cloud GPU for the body step.** The published speed for stock SAM 3D Body on a high-end NVIDIA card is about 2 player-frames a second; this Mac did 5.4. Only the heavily optimised NVIDIA-only version beats it, and not by much.
- **Optimising before the pipeline is complete.** Smoothing, paddle and export will change the cost profile. Build those first, then optimise once it runs end to end.
