# Ball detection, tracking, and monocular 3D trajectory reconstruction for pickleball from single-camera broadcast video

Notes compiled 2026-11-11. Items marked **[NEW]** were published after mid-2025; exact dates are given where I could verify them. "Apple Silicon" assessments are my inferences unless a source is cited; no source I read reports MPS or MLX benchmarks for any of these models.

## 1. What are the current best 2D ball trackers (TrackNet V2/V3/V4, WASB, TTNet, BlurBall, 2025–2026 successors)? Which have public weights, and has anyone trained or evaluated them on pickleball?

### Takeaway
The practical state of the art is still the heatmap-over-stacked-frames family: TrackNetV3 (with background-median input and InpaintNet) and WASB (HRNet) are the strongest baselines with MIT-licensed code and weights, and 2025–2026 work (BlurBall, TOTNet, RacketVision) refines them for blur, occlusion, and multi-sport training rather than replacing them. Nothing peer-reviewed covers pickleball; only hobby/student TrackNetV2/V3 fine-tunes and Roboflow datasets exist, so a pickleball fine-tune on self-labelled PPA/MLP frames is required.

### Cited Findings

**TrackNetV3 (Chen & Wang, ACM MM Asia 2023; badminton)**
- Official repo releases code and pretrained checkpoints (`TrackNet_best.pt`, `InpaintNet_best.pt`, Google Drive zip) under the MIT License, explicitly allowing commercial use. PyTorch; dev environment listed as torch 1.10, Ubuntu. — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- Reported on the Shuttlecock Trajectory Dataset test split: accuracy 97.51%, precision 97.79%, recall 99.33%, F1 98.56%, 25.11 FPS; TrackNetV2 F1 97.03%; YOLOv7 F1 68.00%. — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- Uses a per-match or per-rally median background image as an extra input, and a second module (InpaintNet) that repairs the predicted trajectory where the shuttle is occluded; InpaintNet is trained with sequence length 16 and mask ratio 0.3. — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- A fine-tuned TrackNetV3 for beach volleyball is published on Hugging Face (about 130 MB checkpoint, MIT carried forward), showing the transfer-learning path is routine. — [deadfast/beach-volley-vision-models](https://huggingface.co/deadfast/beach-volley-vision-models)

**WASB (Tarashima et al., BMVC 2023; "Widely Applicable Strong Baseline")**
- Repo is MIT-licensed; tested with Python 3.8, CUDA 11.3, Ubuntu 18.04; Docker recommended. — [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
- Model zoo gives pretrained weights for 7 models (DeepBall, DeepBall-Large, BallSeg, TrackNetV2, ResTrackNetV2, MonoTrack, WASB) across 5 sports (soccer, tennis, badminton, volleyball, basketball), i.e. 35 downloadable checkpoints. No pickleball or table tennis. — [WASB MODEL_ZOO](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md)
- WASB switched from a U-Net to an HRNet backbone; about 1.5 M parameters. — [BlurBall paper](https://arxiv.org/html/2509.18387v2)

**TrackNetV4 (Raj, Wang, Gedeon; arXiv 2409.14543, Sept 2024)**
- Adds frame-differencing "motion attention maps" modulated by a motion prompt layer on top of TrackNet; described as lightweight and plug-and-play. — [arXiv 2409.14543](https://arxiv.org/abs/2409.14543v1)
- I could not verify an official code or weights release. One third-party summary points to `jean50621/Badminton_Challenge`, which another search result identifies as a different badminton project. Treat TrackNetV4 as paper-only until checked. — [deeppaper summary](https://arxiv.deeppaper.ai/papers/2409.14543v1?locale=en)

**BlurBall [NEW] (Gossard, Radovic, Ziegler, Zell, Univ. Tübingen; arXiv 2509.18387, Sept 2025; v2 dated 15 Feb 2026)**
- HRNet/WASB-derived detector that labels the ball at the middle of its motion-blur streak, adds Squeeze-and-Excitation attention over multi-frame input, and outputs blur angle and half-length (via PCA of the predicted heatmap) alongside position. — [arXiv 2509.18387v2](https://arxiv.org/html/2509.18387v2)
- New table tennis dataset: 26 online recordings, 64,119 frames (51,423 train / 12,696 test), each labelled with ball centre, blur angle, blur half-length. Hosted via the project page. — [arXiv 2509.18387v2](https://arxiv.org/html/2509.18387v2); [project page](https://cogsys-tuebingen.github.io/blurball/)
- Benchmark on that dataset (F1, "front" / "mid-streak" labelling): TrackNetV2 90.37/91.61 (11.3 M params, 85 FPS); MonoTrack 93.60/94.97 (2.9 M, 115 FPS); TrackNetV3 93.74/95.93 (17.8 M, 77 FPS); WASB 3-step 94.23/95.77 (1.5 M, 95 FPS); WASB 1-step 95.58/96.00 (42 FPS); BlurBall with SE 96.52 (1-step) and 96.16 (3-step). Labelling the ball at the streak midpoint improved every model. — [arXiv 2509.18387v2](https://arxiv.org/html/2509.18387v2)
- Evaluated on table tennis only; the authors note that pretraining on badminton did not help. TrackNetV4 and TTNet are cited but not benchmarked. — [arXiv 2509.18387v2](https://arxiv.org/html/2509.18387v2)

**TOTNet [NEW] (Xu et al., Deakin Univ. / Paralympics Australia; arXiv 2508.09650, submitted 13 Aug 2025)**
- 3D-convolutional U-Net that keeps the temporal dimension throughout; visibility-weighted BCE loss; occlusion augmentation that masks the ball region; optional RAFT optical-flow input; 5 input frames at 288×512. — [arXiv 2508.09650](https://arxiv.org/html/2508.09650v1)
- On fully occluded frames it reports RMSE/accuracy of 7.19 px / 0.80 on its TTA table tennis set versus WASB 37.30 / 0.63; on tennis fully occluded 15.31 / 0.33 versus WASB 264.45 / 0.17. On badminton "not visible" frames WASB was better (56.50 / 0.79 vs 70.22 / 0.75). The paper's text and table disagree on one tennis figure (0.67 vs 0.33). — [arXiv 2508.09650](https://arxiv.org/html/2508.09650v1)
- Code at `github.com/AugustRushG/TOTNet`; TTA dataset (9,159 samples, 1,996 occluded) on academic request; pretrained weights and code license not stated in the paper. — [arXiv 2508.09650](https://arxiv.org/html/2508.09650v1)

**RacketVision [NEW] (Dong et al., Zhejiang Univ. / Shanghai AI Lab; arXiv 2511.17045, Nov 2025, v3 28 Jan 2026; AAAI 2026 Oral)**
- 942 games, 1,672 clips, 435,179 frames across table tennis, tennis, and badminton; tasks are ball tracking, racket pose, and trajectory prediction. No pickleball. — [arXiv 2511.17045v3](https://arxiv.org/html/2511.17045v3)
- Benchmark (1920×1080; mean distance error in px): single-frame detectors are poor localisers (YOLO11 MDE 30.6 / 17.5 / 29.3 px on TT / tennis / badminton; recall 0.42–0.60). Multi-sport TrackNetV3 with background median and 4 frames is best: precision 0.924 / 0.945 / 0.915, recall 0.762 / 0.880 / 0.865, MDE 3.41 / 1.96 / 1.54 px. Background-median input roughly halves MDE for TrackNetV3 (tennis 7.95 → 3.07 px). — [arXiv 2511.17045v3](https://arxiv.org/html/2511.17045v3)
- Code, dataset (Hugging Face `linfeng302/RacketVision`) and pretrained weights (`linfeng302/RacketVision-Models`) released April 2026 under MIT. — [OrcustD/RacketVision](https://github.com/OrcustD/RacketVision)

**Other 2025–2026 detectors**
- Uplifting Table Tennis [NEW] (25 Nov 2025) uses a Segformer++ B2 ball detector (24.7 M params, 1600×896, 3 stacked frames, 18 FPS; ACC@5px 87.1%, ACC@10px 91.8%) and keeps a detection only if a second model (WASB) agrees within 20 px. — [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1)
- YOLO-Ball [NEW] (March 2026, J. Sports Engineering and Technology) is a single-frame tennis detector targeting occlusion and blur. I did not read the paper. — [metatoc listing](https://metatoc.com/papers/168990-yolo-ball-real-time-tennis-ball-detection-under-occlusion-and-motion-blur)
- A one-shot badminton shuttle detection paper [NEW] (arXiv 2603.06691, March 2026) reviews multi-frame small-ball trackers; not read in detail. — [arXiv 2603.06691](https://arxiv.org/html/2603.06691v1)
- TTNet (Voeikov et al., CVPRW 2020) treats event spotting (bounce, net) as multilabel classification jointly with ball detection. — [arXiv 2004.09927](https://arxiv.org/pdf/2004.09927)

**Pickleball-specific tracking**
- Hudson Griffith (Univ. of Florida, Oct 2024) fine-tuned TrackNetV2 on hand-labelled frames (ball x, y and bounce status) from one PPA Tour match on YouTube. Reported accuracy 94.07%, recall 93.47%, precision 87.81%. Noted overfitting to the single match, trouble with occlusions and with non-straight-on camera angles. No weights or dataset link. — [hudsondev.pages.dev/pickleball](https://hudsondev.pages.dev/pickleball)
- `AndrewDettor/TrackNet-Pickleball`: PyTorch TrackNetV2 (3-in-3-out) workflow from frame extraction and labelling to training and evaluation, with pickleball weights on Google Drive. No license file noted, no accuracy numbers on the README. — [AndrewDettor/TrackNet-Pickleball](https://github.com/AndrewDettor/TrackNet-Pickleball)
- A University of Rochester senior design project also uses transfer learning on TrackNetV2 for a pickleball analytics platform. — [Rochester senior design](https://www.hajim.rochester.edu/senior-design-day/pickleball-analytics/)
- Roboflow Universe hosts several pickleball ball/court detection datasets (for example a 3-class set under CC BY 4.0, a "Ball Detector" trained on 2,025 images with classes for several ball colours, and a "Court-and-Ball" set with about 6,200 images). These are single-frame bounding-box sets with little methodology. — [pickle-ball-detection](https://universe.roboflow.com/pickle-ball/pickle-ball-detection); [pickleball-ref ball detector](https://universe.roboflow.com/pickleball-ref/ball-detector-qbcf1); [pickleballdetection](https://universe.roboflow.com/pickleballdetection-usckc/pickleballdetection)

### Inferences
- Recommended starting point: TrackNetV3 from the RacketVision release (multi-sport weights, 4-frame, background median, MIT) or WASB tennis weights, fine-tuned on a few thousand labelled pickleball frames. Rationale: both have permissive licenses and weights; RacketVision shows multi-sport pretraining plus background median gives the lowest pixel error; the fixed behind-baseline broadcast camera makes a background median cheap and effective.
- BlurBall's two ideas transfer directly even without its weights: label the ball at the middle of the blur streak, and use streak direction/length as an extra velocity observation for the 3D fit. At broadcast frame rates a driven pickleball is a streak, so this matters.
- Apple Silicon: all of these are small fully convolutional PyTorch models (1.5 M to 25 M parameters) with no custom CUDA ops that I saw, so MPS inference should work, and offline CPU inference is also feasible given the hours-per-match budget. TOTNet's 3D convolutions and RAFT optical flow are the most likely to hit MPS operator gaps. Old pinned environments (torch 1.10, CUDA 11.3 Docker) will need dependency updates. None of this is verified by a source.
- Single-frame YOLO-style detectors should be used only as a second-opinion filter (as Kienzle et al. do with two models), not as the primary tracker: RacketVision shows their localisation error is 5–10 times worse.
- The Hudson Griffith result suggests a model trained on one match overfits; label frames from several venues, court colours, and ball colours.

### Gaps
- No peer-reviewed paper or public benchmark evaluates any tracker on pickleball. The perforated ball's appearance (holes, lower contrast streak) is untested in the literature.
- I did not confirm an official TrackNetV4 repository, weights, or license.
- I did not confirm whether BlurBall or TOTNet pretrained weights are downloadable, or their code licenses; the papers only give project/repo links.
- No source reports MPS or MLX performance for any of these models.
- I found no "TrackNetV5" or equivalent; I cannot confirm one exists.

## 2. What methods reconstruct a 3D ball trajectory from a single calibrated camera using physics, and what learned 2D-to-3D lifting networks exist?

### Takeaway
There are two families: per-arc physics optimisation (MonoTrack, TT3D, Volley3D), which fits initial position/velocity (and sometimes spin) of an ODE with gravity, quadratic drag and optionally Magnus force by minimising reprojection error with anchoring constraints; and learned lifting networks trained only on synthetic physics trajectories (SynthNet, Where Is The Ball, Kienzle et al., TT4D), which lift a whole 2D track and tolerate missing detections. No method targets pickleball, and the most complete public code is for table tennis.

### Cited Findings

**MonoTrack (Liu & Wang, CVPRW CVSports 2022; badminton singles, fixed broadcast view)**
- Per-shot model: particle under gravity plus quadratic drag, d²x/dt² = g − Cd‖v‖v, with Cd free per shot. Unknowns are initial position, initial velocity, and Cd, solved by nonlinear regression. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- Loss: reprojection error plus penalties tying the trajectory start to the hitter's position (feet position, about 2 m height) and the end to the receiver's position, plus an out-of-court penalty. Constraints: heights 0–3 m at start and end, start on hitter's side, initial velocity toward the opponent. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- Calibration by DLT from six known points (four court corners, two net-pole tips). — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- On 10k synthetic trajectories the priors cut mean 3D error from 14.9 cm to 8.0 cm. On real video reprojection error is about 8–10 px with ground-truth tracks and hits, 23–37 px end-to-end. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- The source page I read does not state a code release for the full MonoTrack system; WASB re-implements only its 2D shuttle detector. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899); [WASB MODEL_ZOO](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md)

**TT3D (Gossard, Ziegler, Zell; arXiv 2504.10035, 14 Apr 2025; CVPRW CVSports 2025; table tennis)**
- Pipeline: automatic camera calibration from a table segmentation network; BlurBall 2D detections; piecewise second-degree polynomial segmentation by dynamic programming; bounce point by intersecting the viewing ray with the table plane offset by the ball radius; then optimisation of pre-bounce velocity and spin. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Physics: flight ODE with quadratic drag and Magnus force; bounce with a Coulomb friction model with rolling and sliding regimes; parameters COR 0.85, friction μ 0.3, radius 0.02 m, mass 2.7 g. Optimiser: IPOPT via CasADi, minimising reprojection error; spin initialised at zero. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Accuracy on 130 simulated trajectories projected at 25 fps: side view 97.3% success / 8.9 cm MAE; oblique 91.5% / 14.5 cm; back view 92.3% / 22.3 cm (86.9% / 29.8 cm with noise). The back view, which is the pickleball broadcast geometry, is the worst case. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Limitations: needs at least three visible ball frames after a bounce; segmentation breaks with a moving camera; player occlusion corrupts calibration in back views; spin accuracy not validated against ground truth. An appendix sketches a tennis extension with changed constants, not evaluated. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Code: `github.com/cogsys-tuebingen/tt3d`; code license not stated on the project page. Also includes 3D player pose (RTMPose + MotionBERT) with floor-contact translation solving. — [TT3D project page](https://cogsys-tuebingen.github.io/tt3d/)

**Where Is The Ball (Ponglertnapakorn & Suwajanakorn, VISTEC; arXiv 2506.05763, 6 Jun 2025; CVPRW CVSports 2025; sport-agnostic)**
- Input: 2D track plus known intrinsics and extrinsics. Each 2D point is converted to a camera-independent representation: the intersections of its viewing ray with the ground plane (y=0) and a vertical plane (z=0). — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Three LSTM stages: an end-of-trajectory network (soft cue, not a hard segmenter), a height network (forward and backward LSTMs accumulate height differences; predicting only height keeps every 3D point on its pixel ray), and a refinement network for noisy input. — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Trained entirely on Unity/PhysX simulated trajectories with impulse forces; hits and bounces are handled implicitly, with no explicit event detection. — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Results: real ping-pong mocap about 3.4 cm RMSE; real soccer (IPL) about 1.2 m RMSE; broadcast tennis (TrackNet dataset, no 3D ground truth) landing error 0.63 m versus 3.58 m for SynthNet, landing F1 0.807 versus 0.358. — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Limitations: first and last frames must be on the ground; simulator ignores spin and aerodynamics; fails on volleys near the net, which differ from training trajectories. — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Code, Unity simulation, and dataset are listed as "coming soon" on the project page (as fetched 2026-11-11); not released. — [where-is-the-ball.github.io](https://where-is-the-ball.github.io/)

**SynthNet (ITU Copenhagen, published Oct 2024; tennis)**
- Detects hits and bounces to segment shots first, then reconstructs each shot's 3D trajectory with a network trained purely on synthetic physics-based data. — [ITU Pure record](https://pure.itu.dk/en/publications/synthnet-leveraging-synthetic-data-for-3d-trajectory-estimation-f/)
- It was substantially outperformed on tennis landing metrics by Where Is The Ball (see above). — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)

**Kienzle et al. (Univ. Augsburg): table tennis uplifting**
- CVPRW 2025 paper (arXiv 2504.19863): a network trained only on synthetic data infers the full 3D trajectory and initial spin from a 2D broadcast track; 92.0% spin classification accuracy; predicts the whole trajectory rather than only initial conditions to avoid accumulated integration error. — [arXiv 2504.19863](https://arxiv.org/abs/2504.19863)
- **[NEW]** "Uplifting Table Tennis" (arXiv 2511.20250, 25 Nov 2025; WACV 2026): two-stage pipeline. Back end is a transformer (embedding module plus 16 blocks, d=128, 4 heads; the last 4 blocks estimate spin through a learnable spin token), with one token per ball detection. Missing detections are handled by dropping tokens; variable frame rate by a time-proportional rotary position embedding (2 ms resolution), trained over 20–60 FPS with 5% detection dropout. — [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1)
- Synthetic data: 140,000 MuJoCo trajectories (about 4.7 M 2D ball positions). — [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1)
- Results: full pipeline ball reprojection error 12.28 px on TTHQ and 9.41 px on TTST; binary topspin/backspin accuracy 89.5% and 97.1%. No direct 3D ground truth; accuracy is assessed by reprojection. — [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1)
- Code at `github.com/KieDani/UpliftingTableTennis`, with the TTHQ dataset linked from its README; license and weights availability not stated on the project page. — [project page](https://kiedani.github.io/WACV2026/)

**TT4D [NEW] (Rahmanian, Kienzle, Gossard, Kalaria, Lienhart, Sastry; arXiv 2605.01234, v1 2 May 2026, v2 9 Sep 2026; table tennis)**
- Dataset of 140+ hours of singles and doubles play reconstructed from monocular broadcast video, with camera calibration, 3D ball positions, ball spin, time segmentation, and 3D human meshes. — [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- Key design reversal: instead of segmenting the 2D track into shots and then reconstructing, it lifts the whole unsegmented 2D ball track to 3D with a learned network (which also infers spin and copes with unreliable detections and heavy occlusion), and then segments in 3D. The authors argue 2D segmentation fails under occlusion and varied viewpoints. — [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- Spin is inferred from the trajectory, not observed, because it is not directly visible in broadcast video. — [arXiv 2605.01234](https://arxiv.org/html/2605.01234)

**Volley3D [NEW] (Boussouf, Maglo, Chaouch, Saito; arXiv 2610.38358, 28 Oct 2026; volleyball broadcast)**
- TrackNetV3 for 2D; a small temporal convolutional network over (x, y, dx, dy) for deflection detection; per-segment physics lifting with dv/dt = g − Cd‖v‖v, Cd = 0.035 m⁻¹, Magnus neglected. Initial velocity per segment is found by Nelder-Mead minimising endpoint error, initialised from the drag-free ballistic solution; segment endpoints are fixed to reconstructed 3D hand positions of the hitting players. — [arXiv 2610.38358](https://arxiv.org/html/2610.38358)
- 3D ball RMSE on the four-camera VolliQ dataset: 0.27 m with drag and hand boundary conditions, versus 1.56 m for a MonoTrack-style fit. Its 2D reprojection error is higher (about 17 px versus 3.9 px), i.e. fitting pixels tightly did not give the best 3D. — [arXiv 2610.38358](https://arxiv.org/html/2610.38358)
- Project page `noambsf.github.io/Volley3D`; annotations promised; code license not stated. — [arXiv 2610.38358](https://arxiv.org/html/2610.38358)

**Other reference points**
- "Beyond the Broadcast" (arXiv 2507.20006, July 2025; VR tennis) reconstructs matches from 25 fps broadcast video and states direct 3D ball trajectory estimation is not feasible from monocular video, using manually annotated keyframes with a kinematic model instead. — [arXiv 2507.20006](https://arxiv.org/pdf/2507.20006)
- A soccer paper (Yandex; arXiv 2506.07981, June 2025) on real-time single-camera ball localisation frames the problem as depth ambiguity along the viewing ray. — [arXiv 2506.07981](https://arxiv.org/html/2506.07981v1)
- US patent 11,615,540 describes single-camera 3D ball tracking by optimising a 3D physics model (bounce, spin, swing, air resistance) against candidate 2D detections with a fixed projection matrix. — [USPTO 11615540](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11615540)

### Inferences
- For a smooth, replayable twin, the best-supported design is a hybrid: segment into arcs, then fit each arc with a gravity + quadratic drag (+ optional Magnus) ODE by minimising reprojection error, with hard anchors at arc ends. Anchors available in pickleball are: bounce points on the court plane (ray-plane intersection at ball-radius height, as TT3D does) and hit points near the hitter's paddle/wrist in 3D (as MonoTrack and Volley3D do). Volley3D's result (0.27 m vs 1.56 m) is the clearest evidence that anchoring to 3D contact points matters more than pixel fit.
- Because every arc is the output of an ODE, the result is smooth and physically plausible by construction, which matches the stated priority over per-frame accuracy. Consecutive arcs should share their endpoint (the end of one arc is the start of the next) to avoid visible jumps.
- A learned lifter (Kienzle/TT4D style) is attractive as an initialiser and for occlusion robustness, but no released weights apply to pickleball; one would have to generate synthetic pickleball trajectories with the parameters in section 4 and train the transformer. The model is small (16 blocks, d=128), so training on an M-series Mac with MPS is plausible; this is unverified.
- The physics-optimisation route needs only SciPy or CasADi/IPOPT on CPU and is trivially Apple Silicon compatible.
- TT3D's back-view error (22–30 cm on a 2.74 m table) suggests depth error along the camera axis will be the dominant error for the behind-baseline pickleball view; expect errors of several tens of centimetres in the along-court direction without anchors.
- Pickleball has many volleys at the kitchen line (no bounce between hits). Methods that rely on a bounce per arc (TT3D) or ground-start/ground-end (Where Is The Ball) do not fit directly; the MonoTrack/Volley3D hit-to-hit formulation is the closer match.

### Gaps
- No monocular 3D reconstruction paper, dataset, or repo for pickleball was found.
- No 2026 tennis-specific monocular 3D ball paper with spin and bounce modelling was found; recent progress is concentrated in table tennis and volleyball.
- TT4D's quantitative accuracy, dataset/code URL, and license were not on the abstract page; I did not read the full paper.
- The drag/Magnus/bounce models inside Kienzle et al.'s MuJoCo simulator were not extracted.
- Whether MonoTrack's full pipeline was ever open-sourced was not confirmed.
- I could not verify SynthNet's code availability or exact venue from the record I saw.

## 3. How do these methods segment a rally into ballistic arcs, and how are hit/bounce events detected?

### Takeaway
Segmentation is done either from the 2D track (piecewise polynomial fits, small temporal networks on positions and velocities), from track plus player pose (MonoTrack's HitNet, Volley3D's wrist-proximity scoring), or after lifting in 3D (TT4D). Audio is a strong but under-used cue: a Tübingen pipeline localises table tennis bounces to sub-millisecond onsets and classifies the surface with F1 0.97, and older tennis work shows audio-visual fusion beats either modality.

### Cited Findings
- MonoTrack HitNet: a GRU network of about 16K parameters over 12-frame windows of court corners, both players' poses, and shuttle position; classes are no hit / near-player hit / far-player hit. A dynamic-programming post-process enforces about one hit per second, at least 0.5 s between hits, and alternating players. Result: 89.7% accuracy, precision 94.9%, recall 94.3%, F1 0.946, more than 35% better than a trajectory-derivative baseline. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- MonoTrack's authors estimate about 20% of reconstructed shots may be wrong at about 90% segmentation accuracy, because one misclassified hit corrupts the neighbouring arcs. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- TT3D: piecewise second-degree polynomials fitted by dynamic programming, with an extra cost term for blur-direction error; table bounces are separated from racket strikes by whether the direction along the table's long axis reverses. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Volley3D [NEW]: temporal convolutional network on (x, y, dx, dy) with a soft-focal loss; segmentation F1 at ±1 frame is 0.929 on oracle 2D tracks and 0.898 on TrackNetV3 output. A training-free score (ball-to-wrist distance with Gaussian decay, plus wrist velocity) picks the acting player with 85.3% / 83.3% accuracy; a ball closer to the ground than to any wrist is labelled a bounce. — [arXiv 2610.38358](https://arxiv.org/html/2610.38358)
- Where Is The Ball: an end-of-trajectory LSTM predicts per-step arc-end probability from differences of ray/plane intersection points, used as a soft auxiliary signal. — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- TT4D [NEW]: segments in 3D after lifting, arguing 2D segmentation fails under occlusion and viewpoint variation. — [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- TTNet treats bounce and net events as multilabel classification because the ball may be blurred or there may be no frame showing the actual bounce. — [arXiv 2004.09927](https://arxiv.org/pdf/2004.09927)
- Audio (Gossard, Schmalzl, Ziegler, Zell; arXiv 2409.11760, v2 19 Sep 2025): energy-peak onset detection on a 10 kHz high-pass filtered signal followed by a CNN. Onset error 0.09 ms clean and 0.2 ms with speech noise; precision 0.98; recall 1.0 clean and 0.95 noisy; surface classification (racket / table / floor / other) F1 0.97; spin presence F1 0.96. Dataset of 3,396 bounces at 44.1 kHz, controlled single-player recording. Code and data at `github.com/cogsys-tuebingen/tt_sounds`. — [arXiv 2409.11760](https://arxiv.org/html/2409.11760v2)
- Older tennis audio-visual work (Huang, Cox, Zhou, Xie): track the ball visually, find candidate hits along the trajectory, and fuse with audio; on a women's doubles match fusion improved F-score by about 30 points over audio alone and about 10 over video alone. — [Surrey open research record](https://openresearch.surrey.ac.uk/permalink/44SUR_INST/15d8lgh/alma99512922402346)
- A 2025 survey covers precise event spotting in sports and its frame-level temporal demands. — [arXiv 2505.03991](https://arxiv.org/html/2505.03991v2)
- The Hudson Griffith pickleball fine-tune labelled bounce status per frame alongside ball position, but reports no bounce-detection accuracy. — [hudsondev.pages.dev/pickleball](https://hudsondev.pages.dev/pickleball)

### Inferences
- A robust offline pickleball event detector should fuse three cues: (1) audio onsets (paddle "pop" and court bounce are loud, impulsive, and present in broadcast audio), (2) 2D track kinks (velocity discontinuities), and (3) wrist/paddle proximity from player pose. Then apply MonoTrack-style dynamic programming with pickleball rules as priors: hits alternate between teams, the serve and return must bounce (two-bounce rule), at most one bounce per side before a hit, minimum inter-hit time.
- Audio gives sub-frame timing, which matters at 30–60 fps: a hit often falls between frames, and knowing the exact contact time lets adjacent arcs be intersected at a non-integer frame time. Caveats: broadcast audio has commentary and crowd noise, may be offset from video by a variable delay, and the table tennis result was from a controlled recording, so accuracy on PPA broadcasts is unknown.
- Classifying paddle hit versus court bounce from audio alone is plausible given the 0.97 surface F1 in table tennis, but must be trained on pickleball sounds; no pickleball audio dataset was found.
- With offline processing, the fit itself can adjudicate events: try candidate segmentations and keep the one with the lowest total physics-fit residual (an analysis-by-synthesis check). This is my suggestion, not something a cited paper does in exactly this form, though TT3D's dynamic-programming polynomial fit is close in spirit.
- Doubles (four players) makes "alternating player" weaker than in MonoTrack's singles setting; "alternating side of the net" still holds.

### Gaps
- No published hit/bounce detector for pickleball (audio or video) with reported accuracy was found.
- No 2025–2026 tennis-specific audio-visual bounce/hit paper was found; the tennis fusion work I found is older.
- I did not verify how well audio onsets survive broadcast compression or audio-video sync drift on YouTube.

## 4. What are the aerodynamics of a pickleball (drag, speeds, spin, bounce restitution)?

### Takeaway
Published free-flight measurements put the outdoor (40-hole) pickleball drag coefficient at about 0.30–0.33 and the indoor (26-hole) ball at about 0.45, with lift coefficient roughly 0.195 × spin parameter or about 0.2 at low spin; the equipment standard implies a normal coefficient of restitution of 0.62–0.66 on a hard surface. Data are sparse, measured at only 5–13 m/s, and sources disagree, so drag should be a fitted or bounded parameter rather than a constant.

### Cited Findings
- Official ball specification: diameter 2.87–2.97 in (7.29–7.54 cm); weight 0.78–0.935 oz (22.1–26.5 g); bounce 30–34 in (76.2–86.4 cm) to the top of the ball when dropped from 78 in (198.1 cm) onto a granite plate. — [USA Pickleball Equipment Standards Manual (mirror)](https://www.pickleball.ph/uploads/1/2/5/9/125919777/-_official_equipment_standards_manual.pdf)
- Implied ball coefficient of restitution at the spec limits: 0.62 (30 in) to 0.66 (34 in), i.e. the square root of rebound height over drop height. Vendor-published, but consistent with the arithmetic. — [RPM Pickleball ball science](https://rpmpb.com/pages/ball-science)
- Crawford Lindsey (Tennis Warehouse University, 12 May 2025): 86 trajectories from a pendulum launcher filmed at 300 fps by two synchronised cameras, launch speeds 5–13 m/s (11–29 mph), about 5 m of flight, fitted with RK4. Mean Cd about 0.45 for the indoor 26-hole ball (SD 0.092) and about 0.33 for the outdoor 40-hole ball (SD 0.083). Cd "remains relatively constant or increases slightly with velocity". Low-spin lift coefficients: topspin CL about −0.24, backspin about +0.17; backspin |CL| clusters around 0.2 with little spin dependence. Air density used 1.21 kg/m³; ball radius 0.037 m. — [TWU pickleball aerodynamics](https://twu.tennis-warehouse.com/learning_center/pickleball/pickleball_aerodynamics.php)
- Steyn et al., "Executing a Successful Third Shot Drop in Pickleball" (arXiv 2501.00163, Jan 2025): Cd = 0.30 ± 0.02 and Cl = 0.195·S (S = 2πrn/u), fitted to six single-camera trajectories of a 40-hole Selkirk ball; no systematic dependence on spin or Reynolds number seen; large scatter in Cl attributed to spin-measurement uncertainty. Measured shot: 10.0 m/s with 5.9 rev/s; model uses 12 m/s and 10 rev/s (topspin up to 15 rev/s); successful third-shot drops need 10.9–13 m/s down the line and 13.3–16 m/s cross-court; serve range quoted from another source as 18–24 m/s. — [arXiv 2501.00163](https://arxiv.org/html/2501.00163v1)
- Internal inconsistency in Steyn et al.: a figure lists Cl = ±0.12 at spin parameter 0.19, whereas the fitted formula gives about 0.037. — [arXiv 2501.00163](https://arxiv.org/html/2501.00163v1)
- Earlier modelling (Emond, Sun & Swartz 2024) used a fixed Cd = 0.6 borrowed from wiffleball wind-tunnel data (Rossmann & Rau 2007) with no lift; this is well above both measured means. Creer (2023) did a CFD study of 26- and 40-hole balls. Lindsey notes experimental studies are scarce. — [TWU pickleball aerodynamics](https://twu.tennis-warehouse.com/learning_center/pickleball/pickleball_aerodynamics.php)
- Pickleball Science assumed a lift coefficient of 0.075 (half the baseball value) pending wind-tunnel verification. — [Pickleball Science topspin aerodynamics](https://pickleballscience.org/pickleball-topspin-aerodynamics/)
- Speeds and spin from paddle-testing and enthusiast sources (not pro-match measurements): radar-measured serves about 60–61 mph; one analysis argues about 70 mph is the maximum serve speed that stays in with 1,800 rpm topspin, with about 52 mph average; a measured serve of 2,273 rpm; vendor table gives flat drives under 300 rpm, topspin drives 500–900 rpm, pro spin serves 1,200–2,500 rpm. — [Matt's Pickleball](https://www.mattspickleball.com/best-pickleball-paddles); [Pickleball Science, how fast is a pickleball](https://pickleballscience.org/how-fast-is-a-pickleball); [The Dink](https://thedinkpickleball.com/rachel-rohrabacher-rips-a-2-273-rpm-serve-in-extreme-slow-mo/); [RPM spin physics](https://rpmpb.com/pages/spin-physics)
- For comparison, tennis ball wind-tunnel drag coefficients are mostly about 0.5–0.7. — [TWU tennis aerodynamics](https://twu.tennis-warehouse.com/learning_center/aerodynamics2.php)
- PB Vision's help centre warns that Lindsey's measurements cover only 11–29 mph and should not be extrapolated as a universal threshold for harder drives. — [PB Vision help](https://help.pb.vision/articles/7781231-how-does-pb-vision-calculate-speed)

### Inferences
- Suggested flight model: dv/dt = g − k_D‖v‖v + k_L‖v‖(ω̂ × v), with k_D = ρ·Cd·A/(2m). Using mid-spec mass 24.3 g, diameter 74.3 mm (A ≈ 4.34×10⁻³ m²), ρ = 1.21 kg/m³: k_D ≈ 0.036 m⁻¹ for Cd = 0.33 (outdoor ball) and ≈ 0.049 m⁻¹ for Cd = 0.45 (indoor ball). These are my calculations from the cited values.
- Consequences of those numbers (my calculations): terminal velocity is about 16–17 m/s (about 37 mph) for the outdoor ball; at 20 m/s drag deceleration is about 14 m/s², i.e. more than gravity. Drag is therefore first-order for drives and serves and cannot be omitted, unlike in simulators such as Where Is The Ball's. A tennis ball with Cd 0.55 has k_D ≈ 0.021 m⁻¹, so a pickleball decelerates roughly 1.7–2.4 times faster at the same speed.
- Pro tour play uses outdoor-style 40-hole balls, so Cd about 0.30–0.33 is the right prior; but all measurements are at or below 13 m/s (Reynolds number up to roughly 6×10⁴ by my estimate), while pro drives reach 20–27 m/s. Treat Cd as a per-match (or per-arc, bounded, as MonoTrack does) fitted parameter with a prior of about 0.3–0.45.
- Spin parameter for a 1,200 rpm (20 rev/s) topspin drive at 20 m/s is S ≈ 0.23, giving Cl ≈ 0.045 by Steyn's formula but up to about 0.2 by Lindsey's low-spin data. The two sources differ by several times, so Magnus should be a single bounded scalar per arc (topspin/backspin about the horizontal axis perpendicular to travel) or omitted in a first version. Monocular spin is not observable directly and is only inferred from curvature (as TT4D notes for table tennis).
- Bounce model starting values: normal COR about 0.62–0.66 at about 6.2 m/s impact (the spec drop speed, my calculation), on granite. COR on an acrylic court, at higher speeds, with a worn ball, and at different temperatures will differ; fit it from observed pre- and post-bounce arcs.

### Gaps
- No wind-tunnel drag/lift measurements for pickleballs at match speeds (above 13 m/s) were found.
- No published tangential (friction) bounce model or spin-dependent bounce data for a pickleball on a hard court was found.
- No radar- or tracking-based distribution of pro-match ball speeds and spin rates was found; the speed/spin figures above come from paddle-test sites and enthusiast analyses.
- I could only see a mirror listing of a 2025 SAGE article (DOI 10.1177/17479541251365200) that may be the journal version of Steyn et al.; authorship and final values not confirmed.
- I did not open Creer (2023) or Emond et al. (2024) directly; their values are as reported by Lindsey and Steyn.

## 5. What pickleball-specific datasets, papers, open-source repos, and commercial systems exist?

### Takeaway
Pickleball has no academic benchmark; the public ecosystem is a few TrackNet fine-tune repos, small Roboflow detection sets, and one AGPL stats pipeline, none of which does 3D ball reconstruction. Commercial single-camera products (PB Vision, SwingVision) claim 3D game recreation and 3D ball tracking but publish no method or independent accuracy numbers.

### Cited Findings

**Open source and datasets**
- `roberto-cantero/dink-data` ("DinkData"): a fork of "Rafa's PBTrack"; Python pipeline that takes a single-camera PickleCam recording of a doubles game and finds the court, ball, and four players, then builds a game log of rallies, hits, bounces, and rally winners for per-player stats and highlight reels. AGPL-3.0. A search summary describes it as using a TrackNetV3 retrained for pickleball, but the README I fetched names no models; no accuracy numbers; no 3D reconstruction mentioned. — [roberto-cantero/dink-data](https://github.com/roberto-cantero/dink-data)
- `AndrewDettor/TrackNet-Pickleball`: TrackNetV2 training workflow and pickleball weights (Google Drive). — [AndrewDettor/TrackNet-Pickleball](https://github.com/AndrewDettor/TrackNet-Pickleball)
- `optivisionlab/pickleball_track`: MIT-licensed repo; contents not inspected. — [optivisionlab/pickleball_track](https://github.com/optivisionlab/pickleball_track)
- `changtimwu/ballvideo3dreconstruction` issue #1: a plan (not working code) for TrackNetV3-style 2D detection followed by 3D ball position and event detection from match footage. — [GitHub issue](https://github.com/changtimwu/ballvideo3dreconstruction/issues/1)
- Roboflow Universe pickleball sets: see section 1. One by Liberin Technologies targets courts and balls for in/out calls and broadcast analytics. — [Pickleball Vision (Roboflow)](https://universe.roboflow.com/liberin-technologies/pickleball-vision)
- The only pickleball tracking write-up with numbers is the Univ. of Florida TrackNetV2 transfer-learning project on PPA footage (section 1). — [hudsondev.pages.dev/pickleball](https://hudsondev.pages.dev/pickleball)
- RacketVision, the largest 2025–2026 racket-sport benchmark, covers table tennis, tennis, and badminton only. — [arXiv 2511.17045v3](https://arxiv.org/html/2511.17045v3)

**Commercial systems**
- PB Vision: users record a full game on a phone; the site says it "tracks the players, court, and every shot to recreate the game in 3D". — [pb.vision](https://pb.vision/)
- PB Vision capture guidance (third-party coverage): camera at least four feet above the ground with all four court corners visible; upload after the game, one game at a time, with processing done server-side. — [Pickleball Union](https://pickleballunion.com/pb-vision-pickleball-ai/)
- PB Vision reports shot speed as the average over the tracked flight from paddle contact until bounce or volley (distance over time), not peak or contact speed; it notes that for volleyed balls the landing point is never observed and must be modelled with uncertainty, and that spin and air resistance affect landing. — [PB Vision help](https://help.pb.vision/articles/7781231-how-does-pb-vision-calculate-speed)
- SwingVision (press release 5 Dec 2023): claims patent-pending AI that "tracks the ball trajectory in 3D" using only an iPhone or iPad, with real-time stats, highlights, and line calls for pickleball, and calls itself "the most accurate single camera solution"; no methodology or independent verification is given. — [PRWeb release](https://www.prweb.com/releases/ai-has-entered-the-kitchen-swingvision-brings-its-award-winning-ai-stats-and-line-calling-to-pickleball-302005962.html)
- A US patent (11,615,540) describes the generic single-camera recipe: candidate 2D detections, a 3D physics model including bounce, spin, and air resistance, a fixed camera projection matrix, and an optimiser choosing a consistent subset of candidates. The search results did not identify the assignee, so I cannot tie it to a product. — [USPTO 11615540](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11615540)

### Inferences
- PB Vision's choice to report average flight speed rather than instantaneous speed, and its caveats about volleys, suggest its 3D ball estimate is anchored at contact and bounce events and is least certain for volley-to-volley exchanges. This is my reading of the help article, not a statement by the company.
- Because commercial systems accept amateur phone footage from low fence-height cameras, a fixed elevated broadcast camera with a full view of the court lines is an easier geometry for calibration, though broadcast cuts, replays, and graphics add a shot-boundary filtering step the consumer apps do not face.
- The AGPL license on dink-data makes it unsuitable to copy into a differently licensed project, but it is a useful reference for pipeline structure and for pickleball-specific event rules.

### Gaps
- I found no technical blog, paper, or patent in which PB Vision or SwingVision describes its 3D method or reports accuracy against ground truth.
- I did not find information on PlaySight or Hawk-Eye deployments for pickleball; I cannot say what, if anything, they offer for the sport from one camera.
- I did not find the upstream "PBTrack" repository or any public pickleball weights with a stated license.
- No public pickleball dataset with temporally consistent ball tracks from broadcast (PPA/MLP) footage was found; the Roboflow sets are single-frame boxes.
- No published pickleball court-detection or shot-classification paper was found in this search; court detection approaches would need to be adapted from tennis/badminton work (e.g. MonoTrack's modified Farin et al. line method, 85.5% success at IoU > 0.8). — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)

## 6. What are the known failure modes and how do the best methods mitigate them?

### Takeaway
The recurring failures are occlusion and missed detections, event-segmentation errors that corrupt neighbouring arcs, depth ambiguity along the camera axis (worst in exactly the behind-the-court view), the ball leaving the frame, and out-of-distribution trajectories such as volleys. The effective mitigations are temporal models with inpainting or token dropping, anchoring arcs to 3D contact points and the court plane, physics constraints with bounded parameters, and cross-checking two detectors.

### Cited Findings
- Occlusion (2D): TrackNetV3 repairs occluded stretches with InpaintNet; Volley3D trains InpaintNet with 30% random masking to simulate failures. — [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3); [arXiv 2610.38358](https://arxiv.org/html/2610.38358)
- Occlusion (2D): TOTNet [NEW] uses occlusion augmentation and a visibility-weighted loss; even so, tennis fully-occluded accuracy is only 0.33 in its table, so occluded positions remain unreliable. — [arXiv 2508.09650](https://arxiv.org/html/2508.09650v1)
- Missing detections (3D): Kienzle et al. [NEW] drop tokens for missing detections and train with 5% dropout and variable frame rates; TT4D [NEW] lifts the unsegmented track with a learned network specifically to survive heavy occlusion. — [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1); [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- False positives: Kienzle et al. keep a ball detection only if two different detectors agree within 20 px. — [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1)
- Depth ambiguity: TT3D's 3D error grows from 8.9 cm (side view) to 22.3 cm (back view), 29.8 cm with noise. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Depth ambiguity mitigation: MonoTrack anchors each arc to hitter and receiver positions and constrains heights and direction, halving synthetic 3D error (14.9 → 8.0 cm). Volley3D [NEW] fixes segment ends at reconstructed 3D hand positions and gets 0.27 m RMSE versus 1.56 m without. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899); [arXiv 2610.38358](https://arxiv.org/html/2610.38358)
- Ball leaving the frame: MonoTrack reports higher errors for shots to the back court where the shuttle may leave the frame. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- Segmentation error propagation: MonoTrack estimates about 20% of shots may be wrong given about 90% hit-detection accuracy. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899)
- Too few observations: TT3D needs at least three visible frames after a bounce; early returns or occlusion break it. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)
- Out-of-distribution trajectories: Where Is The Ball fails on volleys near the net and requires the sequence to start and end on the ground. — [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Unmodelled physics: MonoTrack omits spin and time-varying drag; Volley3D neglects Magnus and assumes instantaneous hits and no net contacts; Where Is The Ball's simulator ignores spin and aerodynamics. — [arXiv 2204.01899](https://ar5iv.labs.arxiv.org/html/2204.01899); [arXiv 2610.38358](https://arxiv.org/html/2610.38358); [arXiv 2506.05763](https://arxiv.org/html/2506.05763v1)
- Camera motion: TT3D's polynomial segmentation and Kienzle et al.'s keypoint aggregation both assume a static camera. — [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1); [arXiv 2511.20250](https://arxiv.org/html/2511.20250v1)
- Generalisation: the pickleball TrackNetV2 fine-tune overfit to one match and degraded on angled views and occlusions. — [hudsondev.pages.dev/pickleball](https://hudsondev.pages.dev/pickleball)
- Unobserved landing: PB Vision notes a volleyed ball's landing point is never observed and must be modelled with uncertainty. — [PB Vision help](https://help.pb.vision/articles/7781231-how-does-pb-vision-calculate-speed)
- Motion blur: BlurBall [NEW] shows labelling the blur-streak midpoint improves every tested detector and yields blur direction and length as extra cues; TT3D uses blur direction in its segmentation cost. — [arXiv 2509.18387v2](https://arxiv.org/html/2509.18387v2); [arXiv 2504.10035](https://arxiv.org/html/2504.10035v1)

### Inferences
- Lobs leaving the top of the frame: because an arc is an ODE determined by a handful of parameters, it can be fitted from the visible ascending and descending portions plus the anchors at the hit and the next bounce/hit, and integrated through the unseen apex. Drag makes the arc asymmetric, so the fit should use the full drag model rather than a parabola. This is my recommendation; MonoTrack only reports the problem.
- Depth toward the camera: in a behind-the-baseline view, motion along the court's long axis projects mostly to small vertical image changes that trade off against height. The strongest disambiguators are (a) bounce points, where height is known exactly so the ray-plane intersection fixes 3D position, (b) 3D wrist/paddle position of the hitter from pose estimation, (c) apparent ball size and blur length as weak depth/speed cues, and (d) net clearance as an inequality constraint (the ball must pass above the net at the net plane). Items (c) and (d) are my suggestions, not from a cited paper.
- Volley exchanges at the kitchen line are the hardest case for pickleball: short arcs of a few frames, both ends at paddles (no ground anchor), heavy player occlusion. Expect to rely on pose-derived paddle positions and strong priors (bounded speed, near-straight path) and to accept lower accuracy there.
- Near-side players in a behind-baseline view regularly occlude the ball and the far kitchen area; offline processing permits bidirectional smoothing/inpainting across gaps and a global re-fit of each rally.
- Since smoothness is the priority, render from the fitted ODE arcs rather than from per-frame lifted points; per-frame detections only enter through the loss. A robust loss (Huber or similar) limits damage from residual false positives.
- Broadcast-specific issues not covered by the papers: camera cuts, replays, and slow-motion inserts must be filtered before tracking, and a per-shot static-camera check is needed since the background-median trick and fixed-homography calibration assume a stationary camera.

### Gaps
- No source quantifies these failure modes for pickleball or for a doubles sport with four players at the net.
- No cited method evaluates recovery of arcs whose apex is out of frame; the lob-handling approach above is untested inference.
- No source reports using apparent ball size or net-clearance constraints for depth disambiguation in racket sports in the papers I read.
- No end-to-end 3D ground-truth evaluation on broadcast footage exists in the works reviewed (Volley3D and Kienzle et al. both say so), so real-world 3D accuracy of any monocular method on broadcast video is essentially unmeasured.
