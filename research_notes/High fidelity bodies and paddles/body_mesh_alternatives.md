# Alternatives to SAM 3D Body for smooth, world-grounded body meshes from monocular video (as of Nov 2026)

Scope note: numbers are copied from the tables of the cited papers. The same method often has different numbers in different papers (e.g. GVHMR EMDB-2 W-MPJPE is 276.5 in most tables but 292.7 when re-run by HTD-Refine), so compare only within one row-source. "EMDB" camera-space = EMDB-1; world-space = EMDB-2. All benchmark tables below were read from the paper text (PDF/HTML), not from aggregators, unless marked "search snippet".

## 1. Which methods lead in 2024-2026 for world-grounded video mesh recovery, and what accuracy do they report?

### Takeaway
No single method wins. On published tables, PromptHMR-Vid and GENMO/GEM lead the released, runnable methods for pose and world trajectory, while GVHMR remains the best speed/quality trade-off and has the lowest foot sliding among the feed-forward video models; newer 2026 papers (FactorizedHMR, HTD-Refine, DanceHMR, JOSH) post better numbers but mostly have no confirmed code release. On EMDB-2, the leaders are all in the 175-220 mm W-MPJPE range, i.e. world error is still dominated by trajectory, which in the pickleball setup is already known from calibrated foot tracks.

### Cited Findings

Camera-space accuracy (PA-MPJPE / MPJPE / PVE / Accel, mm):
- 3DPW: WHAM 37.2/59.4/71.0/6.9; GVHMR 36.2/55.6/67.2/5.0; TRAM 35.6/59.3/69.6/4.9; PromptHMR 35.5/56.9/67.3/-; GENMO 34.6/53.9/65.8/5.2 (as collected from each method's own paper) — [CalTennis, Table 4](https://arxiv.org/pdf/2606.20542)
- RICH: WHAM 44.7/82.6/93.2/5.6; GVHMR 39.5/66.0/74.4/4.1; PromptHMR 37.0/57.4/65.8/-; GENMO 39.1/66.8/75.4/4.1 — [CalTennis, Table 4](https://arxiv.org/pdf/2606.20542)
- EMDB-1: WHAM 48.8/80.7/93.7/5.9; GVHMR 42.7/72.6/84.2/3.6; TRAM 45.7/74.4/86.6/4.9; PromptHMR 40.1/68.1/79.2/-; GENMO 42.5/73.0/84.8/3.8 — [CalTennis, Table 4](https://arxiv.org/pdf/2606.20542)
- DanceHMR (ByteDance, June 2026, SMPL-X body + hands, video): 3DPW 35.2/53.6/65.1/4.9; EMDB 40.1/64.3/75.4/3.5. Same table lists DUOMO 41.8/67.1/78.2 on EMDB, Multi-HMR 50.1/81.6/95.7 on EMDB and 41.7/61.4/75.9 on 3DPW, SMPLest-X 43.2/70.5 (PA-MPJPE/MPJPE) on 3DPW — [DanceHMR, Table 2](https://arxiv.org/pdf/2605.18102)
- FactorizedHMR (May 2026, SMPL-X, flow matching): 3DPW 36.9/56.2/68.3; RICH 39.1/65.8/74.5; EMDB 45.7/73.6/86.5 — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- Per-frame image models on the same benchmarks (from the SAM 3D Body paper's Table 2): CameraHMR 3DPW 35.1/56.0/65.9, EMDB 43.3/70.3/81.7, RICH 34.0/55.7/64.4; PromptHMR (image) 3DPW 36.1/58.7/69.4, EMDB 41.0/71.7/84.5, RICH 37.3/56.6/65.5; NLF-L+fit 3DPW 33.6/54.9/63.7, EMDB 40.9/68.4/80.6, RICH 28.7/51.0/58.2 (NLF was trained on RICH); SMPLer-X-H EMDB 64.5/92.7/112.0 — [SAM 3D Body paper, Table 2](https://arxiv.org/html/2602.15989)
- Multi-HMR 2 (June 2026; multi-person detection + mesh + tracking) reports 48.3 PA-MPJPE on EMDB with predicted camera (search snippet; not read in full) — [Multi-HMR 2](https://arxiv.org/pdf/2606.14841)

World-space accuracy on EMDB-2 (WA-MPJPE100 / W-MPJPE100 in mm / RTE %):
- WHAM (w/ DPVO) 135.6/354.8/6.0; TRAM 76.4/222.4/1.4; GVHMR (w/ DPVO) 111.0/276.5/2.0; PromptHMR-Vid 71.0/216.5/1.3; JOSH 68.9/174.7/1.3; COIN 152.8/407.3/3.5; Human3R (online) 112.2/267.9/2.2; OnlineHMR (online, CVPR 2026) 93.5/310.4/2.2 — [OnlineHMR, Table 2](https://arxiv.org/pdf/2603.17355)
- GENMO 65.1/210.9/1.0; FactorizedHMR 70.5/192.5/1.5 — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- The GENMO paper itself reports 202.1 mm W-MPJPE on EMDB vs TRAM 222.4 "despite both methods utilizing identical SLAM systems and backbone features" (search snippet; differs from the 210.9 quoted by FactorizedHMR) — [GENMO paper](https://arxiv.org/html/2505.01425)
- WATCH (arXiv 2509.04600) reports 103.3/251.6 WA/W-MPJPE on EMDB-2 with ground-truth camera vs 109.1/274.9 for GVHMR (search snippet) — [WATCH](https://arxiv.org/html/2509.04600v2)
- HTD-Refine (ZJU, May 2026; a post-processing optimizer) takes GVHMR from 118.7/292.7/2.1 to 69.2/192.4/1.5 and TRAM from 78.8/221.3/1.5 to 71.7/204.9/1.5 — [HTD-Refine](https://arxiv.org/html/2605.26879v1)

World-space on RICH (static camera, the closest benchmark to a fixed broadcast camera):
- WA-MPJPE / W-MPJPE / RTE / Jitter / Foot sliding: WHAM 109.9/184.6/4.1/19.7/3.3; GVHMR 78.8/126.3/2.4/12.8/3.0; GENMO 80.7/127.2/2.6/8.6/2.7; FactorizedHMR 86.6/123.3/2.5/17.7/8.6 — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- TRAM (with trajectory filter) on RICH: 103.6/168.4/2.7, jitter 18.7, foot sliding 12.9 mm; GVHMR 77.4/124.0/2.5, jitter 13.0, foot sliding 3.3 — [HTD-Refine](https://arxiv.org/html/2605.26879v1)

Jitter and foot sliding on EMDB-2 (Jitter / Foot sliding mm):
- WHAM 22.5/4.4; GVHMR 16.7/3.5; TRAM 18.1/11.0; GENMO 9.58/8.3; FactorizedHMR 17.7/9.3 — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- After HTD-Refine: GVHMR jitter 17.2 -> 7.2 (foot sliding 4.0 -> 5.7); TRAM jitter 25.1 -> 6.6 (foot sliding 12.0 -> 7.5); on RICH GVHMR jitter 13.0 -> 3.6, foot sliding unchanged at 3.3 — [HTD-Refine](https://arxiv.org/html/2605.26879v1)

Speed (single RTX 6000 Ada, end-to-end incl. preprocessing): WHAM 9.3 FPS; TRAM 2.1 FPS; GVHMR 12.1 FPS; Human3R 4.8 FPS; OnlineHMR 3.3 FPS. The authors caution the FPS figures are indicative only — [OnlineHMR, Table 3](https://arxiv.org/pdf/2603.17355)
- GVHMR's temporal head itself is tiny: 0.41 s and 304 MB peak GPU memory for a 1193-frame sequence on an RTX A5000, vs 3.92 s / 1057 MB for FactorizedHMR (50 denoising steps) — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- On CalTennis, GVHMR "has the best runtime performance ... an optimal tradeoff when it comes to running quickly and accurately"; PromptHMR is "also the slowest" and "seems to scale quadratically" with frame count due to its temporal transformer; GENMO scales linearly — [CalTennis, App. A.6-A.7](https://arxiv.org/pdf/2606.20542)

Independent benchmark on tennis (most relevant third-party evaluation found):
- CalTennis (Caltech, June 2026): 11M+ frames / 51 h of tennis practice and match play, 40 players, 2-6 synchronized 1080p 60 Hz phone cameras; evaluates multi-view consistency (no mocap) of PromptHMR, WHAM, GVHMR, TRAM, GENMO on an NVIDIA H100 over the first 5M frames, each model running its own camera estimation (known camera not supplied) — [CalTennis](https://arxiv.org/pdf/2606.20542)
- CalTennis Table 2 (mm; Foot-Vel in m/s) — Translation / Pose / MPJPE / PA-MPJPE / Foot-Vel / Foot-Height / Stability: PromptHMR 942/105/1785/84/3.23/70/25; WHAM 2664/106/2675/119/0.72/150/44; GVHMR 3587/109/1066/88/2.49/60/21; TRAM 2340/115/958/91/6.65/80/33; GENMO 2560/110/1020/91/4.40/60/16 — [CalTennis, Table 2](https://arxiv.org/pdf/2606.20542)
- CalTennis conclusions: "is any one model best? No." PromptHMR leads translation/pose/PA-MPJPE but has the largest per-video variance and is worst on acceleration-type metrics; WHAM is far best on foot-velocity consistency (attributed to its foot-contact refinement) but much worse on translation; GENMO is most consistent on foot height and stability; "three quantities remain unreliable across every model we tested: absolute distance and depth, ground-contact detection, and body shape (limb lengths, height, proportions)"; "the highest reported performance on CalTennis is significantly worse than other benchmarks for all models across all metrics" — [CalTennis](https://arxiv.org/pdf/2606.20542)
- Internal inconsistency in CalTennis: the Figure 5 caption says "WHAM excels in foot height consistency" while Table 2 lists WHAM with the worst Foot-Height value (150 mm) and the best Foot-Vel (0.72 m/s) — [CalTennis](https://arxiv.org/pdf/2606.20542)

### Inferences
- For a fixed, calibrated camera, the relevant benchmark column is RICH (static camera), where GVHMR and GENMO are essentially tied (about 126-127 mm W-MPJPE) and GVHMR has low foot sliding (3.0-3.3 mm) with no post-filter; the big EMDB-2 gaps between methods mostly reflect moving-camera/SLAM handling, which does not apply here.
- CalTennis translation errors of 0.9-3.6 m arise with each model guessing its own camera and depth. With a known camera and court-plane foot tracks, root translation can be imposed externally, so the method choice should be driven by local pose quality (PA-MPJPE, wrists), smoothness and foot contact, not by the trajectory columns.
- Ranking for local pose from published numbers: PromptHMR-Vid ~ DanceHMR > GENMO ~ GVHMR > TRAM > WHAM; ranking for smoothness without post-filter: GENMO (jitter 8.6-9.6) > GVHMR (12.8-16.7) > TRAM/WHAM; ranking for foot sliding: GVHMR ~ WHAM ~ GENMO(RICH) > TRAM.

### Gaps
- JOSH, DUOMO, COIN, Human3R and WATCH numbers were read only from other papers' tables or search snippets; their code status and requirements were not checked.
- No leaderboard-style neutral re-evaluation of all 2026 methods on EMDB/RICH was found; all numbers except CalTennis are self-reported.
- Multi-HMR 2 and SMPLest-X world-grounded results were not found (both are per-frame camera-space models).

## 2. Published or independent head-to-head comparisons with SAM 3D Body

### Takeaway
The only quantitative head-to-heads found are in Meta's own paper and are per-frame: SAM 3D Body (3DB) beats CameraHMR, PromptHMR (image), NLF and the video methods WHAM/TRAM/GENMO on EMDB camera-space error, but nobody has published a world-grounded, jitter or foot-sliding comparison of 3DB against GVHMR/PromptHMR-Vid/GENMO. Third-party follow-ups all treat raw 3DB as temporally unstable on video and bolt tracking, smoothing and contact optimization on top.

### Cited Findings
- EMDB (PA-MPJPE/MPJPE/PVE): 3DB-H 38.5/62.9/74.3, 3DB-DINOv3 38.2/61.7/72.5 vs NLF-L+fit 40.9/68.4/80.6, PromptHMR 41.0/71.7/84.5, CameraHMR 43.3/70.3/81.7, GENMO (video) 42.5/73.0/84.8, TRAM 45.7/74.4/86.6, WHAM 50.4/79.7/94.4 — [SAM 3D Body paper, Table 2](https://arxiv.org/html/2602.15989)
- 3DPW: 3DB-H 33.2/54.8/64.1 vs NLF-L+fit 33.6/54.9/63.7, GENMO 34.6/53.9/65.8, CameraHMR 35.1/56.0/65.9; RICH: 3DB-DINOv3 30.9/53.7/60.3 vs NLF 28.7/51.0/58.2 (trained on RICH), CameraHMR 34.0/55.7/64.4 — [SAM 3D Body paper, Table 2](https://arxiv.org/html/2602.15989)
- 2D alignment: COCO PCK@0.05 3DB-H 86.8 vs CameraHMR 80.5, PromptHMR 79.2, NLF 74.9; LSPET (extreme sports poses) 68.9 vs PromptHMR 55.6, NLF 54.9, CameraHMR 49.1 — [SAM 3D Body paper, Table 2](https://arxiv.org/html/2602.15989)
- Out-of-domain 3D sets, leave-one-out 3DB-H vs best baseline (PVE/MPJPE): Harmony4D 63.5/54.0 vs CameraHMR 84.6/70.8; EE4D-Phys 49.7/44.3 vs CameraHMR 71.1/58.8; SA1B-Hard Avg-PCK 73.1 vs NLF 66.5, CameraHMR 63.0, PromptHMR 59.0 — [SAM 3D Body paper, Table 3](https://arxiv.org/html/2602.15989)
- The 3DB paper states it is "competitive with video-based approaches that additionally leverage temporal information"; it processes each person separately per image and reports no temporal-consistency analysis — [SAM 3D Body paper](https://arxiv.org/html/2602.15989)
- SAM-Body4D (Dec 2025, training-free): per-frame 3DB on video "cannot ensure continuous mesh trajectories ... When the target becomes small, suffers motion blur, or is heavily occluded, the detector may fail to localise the human, leading to missing meshes"; it adds SAM-3-style video masklets, a diffusion-based occlusion-aware mask refiner and batched multi-person inference; code at github.com/gaomingqi/sam-body4d; compute example is an NVIDIA A100 80 GB; evaluation is qualitative only (no benchmark tables vs GVHMR/PromptHMR) — [SAM-Body4D](https://arxiv.org/pdf/2512.08406)
- "World-Coordinate Human Motion Retargeting via SAM 3D Body" (Dec 2025): uses 3DB as a frozen backbone, locks identity/skeleton scale per track, runs sliding-window smoothing in MHR latent space to remove "high-frequency jitter", then a contact-aware root optimizer with soft foot-contact probabilities and losses for sliding, penetration and contact, because per-frame output otherwise shows "non-physical shaking and foot sliding"; results are qualitative; it lists quantitative evaluation as future work — [World-Coordinate Retargeting via SAM 3D Body](https://arxiv.org/pdf/2512.21573)
- Fast SAM 3D Body (Mar 2026): up to 10.9x end-to-end speedup with "on-par reconstruction fidelity" and a feed-forward MHR-to-SMPL conversion "over 10,000x" faster than iterative fitting; compares only against original 3DB — [Fast SAM 3D Body](https://arxiv.org/abs/2603.15603)
- CalTennis cites SAM 3D Body but did not include it among its five benchmarked models (PromptHMR, WHAM, GVHMR, TRAM, GENMO) — [CalTennis](https://arxiv.org/pdf/2606.20542)
- OnlineHMR, FactorizedHMR, HTD-Refine and DanceHMR (all 2026) do not include SAM 3D Body in any table — [OnlineHMR](https://arxiv.org/pdf/2603.17355); [FactorizedHMR](https://arxiv.org/html/2605.14854); [HTD-Refine](https://arxiv.org/html/2605.26879v1); [DanceHMR](https://arxiv.org/pdf/2605.18102)

### Inferences
- On per-frame accuracy 3DB is ahead of every SMPL-family alternative by roughly 2-6 mm PA-MPJPE on EMDB and by a wide margin on hard/athletic 2D alignment (LSPET); the alternatives' advantage is entirely temporal: built-in smoothness, world trajectory and foot-contact modeling.
- The practical question is therefore not "3DB or GVHMR" but "3DB plus a temporal/contact layer" versus "a native video model"; every third-party 3DB video pipeline found re-implements smoothing and contact-aware root solving, which a known camera and court-plane foot tracks would make simpler.
- A hybrid is plausible but unpublished: per-frame 3DB poses refined by a plug-in optimizer such as HTD-Refine (which was only shown on TRAM, GVHMR, Human3R).

### Gaps
- No independent benchmark of 3DB vs GVHMR/PromptHMR-Vid/GENMO on W-MPJPE, jitter, accel or foot sliding was found.
- No numbers on how 3DB degrades with person height in pixels were found.

## 3. Small, distant, low-resolution or occluded people; hands and wrists

### Takeaway
Evidence on small people is thin and mostly indirect: PromptHMR is the only method that explicitly claims operation on very small boxes via full-image processing, while crop-based video models (GVHMR, GENMO) measurably degrade as the person gets smaller. For hands, body-only video models do not model fingers; the published options are SMPL-X whole-body models (SMPLest-X: accurate but jittery per-frame; DanceHMR: smooth, best hand numbers, code unconfirmed) or pairing a body model with a hand model (HaMeR/WiLoR), which yields unnatural wrists.

### Cited Findings
- PromptHMR processes the whole image rather than person crops and its abstract says it can estimate people from boxes "as small as faces in crowded scenes"; prompts are boxes, masks, language and interaction labels — [PromptHMR abstract](https://arxiv.org/abs/2504.06397)
- CalTennis failure analysis: image scale is strongly negatively correlated with MPJPE for GVHMR (rs = -0.44) and GENMO (rs = -0.44), "indicating that persons appearing smaller in the image (i.e., farther from the camera) are harder to reconstruct"; GVHMR and GENMO failures are "strongly tied to scene geometry (depth, camera distance, scale)", while TRAM and WHAM show near-zero geometric signatures (errors spread more uniformly) — [CalTennis, App.](https://arxiv.org/pdf/2606.20542)
- SAM 3D Body under truncation/occlusion (SA1B-Hard aPCK): lower-body truncated 61.95 vs CameraHMR 39.27, PromptHMR 46.50; occlusion with foot cues 75.43 vs 62.93 / 58.00; severe truncation PVE 113.66 vs CameraHMR 213.64, PromptHMR 168.22. The paper has no explicit low-resolution category — [SAM 3D Body paper](https://arxiv.org/html/2602.15989)
- FactorizedHMR reports its clearest gains under occlusion (3DPW-XOCC) — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- GEM/GENMO's project page says estimation under occlusion benefits from its generative prior — [GEM project page](https://research.nvidia.com/labs/dair/gem)
- Hands, ARCTIC (PA-PVE all/hands; PVE all/hands, mm): SMPLest-X 26.3/11.9; 43.5/28.8. GVHMR+HaMeR 41.0/13.9; 71.0/54.6. DanceHMR 28.1/8.5; 44.7/24.8. UBody hands PA-PVE: SMPLest-X 9.5, AiOS 7.3, DanceHMR 7.4 — [DanceHMR, Table 1](https://arxiv.org/pdf/2605.18102)
- Temporal stability on ARCTIC (Jitter / Accel / MPJVE, whole body): SMPLer-X-B 127.5/23.8/482.6; SMPLest-X 95.1/17.8/364.6; GVHMR+HaMeR 6.0/1.8/109.0; DanceHMR 5.4/1.5/81.3. Hands jitter: SMPLest-X 55.4, GVHMR+HaMeR 7.4, DanceHMR 4.2 — [DanceHMR, Table 3](https://arxiv.org/pdf/2605.18102)
- DanceHMR on stitching body and hand models: GVHMR + HaMeR combinations "often produce unnatural wrist orientations and temporal inconsistencies"; image whole-body models (SMPLer-X, SMPLest-X) "suffer from severe temporal inconsistency"; SMPLest-X hallucinates implausible hands when the hand is out of view. Model sizes: SMPLer-X-B 103M, SMPLer-X-L 327M, SMPLest-X 687M, DanceHMR about 46M parameters — [DanceHMR](https://arxiv.org/pdf/2605.18102)
- SAM 3D Body hand accuracy on FreiHAND without FreiHAND training: PA-MPJPE 5.5 mm (equal to WiLoR 5.5 and MaskHand 5.5; HaMeR 6.0), PA-MPVPE 6.2-6.3 vs WiLoR 5.1 — [SAM 3D Body paper, Table 4](https://arxiv.org/html/2602.15989)
- GEM-X is a separate NVIDIA repository for full-body estimation including hands and face; the released GEM-SMPL checkpoint is body-only SMPL — [NVlabs/GENMO README](https://github.com/NVlabs/GENMO)
- TT4D (table-tennis broadcast) drops reconstructions with "invalid human pos" in 9.4% of clips that reach reconstruction — [TT4D](https://arxiv.org/html/2605.01234)

### Inferences
- For the far players (80-150 px tall), hand articulation is below what any of these models can resolve from pixels (a hand is roughly 8-15 px); wrist orientation there will come from body-model priors and, in this project, from paddle pose. Hand models are only worth running on the two near players.
- A known 2D keypoint track is the most robust conditioning for small people; methods that take 2D keypoints as a first-class input (GVHMR, GENMO/GEM, WHAM) can benefit from the project's existing keypoints, whereas full-image regressors must re-detect the person.
- Upscaling/cropping far players before a crop-based model is a likely necessary step, but no source quantified it.

### Gaps
- No paper found reports HMR accuracy binned by person pixel height for any of these methods; "which handles 80-150 px people best" is unanswered by published evidence.
- NLF's and Multi-HMR's small-person behaviour was not found in the sources read (Multi-HMR is commonly run at high input resolution for this reason, but no source was retrieved).
- DanceHMR code release is unconfirmed (only a project page: https://shenwenhao01.github.io/dancehmr/).

## 4. Which accept a known camera, 2D keypoints or masks, and which are temporally smooth without post-filtering?

### Takeaway
GVHMR, GEM/GENMO and PromptHMR-Vid all ship a static-camera flag that skips SLAM, and GVHMR additionally accepts focal length/intrinsics; none of the released tools documents accepting a full known extrinsic (camera pose + ground plane), so world placement from the court calibration has to be applied by the user. Native video models (GVHMR, GENMO, WHAM, TRAM, PromptHMR-Vid, DanceHMR) are smooth by design; per-frame models (SAM 3D Body, CameraHMR, NLF, SMPLest-X, Multi-HMR) need added smoothing.

### Cited Findings
- GVHMR official repo: `-s` flag skips visual odometry ("if you know the camera is static"); `f_mm` option to specify full-frame focal length; since 2025-03-08 defaults to its own SimpleVO instead of DPVO — [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR)
- Third-party `gvhmr` PyPI fork: `-s/--static-cam`, focal length in pixels (`--f_px`), intrinsics sidecar (`--intrinsics`), `--f_mm`; YOLO detectors and optional RTMPose 2D pose; it does not document feeding precomputed 2D keypoints — [gvhmr on PyPI](https://pypi.org/project/gvhmr/)
- GEM/GENMO: one diffusion model conditioned on video, keypoints, text, audio/music and 3D keyframes; repo has `-s/--static_cam`; no documented known-intrinsics input (outputs `K_fullimg`); webcam pipeline computes 2D keypoints with ViTPose-H — [NVlabs/GENMO README](https://github.com/NVlabs/GENMO); [GEM project page](https://research.nvidia.com/labs/dair/gem)
- PromptHMR: prompts are boxes and masks (spatial) plus text and interaction labels (semantic); the video demo has `--static_camera` and `--gravity_align`; world-video pipeline components are Detectron2, SAM2, DROID-SLAM, Metric3D, ViTPose, SPEC; the video head design is credited to GVHMR — [PromptHMR abstract](https://arxiv.org/abs/2504.06397); [yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR)
- CalTennis describes PromptHMR as conditioning "on prompts such as 2D keypoints and contacts" (differs from the abstract's list of boxes/masks/text/interaction) — [CalTennis](https://arxiv.org/pdf/2606.20542)
- SAM-Body4D feeds video masklets as prompts to SAM 3D Body at both encoder and decoder to stabilize identity — [SAM-Body4D](https://arxiv.org/pdf/2512.08406)
- HTD-Refine is explicitly a plug-in: takes any baseline world-space SMPL trajectory plus camera estimate, predicts 2D keypoints and camera-space 3D joint velocities/accelerations with PVA-Net (ViTPose-L + 8-layer temporal transformer), then globally optimizes with velocity, acceleration, keypoint, jerk and regularization losses, plus optional foot-contact post-processing; about 2 minutes per video on an A6000; "Code and models will be made publicly available" (project page zju3dv.github.io/htd-refine) — [HTD-Refine](https://arxiv.org/html/2605.26879v1)
- Smoothness without post-filter (Accel, mm/s^2 style metric as reported): GVHMR 3.6-5.0, GENMO 3.8-5.2, TRAM 4.9, WHAM 5.3-6.9, DanceHMR 3.5-4.9; per-frame HMR2.0 18.1-19.9 and ReFit 18.5-20.7 for contrast — [CalTennis, Table 4](https://arxiv.org/pdf/2606.20542); [DanceHMR](https://arxiv.org/pdf/2605.18102); [OnlineHMR, Table 1](https://arxiv.org/pdf/2603.17355)
- PromptHMR-Vid does not report Accel in its tables; CalTennis notes it "performs best on RICH and EMDB on standard pose and translation metrics, but worst on acceleration metrics" and has worse foot sliding and stability — [CalTennis, App. A.4](https://arxiv.org/pdf/2606.20542)
- OnlineHMR's frequency analysis warns that smoothness metrics hide over-smoothing: on EMDB-1 GVHMR shows "better pattern reconstruction but slightly larger deviation in amplitude" (RMSE 75.52 vs TRAM 25.82) — [OnlineHMR, Table 6](https://arxiv.org/pdf/2603.17355)
- Volley3D shows the broadcast pattern for a known court: WHAM for pose, court homography from 65 keypoints for ground position of the feet, Butterworth low-pass on the ground-plane track, and a gravity parabola between detected takeoff and landing because the flat-homography assumption fails in jumps (depth-axis MAE 1.050 m -> 0.230 m over 50 jumps) — [Volley3D](https://arxiv.org/html/2610.38358)

### Inferences
- Because GVHMR's temporal network takes 2D keypoints, bounding boxes and per-frame image features as its inputs (its fork exposes swappable detectors/2D pose), substituting the project's own 2D keypoints is an engineering change rather than a research problem; GEM is similar. This was not verified in documentation.
- With extrinsics known, GVHMR's gravity-view output is already gravity-aligned; only yaw and translation need to be tied to the court frame, which the foot tracks give directly.
- 50 fps fast swings are exactly where learned temporal priors (trained mostly on 30 fps AMASS/BEDLAM motion) risk over-smoothing wrist and paddle-arm motion; OnlineHMR's amplitude finding for GVHMR is a warning sign to test on smashes and volleys.

### Gaps
- No source confirmed that any released model accepts a user-supplied full camera pose/ground plane, or user-supplied 2D keypoints via a documented interface.
- Frame-rate sensitivity (50 fps input vs 30 fps training) is not documented for any method in the sources read.

## 5. Physics-based or contact-aware refinement; practicality offline on a Mac

### Takeaway
Simulation-based cleanup (PHC/PHC+, PhysHMR, per-clip RL refinement) does remove floating and sliding but is tied to Isaac Gym / NVIDIA GPUs and is not practical on a Mac mini; the Mac-feasible route is optimization-based contact-aware refinement (soft foot-contact root optimization as in the SAM 3D Body retargeting pipeline, HTD-Refine-style global optimization, WHAM-style contact refinement), which runs on CPU/MPS in principle.

### Cited Findings
- PHC official repo requires Isaac Gym (main training/eval), installs `pytorch-cuda=11.6`, has sample Isaac Lab eval (Aug 2025), uses MuJoCo only for humanoid XML creation and visualization, needs SMPL v1.1.0 / SMPL-X v1.1 files, and includes a video-based control demo; no Mac/CPU mention — [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC)
- PhysHMR (SIGGRAPH Asia 2025) learns a visual-to-action policy controlling a simulated humanoid directly from monocular video, built on GVHMR and the PHC+ tracker, motivated by kinematic methods like GVHMR producing foot floating — [PhysHMR](https://arxiv.org/html/2510.02566v1)
- "The Last Mile to Production Readiness" (EA, Jan 2026): per-sequence deep-RL physics refinement of video capture targeting floating, interpenetration, phantom contact and foot sliding; "Floating and foot sliding are naturally mitigated through physics-based motion tracking"; stated limitation: a new policy must be trained for each motion clip — [arXiv 2601.19036](https://arxiv.org/html/2601.19036v1)
- Plug-and-play physical restoration (arXiv 2412.17377) uses GVHMR as the noisy source for PhysPT, PHC+ and its own method; in its tables GVHMR+PhysPT is often worse than plain GVHMR on some error metrics (search summary of the paper's tables) — [arXiv 2412.17377](https://arxiv.org/pdf/2412.17377)
- PhysPT (CVPR 2024, arXiv 2404.04430) is a transformer with physics-inspired losses that can be applied to kinematic estimates to improve plausibility and estimate forces; no official code repository was surfaced — [PhysPT](https://arxiv.org/abs/2404.04430)
- Contact-aware optimization without a simulator: the SAM 3D Body retargeting pipeline uses soft foot-contact probabilities from foot height/velocity and minimizes E_ground = lambda*L_slide + L_pen + L_contact + L_smooth + aux camera prior to recover the root trajectory; L_slide penalizes horizontal foot velocity during contact — [World-Coordinate Retargeting via SAM 3D Body](https://arxiv.org/pdf/2512.21573)
- HTD-Refine's optional foot-contact post-processing plus jerk/velocity/acceleration optimization cuts jitter by 55-75% but does not reliably reduce foot sliding (GVHMR 4.0 -> 5.7 mm on EMDB-2; 3.3 -> 3.3 on RICH) — [HTD-Refine](https://arxiv.org/html/2605.26879v1)
- WHAM's foot-contact refinement gives by far the lowest foot-velocity inconsistency on tennis (0.72 m/s vs 2.49 GVHMR, 3.23 PromptHMR, 4.40 GENMO, 6.65 TRAM) at the cost of much worse translation — [CalTennis](https://arxiv.org/pdf/2606.20542)
- CalTennis: ground-contact detection is one of three quantities that "remain unreliable across every model we tested" — [CalTennis](https://arxiv.org/pdf/2606.20542)
- Humanoid tracking controllers aimed at robots (GMT) ship MuJoCo simulation scripts, and a 2025 paper reports PULSE-X / PHC-X on both IsaacGym and IsaacLab; MaskedMimic inference runs through ProtoMotions with `+simulator=isaacgym` (search summary) — [GMT repo](https://github.com/zixuan417/humanoid-general-motion-tracking); [arXiv 2505.12278](https://arxiv.org/pdf/2505.12278); [MaskedMimic fork](https://github.com/DanielTruong99/MaskedMimic)

### Inferences
- With court-plane foot tracks already available, foot sliding can be attacked directly: detect stance phases, pin the stance foot to its court-plane track, and solve root translation plus leg IK. This is the same idea as the E_ground optimizer and Volley3D's homography grounding, and needs no NVIDIA hardware.
- Racket-sport footwork (split steps, lunges, slides on purpose, both feet airborne) violates the "one foot is always planted" assumption baked into simple contact heuristics; Volley3D needed an explicit takeoff/landing detector and ballistic interpolation for jumps.
- Simulation-based imitation tends to fail or damp high-dynamic motion and adds its own tracking error; for a visualization-grade digital twin, kinematic optimization is the lower-risk choice. If physics imitation is wanted later, it realistically means renting an NVIDIA GPU.

### Gaps
- No source confirmed any PHC-class tracker running on macOS (CPU MuJoCo/MJX or MLX). Not found is not the same as impossible; it is undocumented.
- PhysPT official code status and license could not be confirmed.
- No physics/contact refinement paper evaluated racket-sport footwork specifically.

## 6. Sports-specific work and datasets

### Takeaway
Racket-sport reconstruction from broadcast is an active 2026 topic, but existing pipelines use older HMR backbones (4DHumans in TT4D, WHAM in Volley3D) and ground players with court/table calibration, exactly the structure of this project. CalTennis is the one directly relevant benchmark and dataset; nothing was found for pickleball, padel or badminton body meshes.

### Cited Findings
- CalTennis: 11M+ frames (51 h), 40 players, 2-6 synchronized cameras, 60 Hz 1920x1080 phone video, automated calibration and sync from court geometry; dataset on Hugging Face (demalenk/caltennis) and project page; authors say they release dataset, capture recipe and evaluation code; introduces footwork (foot velocity/height) and stability metrics; multi-view disagreement is "a lower bound on error" — [CalTennis](https://arxiv.org/pdf/2606.20542)
- TT4D (table tennis, May 2026): 140+ h (146 h in Sec. 4.1) of 3D human meshes, ball trajectories and spin reconstructed from monocular broadcast video of 45,946 games; player meshes from 4DHumans aligned to the world frame via table-corner PnP calibration; moving-camera clips rejected (about 4.6%); racket-stroke reconstruction validated against mocap on 92 strokes: 26.4 +/- 4.4 degrees orientation error, 0.58 +/- 0.40 m/s velocity error; dataset release promised "upon acceptance", no URL; authors say the pipeline can extend to tennis or badminton — [TT4D](https://arxiv.org/html/2605.01234)
- Volley3D (Oct 2026, volleyball broadcast): WHAM (BEDLAM checkpoint, ViTPose-Base 2D joints) + 65-keypoint court homography (0.144 m projection error with PVTv2) + jump correction; no comparison of HMR backbones; notes occlusion from player overlap; annotations to be released — [Volley3D](https://arxiv.org/html/2610.38358)
- CourtSI-Bench (arXiv 2603.09896) builds on RacketVision, 1,672 professional net-sport clips across badminton, tennis and table tennis, and uses recovered human meshes to create spatial questions (search snippet) — [CourtSI-Bench](https://arxiv.org/pdf/2603.09896)
- Older tennis action data: THETIS (3D tennis shots, 2013) — [THETIS](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W08/papers/Gourgari_THETIS_Three_Dimensional_2013_CVPR_paper.pdf)

### Inferences
- The published sports pipelines confirm the architecture: kinematic HMR for local pose, calibration/homography for world placement, explicit handling for airborne phases, low-pass filtering of the ground track. None uses a 2025-2026 HMR backbone, so there is headroom from swapping in GVHMR/GEM/PromptHMR/3DB.
- CalTennis footage (phone cameras near the court) has larger players than a baseline broadcast view of the far side; its findings on scale sensitivity likely understate the difficulty for 80-150 px players.

### Gaps
- No paper or dataset on pickleball or padel 3D body reconstruction was found; no badminton mesh dataset was found (only 2D/doubles analysis work).
- CalTennis dataset license was not confirmed (paper is CC BY 4.0; dataset card not read).
- TT4D and Volley3D give no body-mesh accuracy numbers.

## 7. Code availability, license, compute, and Apple Silicon

### Takeaway
Every leading method is released as CUDA-first research code under non-commercial terms, and all SMPL-family methods also require separately licensed, registration-gated SMPL/SMPL-X files. The only documented Apple Silicon path is a third-party MIT-licensed GVHMR fork claiming end-to-end MPS support with a static-camera mode; GEM and PromptHMR document CUDA wheels only, and physics trackers need Isaac Gym.

### Cited Findings
- GVHMR (SIGGRAPH Asia 2024): official code + checkpoint at zju3dv/GVHMR (LICENSE file present; text not read); checkpoint trained on 2x RTX 4090; last news entry 2025-03-08 — [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR)
- GVHMR weights are described on a Hugging Face mirror as following "the original GVHMR license (non-commercial research)" (secondhand) — [ryanrudes/gvhmr on Hugging Face](https://huggingface.co/ryanrudes/gvhmr)
- `gvhmr` PyPI package v1.6.1 (released 2026-07-17, maintainer Ryan Rudes, MIT-licensed wrapper code, repo github.com/ryanrudes/gvhmr): claims "Apple-Silicon (MPS) support end-to-end" including a moderngl renderer; device auto-select CUDA -> MPS -> CPU, override via `GVHMR_DEVICE`; "Everything except CUDA-only DPVO runs on Apple Silicon"; SMPL-X required for motion recovery and SMPL for mesh rendering, both "registration-gated by MPI and not redistributed"; no speed numbers published (has a `gvhmr bench` command); the project was archived on 2026-08-06 and receives no further updates — [gvhmr on PyPI](https://pypi.org/project/gvhmr/)
- GEM / GENMO (NVIDIA, ICCV 2025 Highlight): code released Oct 2025, renamed GEM Dec 2025, GEM-SMPL checkpoint (`gem_smpl.ckpt`, Hugging Face nvidia/GEM-X) released Mar 2026; license "NVIDIA OneWay Noncommercial License"; install uses PyTorch cu124 wheels, Python 3.10; no VRAM figure; no Mac/MPS/CPU mention; webcam demo uses ONNX Runtime with a one-time ONNX export; README lists SMPL for GEM-SMPL while the training config says SMPL-X — [NVlabs/GENMO](https://github.com/NVlabs/GENMO)
- PromptHMR (CVPR 2025): inference code and PromptHMR-video checkpoints (trained on BEDLAM1 + BEDLAM2) released; "Due to licensing agreements, we currently do not plan to release the training code"; install scripts target torch 2.4.0+cu121 or 2.6.0+cu126; SMPL-X family models fetched by script, SMPL requires registration; depends on Detectron2, SAM2, DROID-SLAM, Metric3D, ViTPose, SPEC for the world-video pipeline (optional `--world-video=true`); no Mac/MPS mention; LICENSE file present but terms not read; the arXiv paper is CC BY-NC-SA 4.0 — [yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR); [PromptHMR arXiv](https://arxiv.org/abs/2504.06397)
- CameraHMR: "code and pGT are available for research purposes" (MPI project page) — [CameraHMR project page](https://camerahmr.is.tue.mpg.de)
- SAM-Body4D: code at github.com/gaomingqi/sam-body4d; reference hardware NVIDIA A100 80 GB; memory and runtime rise further with the occlusion refiner — [SAM-Body4D](https://arxiv.org/pdf/2512.08406)
- PHC: Isaac Gym required, CUDA PyTorch; SMPL + SMPL-X files needed — [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC)
- FactorizedHMR: project page only; paper commits only to releasing synthetic-data generation code; trained 36 h on 4x A5000 — [FactorizedHMR](https://arxiv.org/html/2605.14854)
- HTD-Refine: code "will be made publicly available"; PVA-Net inference 30+ FPS on an A6000 (about 300+ GFLOPs/frame); optimization about 2 min/video — [HTD-Refine](https://arxiv.org/html/2605.26879v1)
- OnlineHMR (CVPR 2026): project page tsukasane.github.io/Video-OnlineHMR; trained on an 80 GB H100; 3.3 FPS on RTX 6000 Ada — [OnlineHMR](https://arxiv.org/pdf/2603.17355)
- CalTennis ran all five baselines (PromptHMR, WHAM, GVHMR, TRAM, GENMO) on an NVIDIA H100 — [CalTennis](https://arxiv.org/pdf/2606.20542)

### Inferences
- Realistic Mac-mini candidates, in order of documented feasibility: (1) GVHMR via the MPS fork with `-s` and known intrinsics (lightweight temporal head, preprocessing is ViT feature extraction + 2D pose that PyTorch-MPS handles); (2) GEM-SMPL, likely portable since the static-camera path avoids SLAM and it already has an ONNX export, but untested on MPS; (3) PromptHMR image model with `--static_camera`, where the CUDA-bound parts (DROID-SLAM, Detectron2 builds) are avoidable in principle but porting is undocumented.
- The MPS fork being archived in Aug 2026 means it works as a frozen snapshot; expect to maintain it yourself.
- Licensing is uniformly non-commercial for weights (GVHMR, GEM, PromptHMR, CameraHMR) plus the SMPL/SMPL-X model licenses; this is fine for personal/research use and a blocker for a commercial product. SAM 3D Body's MHR-based stack is the main route that avoids SMPL licensing (covered by another researcher).
- 64 GB unified memory is ample for any of these inference models (the published peak figures are about 0.3-1 GB for temporal heads; ViT-H backbones are a few GB); throughput, not memory, is the constraint. No Apple Silicon throughput numbers exist in the sources; expect well below the 12 FPS that GVHMR reaches on an RTX 6000 Ada, times four players.

### Gaps
- Exact license texts for GVHMR, TRAM, WHAM, PromptHMR code, NLF, SMPLest-X, Multi-HMR and CameraHMR were not retrieved; "non-commercial" is confirmed only for GEM (license name), GVHMR weights (secondhand) and CameraHMR ("research purposes").
- No tested, first-hand reports of GVHMR, GEM, PromptHMR, TRAM, WHAM, NLF or SMPLest-X running on Apple Silicon (speed, correctness vs CUDA) were found; the MPS claim for GVHMR comes solely from the fork maintainer. No MLX ports were found.
- TRAM and WHAM repo status/requirements in 2026 were not checked (both depend on DROID-SLAM/DPVO for moving cameras, which a static camera makes unnecessary; unverified here).
- NLF and SMPLest-X compute needs and licenses not found in this pass.
