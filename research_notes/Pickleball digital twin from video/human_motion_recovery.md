# World-grounded 3D human motion recovery and physics-based cleanup for pickleball broadcast video

Research date: 2026-11-11. Items published after mid-2025 are tagged **[NEW, date]**.

Provenance convention used throughout:
- **(fetched)** = I opened the primary page (GitHub README, arXiv abstract/HTML) in this session.
- **(search)** = the claim comes from a search-result summary of the linked page; the page itself was not opened. Treat numbers as needing a check against the PDF.
- **(prior)** = from my background knowledge of the paper; the URL is the canonical paper/repo link but was not re-opened in this session.

---

## 1. What exactly is Meta's SAM 3D (SAM 3D Body / SAM 3D Objects)? Output format, video support, world frame, license, hardware, Apple Silicon, 2026 follow-ups

### Takeaway
SAM 3D Body is a **single-image**, promptable full-body (body + feet + hands) mesh recovery model that outputs Meta's **Momentum Human Rig (MHR)**, not SMPL; it has no temporal model and no world-frame trajectory of its own, so on video it is a strong per-frame pose estimator that needs a separate tracking, smoothing and global-trajectory layer. Its license (SAM License; MHR reported as Apache-2.0) is far friendlier than SMPL's, and it has been made to run on Apple Silicon via a CPU fallback for the MHR head, though MPS support is not merged upstream.

### Cited Findings
- The official repo's only dated news entry is 2025-11-19: "Checkpoints Launched, Dataset Released, Web Demo and Paper are out!" No 2026 news entries and no video version are listed in the README. **[NEW, 2025-11-19]** (fetched) — [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- The README describes the task as "single-image full-body 3D human mesh recovery (HMR)", covering body, feet and hands, using the Momentum Human Rig (MHR) parametric mesh; it accepts optional 2D keypoint and mask prompts and has a hand decoder for hand refinement. (fetched) — [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- Two checkpoints: DINOv3-H+ (840M params) and ViT-H (631M params). README-reported accuracy: DINOv3-H+ 3DPW MPJPE 54.8, EMDB MPJPE 61.7, RICH PVE 60.3; ViT-H 3DPW 54.8, EMDB 62.9, RICH 61.7. These are camera-frame per-image metrics, not world-frame. (fetched) — [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- Default pipeline dependencies: ViTDet human detector (optionally SAM 3 as detector via `--detector_name sam3`) and MoGe-2 as field-of-view estimator. Checkpoints are gated on Hugging Face (access request). (fetched) — [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- License: "checkpoints and code are licensed under the SAM License". (fetched) — [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body). Third-party summaries say the SAM License permits commercial use royalty-free with pass-through of the license text and use restrictions (trade control, weapons-related). (search) — [LibreYOLO HF mirror](https://huggingface.co/LibreYOLO/LibreSAM3DBodyh-mesh)
- The paper is on arXiv as 2602.15989 (i.e. a February 2026 arXiv posting of the November 2025 release); a third-party write-up labels it CVPR 2026. (search) — [arXiv 2602.15989](https://arxiv.org/pdf/2602.15989); [mer.vin write-up](https://mer.vin/2026/06/sam-3d-body-promptable-full-body-3d-mesh-from-one-image-cvpr-2026-mhr/)
- MHR: Meta's parametric body rig, paper arXiv 2511.15586; the paper says it is released under "a clear, industry-friendly license"; a mirror describes it as Apache-2.0. The MHR repo ships an SMPL/SMPL-X <-> MHR conversion tool (PyMomentum or PyTorch backends), which requires you to obtain the SMPL-X assets yourself. **[NEW, Nov 2025]** (search) — [MHR paper](https://arxiv.org/html/2511.15586v3); [MHR project page](https://facebookresearch.github.io/MHR)
- Apple Silicon: Issue #93 (opened 2025-12-15) reports `SAM3DBodyEstimator.process_one_image` crashing on MPS inside `MHRHead.mhr_forward` with "Placeholder storage has not been allocated on MPS device!" because the MHR TorchScript asset is not MPS-compatible. The reported workaround is to keep MHR on CPU while the rest of the model is on MPS; the reporter says this produces outputs. A fork commit "feat: add Apple Silicon (MPS) backend support" (referenced 2026-04-08) is linked but **not merged** into upstream main as of the page content. No Mac speed numbers are given. (fetched) — [sam-3d-body issue #93](https://github.com/facebookresearch/sam-3d-body/issues/93)
- SAM 3 itself (useful as the detector/tracker) has an open PR "Run inference on Apple Silicon (MPS) and CPU; fix SAM 3.1 start_session" and a Hugging Face thread about failing on M4 because of a Triton dependency. (search) — [sam3 PR #627](https://github.com/facebookresearch/sam3/pull/627); [HF discussion](https://huggingface.co/facebook/sam3/discussions/11)
- Follow-up 1, **SAM-Body4D [NEW, 2025-12-09]** (Gao, Miao, Han; arXiv 2512.08406): a training-free video wrapper. It notes SAM 3D Body run per frame is temporally inconsistent and degrades under occlusion; it generates identity-consistent masklets with a promptable video segmenter, refines them with an occlusion-aware module, and uses them to prompt SAM 3D Body; a padding-based parallel strategy gives multi-human inference. The abstract gives no quantitative results. (fetched) — [arXiv 2512.08406](https://arxiv.org/abs/2512.08406)
- SAM-Body4D code is MIT-licensed; it requires SAM 3, Diffusion-VAS (amodal segmentation + completion), SAM 3D Body (DINOv3), MoGe-2, Depth-Anything V2 ViT-L, detectron2, and PyTorch 2.7.1 with CUDA; no macOS/MPS mention; memory profile is in `assets/doc/resources.md` (not read). (fetched) — [gaomingqi/sam-body4d](https://github.com/gaomingqi/sam-body4d)
- Follow-up 2, **"World-Coordinate Human Motion Retargeting via SAM 3D Body" [NEW, 2025-12-25]** (Tu, Su, Zhu, Zheng; arXiv 2512.21573, cs.RO): uses SAM 3D Body as a frozen backbone; locks identity and skeleton-scale parameters per tracked subject; smooths by sliding-window optimization in MHR's low-dimensional latent space; recovers global root trajectories with a differentiable soft foot-ground contact model and contact-aware global optimization; retargets to Unitree G1. The abstract has no quantitative metrics and no code link. (fetched) — [arXiv 2512.21573](https://arxiv.org/abs/2512.21573)

### Inferences
- SAM 3D Body alone does not satisfy the "smooth, no foot sliding" requirement: it is per-frame. The two December 2025 follow-ups are essentially recipes for the missing layer (masklet prompting for identity/occlusion; fixed shape + windowed smoothing + contact-based root solve). Both recipes are reproducible without their code and fit a known-camera setup well.
- With a court-calibrated camera, the world placement of a SAM 3D Body result can be obtained without any learned global-trajectory model: intersect foot/ankle rays with the court plane and fix body scale per player. This sidesteps the fact that SAM 3D Body gives no world trajectory. (My inference; not from a source.)
- MHR (reported Apache-2.0) + SAM License is the cleanest licensing path found for a twin that might ever be shown publicly or commercialized; anything touching SMPL/SMPL-X inherits the non-commercial research license.
- 1.2B-parameter-class per-frame inference at 30/60 fps for 4 players for a whole match will be slow on MPS (MHR head on CPU). No measured Mac throughput was found; this must be benchmarked locally.

### Gaps
- **SAM 3D Objects**: I did not retrieve a primary source for it in this session, so no verified details are recorded here. It reconstructs objects (not humans) from a single image and is not directly relevant to player motion; whether it could usefully reconstruct a paddle is unknown.
- No official video version or dated 2026 update of SAM 3D Body was found. I could not confirm the CVPR 2026 venue from a primary source.
- Whether SAM 3D Body returns a per-person camera-space translation (it must, for rendering, but the README summary did not state it) was not verified.
- No measured inference speed or memory on Apple Silicon was found. PR #125 on the repo was not inspected.
- The MHR LICENSE file was not opened directly; "Apache-2.0" comes from a mirror description.

---

## 2. Leading world-grounded video methods (WHAM, TRAM, GVHMR, SLAHMR, PromptHMR, CoMotion, Human3R, GENMO/GEM, and 2026 work): accuracy, multi-person, static camera, known-camera input

### Takeaway
As of November 2026 the best published offline world-frame numbers are from DuoMo (Meta, March 2026; code status unconfirmed), then GENMO/GEM, PromptHMR-Vid and TRAM; **GVHMR remains the most practical baseline** because it has a static-camera flag, the lowest foot sliding of the released regression models, and working community Apple Silicon ports. For a fixed, court-calibrated broadcast camera the learned "global trajectory" component matters much less than on the moving-camera benchmarks, since camera pose, scale and ground plane are known.

### Cited Findings

**Benchmark table (single source, so numbers are mutually comparable).** DuoMo Table 4, world-space, 24 joints; columns WA-MPJPE(mm) / W-MPJPE(mm) / RTE(%) / Jitter / Foot-sliding(mm). EMDB uses TRAM's estimated camera motion; RICH is a static-camera dataset; ground-truth intrinsics used. (fetched) — [DuoMo arXiv 2603.03265](https://arxiv.org/html/2603.03265)

| Method | EMDB (moving cam) | RICH (static cam) |
|---|---|---|
| TRACE | 529.0 / 1702.3 / 17.7 / 2987.6 / 370.7 | 238.1 / 925.4 / 610.4 / 1578.6 / 230.7 |
| GLAMR | 280.8 / 726.6 / 11.4 / 46.3 / 20.7 | 129.4 / 236.2 / 3.8 / 49.7 / 18.1 |
| SLAHMR | 326.9 / 776.1 / 10.2 / 31.3 / 14.5 | 98.1 / 186.4 / 28.9 / 34.3 / 5.1 |
| WHAM | 135.6 / 354.8 / 6.0 / 22.5 / 4.4 | 109.9 / 184.6 / 4.1 / 19.7 / 3.3 |
| GVHMR | 111.0 / 276.5 / 2.0 / 16.7 / 3.5 | 78.8 / 126.3 / 2.4 / 12.8 / 3.0 |
| TRAM | 76.4 / 222.4 / 1.4 / 18.1 / 11.0 | n/a |
| GENMO | 74.3 / 202.1 / 1.2 / 17.8 / 8.8 | 75.3 / 118.6 / 1.9 / 15.0 / 6.7 |
| DuoMo | 67.6 / 171.6 / 1.2 / 8.7 / 3.7 | 53.6 / 80.8 / 1.3 / 5.3 / 3.1 |
| DuoMo (w/ height) | 66.0 / 167.1 / 1.1 / 8.7 / 3.7 | 53.5 / 80.4 / 1.3 / 5.2 / 3.1 |

- **DuoMo [NEW, 2026-03-03]** (Wang, Ng, Shin, Khirodkar, ..., Kitani, Richard, Prada, Zollhöfer; Meta Reality Labs / UPenn / CMU; arXiv 2603.03265; a third-party notes site files it under CVPR 2026): two diffusion models, a camera-space model whose output is lifted to world using given camera poses, then a world-space refiner; generates mesh vertices directly with no parametric body model; uses test-time guided sampling (2D reprojection and displacement guidance). Takes camera poses as input. Authors claim W-MPJPE 16% lower on EMDB and 30% lower on RICH than second best. The arXiv page links only a project page; code release not stated; multi-person not addressed. (fetched) — [arXiv 2603.03265](https://arxiv.org/html/2603.03265); venue hint (search): [papernotes](https://en.papernotes.org/CVPR2026/3d_vision/duomo_dual_motion_diffusion_for_world-space_human_reconstruction/)
- **GVHMR** (Shen et al., SIGGRAPH Asia 2024; arXiv 2409.06662; repo "About" also lists TPAMI 2026): predicts motion in a gravity-view coordinate frame. The README says "Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO"; a later news item says DPVO is no longer the default and a built-in "SimpleVO" is used. Released checkpoint trained on 2x4090. Acknowledges ViTPose and 4D-Humans (its 2D pose and image-feature front ends). README does not state license terms, multi-person support, or CPU/MPS support. (fetched) — [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR)
- GVHMR on Apple Silicon (community, not official): a macOS port PR reports GVHMR prediction on MPS matching CPU to about 1.1 mm of translation and needing a contiguous crop for HMR2 because MPS conv2d rejects a strided view; another fork states macOS works with a plain `uv sync` (MPS) and pins torch < 2.8; a third project logs that `--camera vggt` grew to 50.2 GB and caused a kernel panic on a Mac, and that DPVO is CUDA-only and optional. (search) — [fontanierh/atelier PR #199](https://github.com/fontanierh/atelier/pull/199); [ryanrudes/gvhmr AGENTS.md](https://github.com/ryanrudes/gvhmr/blob/main/AGENTS.md); [cmenguy/anim8te PR #12](https://github.com/cmenguy/anim8te/pull/12)
- **GENMO / GEM** (NVIDIA; arXiv 2505.01425, ICCV 2025 per prior knowledge): unified generative model for motion estimation and generation. The NVlabs/GENMO repo was renamed GEM (Dec 2025), code released Oct 2025 per one summary (the GEM-X README says the original GEM codebase was released 2025-05; these conflict), under the **NVIDIA OneWay Noncommercial License**. (search) — [NVlabs/GENMO](https://github.com/NVlabs/GENMO)
- **GEM-X [NEW; latest README news 2026-10-21]** (NVIDIA): "Monocular whole-body 3D human pose estimation using the SOMA body model", with Body Pose, Hand Pose (added 2026-10-21) and temporal Body Motion. Code Apache-2.0; weights on Hugging Face under their own terms (a third-party summary says NVIDIA Open Model License; the model card says trained on NVIDIA-owned/permissive data and "ready for commercial use"). Supports **calibrated cameras via camera adapters (pinhole, fisheye)** and **`--camera-mode static`** (moving camera via ViPE). YOLOX person detector (ONNX), 77-keypoint 2D observations, person tracking in Body Motion. Requires Python 3.10+, PyTorch 2.10+; README mentions CPU builds and ARM64 inference; rendering requires Linux x86-64; no MPS mention. README numbers: Body Pose 3DPW MPJPE 62.20 / PA 39.56 / PVE 72.89; EMDB-1 MPJPE 67.83 / PA 43.14; Body Motion EMDB-2 W-MPJPE100 267.20, WA-MPJPE100 112.76, RTE 2.45%, jitter 34.21, foot sliding 4.22. (fetched) — [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X); license detail (search): [nvidia/GEM-X model card](https://huggingface.co/nvidia/GEM-X/blob/main/README.md)
- **PromptHMR** (Wang et al., CVPR 2025; arXiv 2504.06397): promptable image HMR plus PromptHMR-Vid; repo releases the image model, video checkpoints (BEDLAM1/BEDLAM2-trained), and an optional multi-human world-coordinate video pipeline (`scripts/demo_video.py`, `--world-video=true`) depending on Detectron2, SAM2, DROID-SLAM, Metric3D, ViTPose, SPEC; outputs SMPL-X; PyTorch 2.4.0+cu121 / 2.6.0+cu126 builds; no training code; no macOS/MPS mention; license file not read. (fetched) — [yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR)
- PromptHMR-Vid EMDB-2 as listed on a leaderboard aggregator: WA-MPJPE 71, W-MPJPE 216.5, RTE 1.3, jitter 16.3, foot skating 3.5. Another paper notes PromptHMR uses GT focal length and DINOv2 with multimodal supervision, so comparisons with ViT-based methods are not like-for-like. (search) — [wizwand EMDB-2 leaderboard](https://www.wizwand.com/sota/global-human-motion-estimation-on-emdb-2-test); [alphaxiv summary](https://alphaxiv.org/overview/2504.06397v2)
- PromptHMR has been used in a racket-sport context: CourtSI ("Stepping VLMs onto the Court", arXiv 2603.09896 **[NEW, March 2026]**) uses PromptHMR to recover SMPL-X meshes in camera coordinates for badminton/tennis/table-tennis broadcast views. (search) — [arXiv 2603.09896](https://www.alphaxiv.org/abs/2603.09896)
- **TRAM** (Wang et al., ECCV 2024; arXiv 2403.17346): masked DROID-SLAM with metric depth for camera, VIMO video transformer for body. Repo is MIT-licensed with third-party models under their own licenses. Per the Human3R paper's comparison, TRAM supports multi-person but runs offline at 0.86 FPS. Note its foot sliding (11.0 mm on EMDB) is 3x GVHMR's in the table above. (search) — [yufu-wang/tram](https://github.com/yufu-wang/tram); [Human3R paper](https://arxiv.org/pdf/2510.06219)
- **WHAM** (Shin et al., CVPR 2024; arXiv 2312.07531): RNN-based, online-capable, predicts foot-ground contact probabilities and uses them in a trajectory refiner; EMDB/RICH numbers in table above. (prior for description; numbers fetched from DuoMo table) — [arXiv 2312.07531](https://arxiv.org/abs/2312.07531)
- **SLAHMR** (Ye et al., CVPR 2023; arXiv 2302.12827): optimization-based, multi-person, uses the HuMoR motion prior and solves for a shared ground plane and scale; slow (optimization over the whole clip). Numbers in table above. (prior for description) — [arXiv 2302.12827](https://arxiv.org/abs/2302.12827)
- **CoMotion** (Apple; Newell, Hu, Lipson, Richter, Koltun; ICLR 2025): online multi-person detection + tracking of SMPL poses from one monocular camera, "enables online tracking through occlusion"; outputs SMPL pose `.pt`, MOT-format track boxes, rendered video. **Ships a pre-compiled Core ML version of the detection stage for macOS** with "significant speedups". Code and weights are under separate Apple licenses (LICENSE.md, LICENSE_MODEL.md; terms not read). Requires SMPL neutral model download. (fetched) — [apple/ml-comotion](https://github.com/apple/ml-comotion)
- **Human3R [NEW, October 2025]** (arXiv 2510.06219, "Everyone Everywhere All at Once"): online feed-forward model built on a 3R-style reconstruction backbone that jointly outputs multi-person SMPL-X, scene and camera; the lightweight variant runs 13.59 to 15.87 FPS. Aggregator-listed EMDB-2 numbers: WA-MPJPE 112.2 (online category) or 118.2 / W-MPJPE 286.1 / RTE 2.4 in another setting. (search) — [arXiv 2510.06219](https://arxiv.org/pdf/2510.06219); [wizwand EMDB-2](https://www.wizwand.com/sota/human-motion-estimation-in-world-coordinates-on-emdb-2-24-joints-test)
- **OnlineHMR [NEW, March 2026; CVPR 2026]** (arXiv 2603.17355): causal/online world-grounded HMR; EMDB-2 WA-MPJPE 93.5, RTE 2.2%, W-MPJPE 310.4 (trails Human3R's 267.9, attributed to scale drift). Online methods are not needed for an offline pipeline. (search) — [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_OnlineHMR_Video-based_Online_World-Grounded_Human_Mesh_Recovery_CVPR_2026_paper.html); [arXiv 2603.17355](https://arxiv.org/html/2603.17355v1)
- **WATCH [NEW, September 2025]** (arXiv 2509.04600, "World-aware Allied Trajectory and pose reConstruction for Camera and Human") appeared as another world-frame method; details not read. (search) — [arXiv 2509.04600](https://arxiv.org/pdf/2509.04600)
- **FlowHMR [NEW, 2026]**: video-conditioned generative mocap post-trained with RL-style rewards (video fidelity + trackability by a physics controller); introduces the Wild-4K dataset (~4K internet videos). Details from an aggregator only. (search) — [deeplearn.org listing](https://deeplearn.org/arxiv/838607/flowhmr:-physically-plausible-motion-capture-from-video)
- **CalTennis [NEW, 2026-06-18]** (Demler, Xie, Werner, Szczuka, Perona; arXiv 2606.20542): 11M+ frames (51 h) of tennis from 40 players, 2 to 6 synchronized cameras at 60 Hz, used to benchmark monocular-to-3D methods. Abstract conclusion: "3D joint angle recovery is now fairly accurate, but all models struggle with depth and foot contact"; proposes footwork and stability metrics and notes body-shape inconsistency. (fetched, abstract only) — [arXiv 2606.20542](https://arxiv.org/abs/2606.20542)

### Inferences
- **Known camera makes most of the "world-grounded" research problem disappear.** EMDB errors are dominated by moving-camera estimation and metric scale. Here the camera is static and calibrated to a court with known dimensions, so the world frame, gravity direction, ground plane and metric scale are given. RICH (static cameras) is the more relevant benchmark column, and on it GVHMR's 126 mm W-MPJPE / 3.0 mm foot sliding is already reasonable; DuoMo's 80.8 mm is the best seen.
- Methods that explicitly accept a known/static camera: GVHMR (`-s`, static only; intrinsics can be supplied in its pipeline per prior knowledge), GEM-X (static mode + calibrated-camera adapters, verified), DuoMo (takes camera poses; code unconfirmed), SLAHMR-style optimization (any camera can be fixed by construction). TRAM, PromptHMR-world and Human3R estimate their own cameras via SLAM/3R; using them with a known camera means discarding their camera branch and re-posing camera-frame output with the calibrated extrinsics.
- **CalTennis's finding (depth and foot contact are the weak points of every model) is exactly what court calibration fixes**: depth along the camera axis is the worst-observed quantity from an elevated behind-baseline view, and it can be replaced by ray/ground-plane intersection at the feet plus a fixed per-player height.
- Multi-person: none of GVHMR/WHAM/GENMO are jointly multi-person; they run per track. That is fine for pickleball (2 to 4 tracked, mostly non-interacting players). PromptHMR, Human3R, CoMotion and SAM-Body4D are natively multi-person.
- Small/distant far-side players: no source quantified performance vs. person pixel height. Top-down methods that crop each person (GVHMR/HMR2, SAM 3D Body, PromptHMR with box prompts, NLF) can be fed upscaled crops, whereas whole-image feed-forward methods (Human3R, Multi-HMR) lose resolution on small subjects. This is an inference to be tested, not a sourced fact.
- Practical ranking for this project (inference): (1) GVHMR static mode as the first working baseline on the Mac (community MPS ports exist); (2) SAM 3D Body per-frame with mask/keypoint prompts as a higher-fidelity pose source including hands, fused over time by custom optimization; (3) GEM-X as the commercially clean temporal model with calibrated static-camera support, if it runs on CPU/MPS, noting its own README numbers (jitter 34.21, WA-MPJPE100 112.76) are weaker than GENMO's SMPL research model; (4) DuoMo if code appears.

### Gaps
- GVHMR and WHAM license terms were not verified (README did not show them).
- GENMO's original paper numbers (venue, EMDB table from its own paper) were not fetched; values above are as re-reported by DuoMo. One aggregator lists different GENMO numbers (WA 65.1 / W 210.9 / jitter 9.58), which conflicts with DuoMo's table; protocols differ. — [wizwand](https://www.wizwand.com/sota/world-space-human-motion-recovery-on-emdb)
- DuoMo code/weights availability and venue are unconfirmed from primary sources.
- No evidence found of TRAM, PromptHMR, Human3R, GEM-X, or DuoMo running on MPS. PromptHMR's world pipeline depends on DROID-SLAM (CUDA extensions), which is unnecessary for a static camera but entangled in the demo script.
- CalTennis per-method rankings (which of GVHMR/PromptHMR/SAM 3D Body did best on tennis) were not accessible from the abstract; the full paper should be read since it is the closest existing benchmark to this use case.
- No benchmark numbers found for any method specifically on small/far subjects or behind-net occlusion.

---

## 3. 2D/3D pose foundations as inputs or sanity checks, and Core ML / MLX / MPS support

### Takeaway
ViTPose (Apache-2.0) and the RTMPose/RTMW family via `rtmlib` (Apache-2.0, ONNX, no mmcv) are the permissive, Mac-friendly 2D keypoint choices; Sapiens is non-commercial and heavy. Apple's CoMotion is the only model found with an official Core ML artifact. I found no official MLX or Core ML ports of ViTPose, RTMW, Sapiens, NLF or Multi-HMR.

### Cited Findings
- `rtmlib` packages RTMPose, DWPose, RTMO and RTMW "without mmcv, mmpose, mmdet", under Apache-2.0. (search) — [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib)
- ViTPose / ViTPose++ official repo is Apache-2.0. (search) — [ViTAE-Transformer/ViTPose](https://github.com/vitae-transformer/vitpose)
- A 2026 practitioner guide notes that for the MMPose family "the catch in 2026 is maintenance, not licensing", the last MMPose release being July 2024. (search) — [Fora Soft guide](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/openpose-mediapipe-rtmpose-pose-tracking)
- Sapiens (Meta, ECCV 2024; arXiv 2408.12569) is non-commercial: the paper header says CC BY-NC-SA 4.0 while a 2026 guide says CC BY-NC 4.0 (sources conflict on the exact variant). **Sapiens2 [NEW, April 2026]** (arXiv 2604.21681) exists; a search summary lists it as CC BY 4.0, which may refer to the arXiv paper license rather than the model license. (search) — [Sapiens paper](https://arxiv.org/html/2408.12569v1); [Sapiens2](https://arxiv.org/html/2604.21681v1)
- GVHMR's front end uses ViTPose and 4D-Humans (HMR2.0) features (acknowledged in README), and a community port has HMR2 running on MPS with a contiguous-crop fix. (fetched / search) — [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR); [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- GEM-X uses a YOLOX ONNX detector and 77-point 2D keypoints, with ViTPose used in its evaluation. (fetched) — [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)
- CoMotion provides a pre-compiled Core ML detection stage for macOS. (fetched) — [apple/ml-comotion](https://github.com/apple/ml-comotion)
- RacketVision's baseline for racket keypoints is RTMPose, showing the RTM family is retrainable for sport-specific keypoints. (search) — [RacketVision](https://arxiv.org/html/2511.17045v3)
- AthletePose3D (arXiv 2503.07499, CVPR 2025 workshop era) reports that models trained on conventional datasets perform poorly on athletic motions. (search) — [arXiv 2503.07499](https://pdf.arxiv.org/abs/2503.07499)
- 4D-Humans / HMR2.0 (Goel et al., ICCV 2023; arXiv 2305.20091), Multi-HMR (Baradel et al., ECCV 2024; arXiv 2402.14654; single-shot multi-person SMPL-X) and NLF (Sárándi & Pons-Moll, NeurIPS 2024; arXiv 2407.07532; per-frame, can output arbitrary skeleton/mesh points, widely used as a robust per-frame 3D pose source) are the main per-frame 3D alternatives. (prior; not re-opened) — [4D-Humans](https://arxiv.org/abs/2305.20091); [Multi-HMR](https://arxiv.org/abs/2402.14654); [NLF](https://arxiv.org/abs/2407.07532)

### Inferences
- ONNX models (rtmlib RTMW/RTMPose, YOLOX) can run on Mac through ONNX Runtime's CoreML execution provider or CPU; this is the lowest-friction route to accelerated 2D keypoints on Apple Silicon. (Inference from the ONNX packaging; not verified by a source in this session.)
- 2D keypoints matter beyond being model inputs: with a calibrated camera they are the reprojection target for any optimization-based cleanup, and whole-body models (RTMW, DWPose, 133 keypoints) include feet and hands, which are needed for foot-contact detection and wrist/paddle orientation.
- Given the AthletePose3D finding and CalTennis's, expect fast swings and lunges to be where kinematic estimators are weakest; a per-frame second opinion (e.g. NLF or SAM 3D Body vs GVHMR) is a useful disagreement detector.

### Gaps
- No verified Core ML / MLX ports of ViTPose, RTMW, Sapiens, NLF, or Multi-HMR were found; no MPS compatibility reports for NLF or Multi-HMR were found.
- Licenses for NLF, Multi-HMR and 4D-Humans weights were not verified in this session.
- Sapiens2's actual model license needs checking on its repo.

---

## 4. Enforcing physical plausibility after the fact: physics imitation, optimization, priors, and what works without an Nvidia GPU

### Takeaway
Published head-to-head evidence (PhysHMR, SIGGRAPH Asia 2025) shows physics trackers do cut floating/penetration but that naively running kinematic output through PHC+ **increases foot sliding and position error**, and PhysPT wrecks global trajectory; almost all learned physics trackers are Isaac Gym / CUDA-bound. For a Mac with a known ground plane, contact-aware trajectory optimization (the approach used by Vid2Player3D's GLAMR-based root solve and by the SAM 3D Body world-coordinate paper) is the practical first line, with MuJoCo CPU-based tracking as an optional second stage.

### Cited Findings
- **PhysHMR [NEW, 2025-10-02; SIGGRAPH Asia 2025]** (Feng, Huang, Wang, Gu, Liu; UPenn; arXiv 2510.02566): visual-to-action policy controlling a simulated humanoid directly from video, using GVHMR encoder features, a "pixel-as-ray" module lifting 2D keypoints to global rays, and PPO + distillation from a mocap-trained PHC+ teacher. Simulator: **NVIDIA Isaac Gym** (60 Hz physics, 30 Hz control). No code link in the paper. One humanoid per video. (fetched) — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysHMR EMDB2 table (PA-MPJPE / WA-MPJPE / MPJPE / foot sliding FS / foot height variance HV; mm): TRAM 35.51 / 148.05 / 56.74 / 11.76 / 22.97; GVHMR 40.95 / 228.67 / 65.21 / 5.65 / 26.42; PhysPT(CLIFF) 48.40 / 762.78 / 77.00 / 11.02 / 6.54; TRAM x PhysPT 39.90 / 704.57 / 61.42 / 8.49 / 7.02; GVHMR x PhysPT 41.34 / 682.03 / 66.08 / 10.71 / 8.46; TRAM x PHC+ 52.94 / 158.58 / 74.34 / 23.41 / 7.64; GVHMR x PHC+ 46.24 / 193.01 / 72.50 / 12.71 / 7.71; PhysHMR 39.34 / 189.26 / 55.48 / 4.60 / 5.04. (fetched) — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- Reading of that table (the paper's own data): two-stage "kinematic then PHC+ tracking" cuts foot-height variance about 3x (26.42 -> 7.71 for GVHMR) but more than doubles foot sliding (5.65 -> 12.71) and worsens MPJPE (65.21 -> 72.50); PhysPT post-processing roughly triples WA-MPJPE. (fetched) — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysHMR's stated limitations: real-to-sim gap in body mechanics and contact causes visible artifacts; no human-scene interaction; deterministic policy. (fetched) — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- **PHC in MuJoCo**: `ZhengyiLuo/PHC_MJX` is the "Official Implementation of ... Perpetual Humanoid Control ... in MuJoCo", marked "[Repo still under construction]", depending on SMPLSim; the older UHC (MuJoCo) repo points to it. (search) — [PHC_MJX](https://github.com/ZhengyiLuo/PHC_MJX); [UHC](https://github.com/ZhengyiLuo/UHC)
- A student project issue reports a working SMPL humanoid setup on macOS Apple Silicon with Python 3.12, MuJoCo 3.x and SMPLSim (early-stage, not a PHC guide). (search) — [CSID-DGU issue](https://github.com/CSID-DGU/2026-2-CECD1-1-JJJN-01/issues/1)
- PHC and ProtoMotions both require source motion in SMPL format. (search) — [Retargeting Matters, arXiv 2510.02252](https://arxiv.org/html/2510.02252v1)
- **Vid2Player3D** (SIGGRAPH 2023, NVIDIA) is the canonical "broadcast video -> physics-corrected motion" precedent: kinematic estimates are corrected by a low-level imitation policy (adapted from EmbodiedPose) trained first on AMASS then fine-tuned on the tennis motions, in **Isaac Gym Preview 4**; the root trajectory comes from a GLAMR-based optimization with 2D foot-contact detection and court line calibration. Trained models are not released ("license issues of the video footage"). (fetched) — [nv-tlabs/vid2player3d](https://github.com/nv-tlabs/vid2player3d)
- **MultiPhys** (CVPR 2024; arXiv 2404.11987): physics simulator in the loop for multi-person motion; reports skating reduced by 35% and 137% on Hi4D and ExPI. (search) — [arXiv 2404.11987](https://arxiv.org/html/2404.11987v1)
- Trajectory-optimization lineage: Gärtner et al. 2022 (trajectory optimization with a full physics engine from monocular video; arXiv 2205.12292) and DiffPhy (differentiable physics; arXiv 2205.12256). (search) — [arXiv 2205.12292](https://arxiv.org/abs/2205.12292v1); [DiffPhy](https://arxiv.org/pdf/2205.12256v1.pdf)
- **MonoMSK [NEW, November 2025]** (arXiv 2511.19326): biomechanics-oriented, transformer inverse dynamics + differentiable forward kinematics + ODE simulation, recovers forces. (search) — [arXiv 2511.19326](https://arxiv.org/abs/2511.19326v1)
- The SAM 3D Body world-coordinate paper **[NEW, 2025-12-25]** is a concrete optimization-only recipe: lock shape/scale, sliding-window smoothing in latent pose space, differentiable soft foot-ground contact, contact-aware global root optimization. (fetched) — [arXiv 2512.21573](https://arxiv.org/abs/2512.21573)
- DuoMo **[NEW, 2026-03-03]** shows a learned world-space diffusion refiner can reach jitter 5.3 and foot sliding 3.1 mm on static-camera RICH with test-time 2D reprojection guidance, i.e. diffusion refinement competes with physics on smoothness metrics. (fetched) — [arXiv 2603.03265](https://arxiv.org/html/2603.03265)
- Background methods (prior knowledge; canonical links, not re-opened this session): PhysPT (CVPR 2024, physics-aware pretrained transformer, no simulator; [arXiv 2404.04430](https://arxiv.org/abs/2404.04430)); PHC (ICCV 2023; [arXiv 2305.06456](https://arxiv.org/abs/2305.06456)); PULSE (ICLR 2024, latent skill space distilled from PHC+; [arXiv 2310.04582](https://arxiv.org/abs/2310.04582)); MaskedMimic (SIGGRAPH Asia 2024, masked-goal physics controller; [arXiv 2409.14393](https://arxiv.org/abs/2409.14393)); SimPoE (CVPR 2021, simulation-based pose estimation in MuJoCo; [arXiv 2104.00683](https://arxiv.org/abs/2104.00683)); HuMoR (ICCV 2021, CVAE motion prior with contact and ground-plane optimization; [arXiv 2105.04668](https://arxiv.org/abs/2105.04668)); RoHM (CVPR 2024, diffusion-based robust motion denoising/infilling under occlusion; [arXiv 2401.08570](https://arxiv.org/abs/2401.08570)); SmoothNet (ECCV 2022, plug-in temporal smoother; [arXiv 2112.13715](https://arxiv.org/abs/2112.13715)); WHAM's contact head and trajectory refiner ([arXiv 2312.07531](https://arxiv.org/abs/2312.07531)).

### Inferences
- **The user's priority (smoothness and grounding over per-frame accuracy) is best served by optimization against known geometry, not by a learned physics tracker.** Evidence: (a) PhysHMR's table shows off-the-shelf PHC+ tracking makes foot sliding worse; (b) Isaac Gym is Linux + Nvidia only (Vid2Player3D, PhysHMR, PHC, PULSE, MaskedMimic all target it), so none run on the Mac as shipped; (c) Vid2Player3D and the SAM 3D Body retargeting paper both solve the root trajectory by contact-aware optimization before any physics.
- A concrete Mac-feasible cleanup stack (my synthesis, PyTorch CPU/MPS or SciPy):
  1. Fix body shape/scale per player for the whole match (kills limb-length flicker).
  2. Detect foot contacts (foot keypoint velocity + height, or WHAM/GVHMR-style contact predictions).
  3. Solve per rally a trajectory optimization over root translation, root orientation and joint rotations with terms: 2D reprojection through the calibrated camera (confidence-weighted), stay-near-network-pose prior, acceleration/jerk penalty or B-spline parameterization, foot height >= 0 and == 0 in contact, zero foot velocity in contact, joint limits.
  4. Final IK pass to pin stance feet exactly (standard foot-lock), then light low-pass on non-contact DOFs.
  5. Fill occluded spans (far players behind the net band, player crossings) by spline/prior infilling rather than trusting per-frame output.
- MuJoCo itself is CPU-native on macOS arm64 and a single SMPL-like humanoid simulates far faster than real time on CPU, so (i) inverse-dynamics residual checks and (ii) sampling-based or PD tracking of the cleaned reference in MuJoCo are feasible on the M5 Pro; training a PHC-class universal RL tracker locally is not realistic without CUDA, though inference of a pre-trained policy in MuJoCo CPU would be. MJX (JAX) on Apple Metal was experimental as of my knowledge and should be assumed CPU-only. (Background knowledge; not verified this session.)
- If physics tracking is ever added, the PhysHMR result suggests it should be judged on foot sliding and trajectory drift, not only on penetration/float, and blended (e.g. keep optimized root, take contact timing/vertical dynamics from sim).
- Pickleball-specific simplification: play is on a flat rigid court, with no scene interaction other than feet (and rare falls), which is the best case for hard ground-plane constraints.

### Gaps
- No source found demonstrating PHC/PHC+/PULSE/MaskedMimic inference on macOS, or ProtoMotions/mjlab on Apple Silicon (the search returned nothing on these two).
- PhysHMR code availability unknown. FlowHMR details unverified (aggregator only).
- I did not find a 2025 to 2026 benchmark comparing optimization-based cleanup vs. diffusion refiners (RoHM-like) vs. physics trackers on the same static-camera data; the PhysHMR table is the closest and omits jitter.
- SmoothNet / HuMoR / RoHM license and MPS status not checked (HuMoR and RoHM depend on SMPL-family models, so the SMPL license applies).

---

## 5. Sports-specific work: 3D reconstruction of racket-sport players from broadcast

### Takeaway
No pickleball-specific 3D reconstruction paper or dataset was found. The closest precedents are Vid2Player3D (tennis, SIGGRAPH 2023), TT4D (table tennis, May 2026) and CalTennis (June 2026), and their shared lessons are: calibrate from court lines, solve the root trajectory with foot contacts on the known plane, and expect depth and foot contact to be the failure modes.

### Cited Findings
- **Vid2Player3D** ("Learning Physically Simulated Tennis Skills from Broadcast Videos", Zhang, Yuan, Makoviychuk, Guo, Fidler, Peng, Fatahalian; SIGGRAPH 2023). Annotation pipeline per the official repo: YOLOv4 player detection/tracking; ViTPose 2D keypoints; HybrIK 3D pose and mesh; 2D foot-contact detection; GLAMR-based global root trajectory optimization; tennis court line detection for calibration. Physics correction via an imitation policy adapted from EmbodiedPose in Isaac Gym, trained on AMASS then on the tennis motions. Trained models withheld due to footage licensing. (fetched) — [nv-tlabs/vid2player3d](https://github.com/nv-tlabs/vid2player3d); [paper PDF](https://research.nvidia.com/labs/toronto-ai/vid2player3d/data/tennis_skills_main.pdf)
- A community reimplementation swaps HybrIK for GVHMR and keeps Vid2Player3D's global-trajectory correction/projection stage. (search) — [maminsed/vid3player3d](https://github.com/maminsed/vid3player3d)
- **TT4D [NEW, 2026-05-02]** (Rahmanian, Kienzle, Gossard, Kalaria, Lienhart, Sastry; arXiv 2605.01234): pipeline + dataset reconstructing 140+ hours (146 h; 211,534 points) of table tennis singles and doubles from monocular broadcast video, sourced from 45,946 broadcast games (2021 to 2024); points are split at scoreboard changes and filtered by camera calibration and ball-trajectory consistency; "lift-first" approach lifts the whole 2D ball track to 3D (with spin) before temporal segmentation; includes 3D human meshes over time; validated by estimating racket pose and velocity at impact and by training a generative rally model. Compared against LATTE-MV (1,017 games, singles only, 26 h). (search; full PDF not read) — [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- **TT3D** (arXiv 2504.10035, April 2025): table tennis 3D reconstruction from broadcast that adapted an existing 3D pose model lacking depth motion to track player movement. (search) — [arXiv 2504.10035](https://arxiv.org/abs/2504.10035v1)
- **CalTennis [NEW, 2026-06-18]**: multi-view tennis benchmark; "all models struggle with depth and foot contact"; proposes footwork and stability metrics. (fetched) — [arXiv 2606.20542](https://arxiv.org/abs/2606.20542)
- **CourtSI [NEW, March 2026]** (arXiv 2603.09896): reconstructs badminton/tennis/table tennis broadcast scenes for a VLM spatial benchmark, using PromptHMR for SMPL-X meshes in camera coordinates, manual ball annotation, and RacketVision source video. (search) — [arXiv 2603.09896](https://www.alphaxiv.org/abs/2603.09896)
- **WorldPose** (ETH/FIFA/Microsoft; arXiv 2501.02771; ECCV 2024 per prior knowledge): multi-person global 3D pose dataset from the 2022 FIFA World Cup, 80+ sequences and about 2.5M 3D poses; a benchmark for distant small players from broadcast-like views. (search) — [arXiv 2501.02771](https://arxiv.org/html/2501.02771v2)
- **SportsPose** (arXiv 2304.01865, 2023): markerless 3D sports pose dataset covering soccer, volleyball, jump, baseball pitch and tennis. (search) — [arXiv 2304.01865](https://arxiv.org/pdf/2304.01865)
- **AthletePose3D** (arXiv 2503.07499): athletic-motion benchmark; conventional-dataset models perform poorly on athletic motion. (search) — [arXiv 2503.07499](https://pdf.arxiv.org/abs/2503.07499)
- **MonoTrack** (CVPRW CVSports 2022; arXiv 2204.01899): badminton shuttle 3D trajectory from monocular broadcast using court and player-pose cues. (search) — [arXiv 2204.01899](https://arxiv.org/pdf/2204.01899)
- Pickleball: the only pickleball-labelled resource surfaced was a 2D player-detection dataset on Roboflow Universe with no description. (search) — [Roboflow PlayerDetection](https://universe.roboflow.com/pickleball-dataset/playerdetection-ynnr6)

### Inferences
- Pickleball is geometrically closer to tennis doubles on a badminton-sized court (13.4 m x 6.1 m court; background knowledge), so far-side players are closer to the camera and larger in frame than far-side tennis players at equivalent camera placement; Vid2Player3D's success on the harder tennis case is encouraging. Its quality bar (plausible, simulated motion with acknowledged wrist/racket inaccuracy) came only after physics correction, consistent with the user's priorities.
- TT4D shows match-scale (100+ hours) automated broadcast reconstruction is now done routinely offline, with rally segmentation and calibration-consistency filtering as key engineering steps. Its human-mesh method should be read from the PDF before choosing a backbone.
- The far-side net occlusion (shins/feet hidden behind net tape/band) is specific to behind-baseline views; none of the sources discuss it directly. SAM-Body4D's amodal completion is the only found method explicitly targeting occluded-region recovery for mesh prompting.

### Gaps
- No pickleball 3D paper/dataset found (searched "pickleball 3D pose player reconstruction paper dataset arXiv 2025 2026").
- "SMART" (named in the brief) was not found in any search result; I cannot say what it refers to.
- Vid2Player3D's quantitative quality numbers and exact handling of the wrist/racket were not retrieved (the PDF exceeded the fetch size limit).
- TT4D and CalTennis full-paper details (which HMR backbone, per-method errors) not read.

---

## 6. Racket/paddle 6-DoF and hand-object contact from monocular sports video

### Takeaway
There is no off-the-shelf monocular broadcast paddle 6-DoF estimator. The available building blocks are 2D racket keypoints (RacketVision), a table-tennis monocular 6D racket pose network, and impact-time racket pose inferred from ball/hand constraints (TT4D, MonoTrack); a pickleball paddle would need a custom keypoint model plus PnP against the known paddle shape, constrained to the wrist.

### Cited Findings
- **RacketVision [NEW, November 2025; AAAI 2026 proceedings link]** (arXiv 2511.17045): 1,672 clips (435,179 frames, 12,755 s) across badminton, tennis and table tennis; racket annotated with a bounding box and five keypoints per racket type; baseline is 2D RTMPose; tennis reaches PCK@0.2 of 89.69% under multi-sport training. These are 2D annotations, not 6-DoF. (search) — [arXiv 2511.17045](https://arxiv.org/html/2511.17045v3); [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/download/37362/41324)
- "Geometry-Aware Table Tennis Racket Pose Estimation via Multi-View Consistency and Racket Priors": a monocular 6D racket pose network trained with multi-view consistency (ground truth from four synchronized cameras), reported to beat FoundationPose and Any6D baselines on a curated table tennis dataset. Publication date not confirmed. (search) — [ACM DOI](https://doi.org/10.1145/3799825.3818754)
- TT4D validates its reconstructions by estimating racket pose and velocity at impact. **[NEW, 2026-05-02]** (search) — [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- MonoTrack estimates racket pose and velocity at impact in badminton. (search) — [MonoTrack](https://cs.stanford.edu/people/paulliu/files/cvpr-2022.pdf)
- SAM 3D Body includes hands (hand decoder, FreiHAND PA-MPJPE 5.5 mm), and GEM-X added Hand Pose on 2026-10-21, so wrist/hand orientation is available from the body model side. (fetched) — [sam-3d-body](https://github.com/facebookresearch/sam-3d-body); [GEM-X](https://github.com/NVlabs/GEM-X)

### Inferences
- For a visually convincing twin, attaching a rigid paddle to the hand with a fixed grip transform and letting wrist orientation drive it (as Vid2Player3D effectively did by simulating a racket rigidly attached to the hand; background knowledge) is the pragmatic baseline. At broadcast resolution, far-side hands are a few pixels, so hand-pose output there will be mostly prior.
- A refinement path: fine-tune an RTMPose-style model on a few hundred labelled pickleball paddle keypoints (RacketVision as pretraining), solve PnP with the known paddle outline and the calibrated camera, and regularize with the wrist position and ball-contact events (at impact the paddle face must intersect the reconstructed ball trajectory and its normal/velocity must be consistent with the outgoing ball).
- Generic 6D object pose models (FoundationPose, Any6D) were outperformed by a racket-specific model in the table tennis paper, suggesting thin, fast, motion-blurred paddles are hard for generic estimators.

### Gaps
- No paper found on monocular hand-paddle contact or grip estimation in sports broadcast.
- No pickleball paddle dataset found.
- Date/venue of the table tennis 6D racket paper unverified.

---

## 7. Multi-person identity tracking through occlusion and side switches

### Takeaway
SAM-family mask propagation combined with a sports tracker is the current top performer on SportsMOT (about 87 HOTA vs 64 for ByteTrack), but for pickleball the court calibration gives a much simpler and more robust identity signal (ground-plane position, court side, team/partner), so a light tracker plus court-space association should suffice.

### Cited Findings
- **"Selective Mask Propagation for Multi-Object Tracking" [NEW, June 2026]** (arXiv 2606.13033): pairs SAM 3 mask propagation with the Deep-EIoU tracker; SportsMOT test table reports 87.2 HOTA for SAM3-Deep-EIoU vs 64.1 for ByteTrack (the abstract cites 86.8 HOTA for a variant with global track association; the two figures differ by configuration). It distinguishes on-screen identity preservation through occlusion (which it targets) from off-screen re-identification. Preprint. (search) — [arXiv 2606.13033](https://arxiv.org/abs/2606.13033)
- **SAMIDARE [NEW, April 2026]** (arXiv 2604.22162): SAM 2-based tracking-by-segmentation with SportsMOT gains under frequent occlusion and frame-out events. **McByte++ [NEW, 2026-08-16]** (arXiv 2608.15688): up to +3.0 HOTA over McByte on SportsMOT online. (search) — [SAMIDARE](https://arxiv.org/pdf/2604.22162); [McByte++](https://www.alphaxiv.org/abs/2608.15688)
- "Team-Aware Football Player Tracking with SAM" **[NEW, December 2025]** (arXiv 2512.08467) describes ByteTrack as struggling with heavy occlusion and similar-looking players in team sports. (search) — [arXiv 2512.08467](https://arxiv.org/html/2512.08467v1)
- BoT-SORT is ByteTrack-based with added camera-motion compensation and re-identification. (search) — [arXiv 2511.09455](https://arxiv.org/pdf/2511.09455v1)
- SAM-Body4D uses SAM 3 masklets for identity-consistent multi-human mesh recovery with occlusion-aware refinement. (fetched) — [arXiv 2512.08406](https://arxiv.org/abs/2512.08406)
- CoMotion performs joint online multi-person tracking and SMPL pose estimation and tracks through occlusion; outputs MOT-format tracks. (fetched) — [apple/ml-comotion](https://github.com/apple/ml-comotion)
- SAM 3 on Apple Silicon is not officially supported out of the box: open PR for MPS/CPU inference and reports of Triton-related failures on M4; SAM 2 has an issue thread on Apple Silicon use; a third-party project reports running SAM 3.1 on MPS. (search) — [sam3 PR #627](https://github.com/facebookresearch/sam3/pull/627); [sam2 issue #687](https://github.com/facebookresearch/sam2/issues/687); [uppin/surmap PR #63](https://github.com/uppin/surmap/pull/63)
- Vid2Player3D used plain YOLOv4 detection/tracking for two tennis players. (fetched) — [nv-tlabs/vid2player3d](https://github.com/nv-tlabs/vid2player3d)

### Inferences
- Pickleball has at most 4 players, a static camera, no camera-motion compensation need, and players are confined to their half between side switches. Identity can be resolved mostly in court coordinates: project foot points to the court plane, associate by nearest ground position with a constant-velocity model, and use the net line as a hard separator. Appearance (clothing colour histogram or a ReID embedding) is needed only for same-side partner crossings in doubles and for re-labelling after end changes and time-outs (cuts/replays in the broadcast must be detected and excluded first).
- Masks (SAM 2/3) are still valuable for a different reason: as prompts to SAM 3D Body / PromptHMR to disambiguate overlapping partners and to exclude ball kids, referees and spectators behind the court.
- Scoreboard/rally segmentation (as in TT4D) naturally bounds tracking to single rallies, where identity switches are rare; cross-rally identity is a small assignment problem.

### Gaps
- No BoT-SORT SportsMOT number retrieved; no racket-sport-specific tracker paper found.
- SAM 3 throughput and memory on Apple Silicon unknown.
- No source addresses identity through broadcast side switches specifically; the approach above is inference.

---

## Cross-cutting: licensing and Apple Silicon summary

### Takeaway
SMPL/SMPL-X-based methods (GVHMR, WHAM, TRAM, PromptHMR, CoMotion, Human3R, PHC and friends) inherit a non-commercial research license; the permissive routes found are MHR (SAM 3D Body) and SOMA (GEM-X). On Apple Silicon, verified-by-report options are GVHMR (community MPS ports), SAM 3D Body (MPS with MHR on CPU), CoMotion (Core ML detection stage), MuJoCo (CPU), and ONNX-packaged 2D pose models.

### Cited Findings
- SMPL commercial licensing is handled via Max Planck Innovation / Meshcapade; SMPL-X is under a "Software Copyright License for non-commercial scientific research purposes". (search) — [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/); [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)
- MHR is described by its paper as released under an industry-friendly license and by a mirror as Apache-2.0; an MHR <-> SMPL/SMPL-X converter exists in the MHR repo. (search) — [MHR paper](https://arxiv.org/html/2511.15586v3)
- SAM 3D Body: SAM License. (fetched) — [sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- GEM-X: Apache-2.0 code, SOMA body model, weights under NVIDIA terms described as commercially usable; original GENMO/GEM (SMPL) is NVIDIA OneWay Noncommercial. (fetched / search) — [GEM-X](https://github.com/NVlabs/GEM-X); [GENMO](https://github.com/NVlabs/GENMO)
- TRAM code MIT (third-party components separate); SAM-Body4D code MIT; rtmlib and ViTPose Apache-2.0; Sapiens non-commercial. (search / fetched) — [tram](https://github.com/yufu-wang/tram); [sam-body4d](https://github.com/gaomingqi/sam-body4d); [rtmlib](https://github.com/Tau-J/rtmlib); [Sapiens](https://arxiv.org/html/2408.12569v1)
- Apple Silicon reports: GVHMR MPS port matches CPU within about 1.1 mm; SAM 3D Body needs MHR head on CPU; VGGT camera estimation blew up to 50.2 GB on a Mac (irrelevant for static camera); DPVO is CUDA-only. (search / fetched) — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199); [sam-3d-body #93](https://github.com/facebookresearch/sam-3d-body/issues/93); [anim8te PR #12](https://github.com/cmenguy/anim8te/pull/12)

### Inferences
- If the project is personal/non-commercial, SMPL-based GVHMR is the fastest path to a working result on the Mac. If there is any chance of public/commercial use, build the internal motion representation on MHR (or SOMA) from the start and treat SMPL-based models only as replaceable estimators whose joint rotations are retargeted.
- Everything needing SLAM (DROID-SLAM, DPVO, ViPE, VGGT) can be dropped because the camera is static and calibrated, which removes the largest block of CUDA-only dependencies from every pipeline listed.

### Gaps
- No verified MLX ports of any HMR model were found.
- Actual throughput on an M5 Pro for any of these models is unmeasured in public sources; a local benchmark on a 30-second rally clip is the necessary next step.
- Whether SOMA assets are freely redistributable and whether a SOMA/MHR <-> SMPL converter exists for GEM-X was not determined.
