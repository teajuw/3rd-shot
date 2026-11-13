# SAM 3D Body and SAM 3D Objects for broadcast pickleball player (and paddle) reconstruction

Research date: 2026-11-12. Target use: four players (two near, two far at roughly 80-150 px tall), one fixed 1080p50 broadcast camera, offline on a 64 GB Apple Silicon Mac mini, with an existing calibrated camera, YOLO11 COCO-17 keypoints and court-plane foot positions.

Source-quality note: the CVPR open-access PDF returned HTTP 403, so paper numbers below come from the arXiv HTML version and the official GitHub README. Several third-party pages (papernotes, ComfyUI wiki, a fork's issue tracker) are secondary and are marked as such.

## 1. What exactly are SAM 3D Body (3DB) and SAM 3D Objects?

### Takeaway
SAM 3D Body is a single-image, promptable, full-body (body + feet + hands) human mesh recovery model released by Meta on 19 Nov 2025; it outputs Momentum Human Rig (MHR) parameters rather than SMPL, and accepts 2D keypoint and mask prompts. SAM 3D Objects is a separate single-image, mask-conditioned object reconstructor (shape, texture, pose/layout) from the same release.

### Cited Findings
- Release date for both models, checkpoints, demo and paper: 19 November 2025. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/); [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- The paper "SAM 3D Body: Robust Full-Body Human Mesh Recovery" (Yang, Kukreja, Pinkus, Sagar, Fan, Park, Shin, Cao, Liu, Ugrinovic, Feiszli, Malik, Dollar, Kitani) is on arXiv as 2602.15989, v1 submitted 17 Feb 2026 (the paper PDF had been distributed by Meta since Nov 2025). — [arXiv 2602.15989](https://arxiv.org/abs/2602.15989)
- A CVPR 2026 open-access version and supplemental exist (third-party write-ups describe it as a CVPR 2026 oral; I could not open the CVF PDF to confirm "oral"). — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Yang_SAM_3D_Body_CVPR_2026_supplemental.pdf); [mer.vin write-up](https://mer.vin/2026/06/sam-3d-body-promptable-full-body-3d-mesh-from-one-image-cvpr-2026-mhr/)
- Scope per the README: "single-image full-body 3D human mesh recovery"; no video or temporal mode is documented. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- Two checkpoints: DINOv3-H+ (840M parameters, `facebook/sam-3d-body-dinov3`) and ViT-H (631M parameters, `facebook/sam-3d-body-vith`); both gated behind a Hugging Face access request. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- Architecture: shared image encoder over a person crop; a decoder whose query tokens include pose, prompt, 2D-keypoint, 3D-keypoint and hand tokens; a separate hand decoder that consumes hand crops; at inference the hand-decoder wrist and body-decoder elbow positions are fed back as prompts to refine the full-body result. — [papernotes summary of the paper (secondary)](https://en.papernotes.org/CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery/)
- Encoder input is a person crop resized to 512x512. — [arXiv 2602.15989 HTML](https://arxiv.org/html/2602.15989)
- Prompts: optional 2D keypoints (encoded as tokens) and masks (convolved and added to image embeddings); person boxes come from a detector (ViTDet by default, or SAM 3 via `--detector_name sam3`). — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body); [papernotes (secondary)](https://en.papernotes.org/CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery/)
- Camera: the model regresses camera parameters with the MHR tokens; intrinsics/FOV come from an external estimator, MoGe-2, by default in the demo. Benchmarks on 3D datasets used ground-truth intrinsics. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body); [arXiv 2602.15989 HTML](https://arxiv.org/html/2602.15989)
- Outputs: MHR parameters for pose, shape, camera and skeleton, giving a full-body mesh with body, feet and hands. — [arXiv 2602.15989](https://arxiv.org/abs/2602.15989); [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- MHR "separates skeletal structure from surface shape" (unlike SMPL, where bone lengths are a by-product of shape betas). — [arXiv 2602.15989](https://arxiv.org/abs/2602.15989)
- MHR parameterisation: 45 identity/shape parameters (20 body, 20 head, 5 hands), 204 pose parameters (joint angles and scalings), 72 facial expression blendshapes, 7 levels of detail (the TorchScript model supports LOD 1 only), neural non-linear pose correctives; Python 3.11+, runs on CPU or GPU via PyTorch, integrates with PyMomentum. — [MHR README](https://github.com/facebookresearch/MHR)
- The MHR mesh used by 3DB has 18,439 vertices versus 6,890 for SMPL. — [Fast SAM 3D Body, arXiv 2603.15603](https://arxiv.org/html/2603.15603)
- MHR repo ships a `tools/mhr_smpl_conversion` tool for converting between MHR and SMPL/SMPL-X. — [MHR README](https://github.com/facebookresearch/MHR)
- Training data: about 8 million images (Meta blog) / about 7 million high-quality annotations (paper summary), fitted with 595 dense 2D keypoints, mixing single-view, multi-view (Ego-Exo4D, Harmony4D, Goliath, InterHand2.6M) and synthetic data. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/); [papernotes (secondary)](https://en.papernotes.org/CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery/)
- SAM 3D Objects: input is an image plus a mask per object; output is per-object "pose, shape, texture, and layout", exported as a Gaussian splat PLY in the example; a notebook `demo_3db_mesh_alignment.ipynb` puts SAM 3D Objects and SAM 3D Body outputs in one frame of reference. — [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)
- SAM 3D Objects training data: about 1 million annotated images and about 3.14 million model-in-the-loop meshes. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)

### Inferences
- The existing pipeline's YOLO11 COCO-17 keypoints and any player masks can be fed directly as 3DB prompts, and the calibrated camera can replace MoGe-2, which removes one external model and one error source (the paper lists MoGe-2 dependence as an error source per the secondary summary).
- Because MHR decouples skeleton from shape, per-player bone lengths can be locked once per match, which is the natural hook for the temporal pipelines in section 3.
- Anything downstream that expects SMPL/SMPL-X (most physics, retargeting and motion-prior tooling) needs the MHR-to-SMPL conversion step.

### Gaps
- MHR joint count and per-LOD vertex counts were not found in the pages read (only the 18,439-vertex figure from a third-party paper).
- I could not confirm the exact field names of the Python output dict (e.g. 3D keypoints, camera translation) from the README; the demo only documents rendered visualisation.

## 2. Reported accuracy versus prior methods; small, low-resolution or occluded people

### Takeaway
As a per-image model 3DB is state of the art or near it on the standard benchmarks and even beats video methods on EMDB joint error, with large margins on hard poses, truncation and occlusion; however there is no published evaluation on small, far, low-resolution people, and one third-party ablation shows error rising steeply as input resolution drops.

### Cited Findings
- Official checkpoint table (11/19/2025 checkpoints): DINOv3-H+ 3DPW MPJPE 54.8, EMDB MPJPE 61.7, RICH PVE 60.3, COCO PCK@.05 86.5, LSPET PCK@.05 68.0, FreiHAND PA-MPJPE 5.5; ViT-H 54.8 / 62.9 / 61.7 / 86.8 / 68.9 / 5.5. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- Comparison with image methods (3DPW PA-MPJPE / EMDB MPJPE / EMDB PVE / RICH PA-MPJPE): HMR2.0b 54.3 / 118.5 / 140.6 / 48.1; CameraHMR 35.1 / 70.3 / 81.7 / 34.0; PromptHMR 36.1 / 71.7 / 84.5 / 37.3; NLF-L+fit 33.6 / 68.4 / 80.6 / 28.7; 3DB-H 33.2 / 62.9 / 74.3 / 31.9. NLF remains better on RICH. — [papernotes transcription of paper table (secondary)](https://en.papernotes.org/CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery/)
- Comparison with video methods (PA-MPJPE / MPJPE / PVE): 3DPW — WHAM 35.9 / 57.8 / 68.7, TRAM 35.6 / 59.3 / 69.6, GENMO 34.6 / 53.9 / 65.8, 3DB-H 33.2 / 54.8 / 64.1. EMDB — WHAM 50.4 / 79.7 / 94.4, TRAM 45.7 / 74.4 / 86.6, GENMO 42.5 / 73.0 / 84.8, 3DB-H 38.5 / 62.9 / 74.3. The paper words this as "competitive with video-based approaches". — [arXiv 2602.15989 HTML, Table 2](https://arxiv.org/html/2602.15989)
- SA1B-Hard categorical 2D results (aPCK body / feet), CameraHMR vs PromptHMR vs 3DB: sports or athletic pose 78.93/69.34 vs 73.62/60.37 vs 85.10/82.80; occlusion 62.93/26.83 vs 58.00/30.81 vs 75.43/54.74; overlapped multi-people 53.11/41.88 vs 57.17/41.43 vs 70.82/64.71; overhead view 55.08/39.46 vs 43.65/24.63 vs 73.33/66.94. — [arXiv 2602.15989 HTML, Table 5](https://arxiv.org/html/2602.15989)
- 3D categorical results (PVE / MPJPE / PA-MPJPE): "very hard" 3D poses — CameraHMR 213.66 / 206.34 / 143.23, PromptHMR 186.35 / 179.46 / 129.51, 3DB 114.20 / 110.62 / 86.43; severe truncation — 230.51 / 213.64 / 124.01, 186.57 / 168.22 / 122.70, 126.53 / 113.66 / 88.42. — [arXiv 2602.15989 HTML, Table 6](https://arxiv.org/html/2602.15989)
- 2D keypoint prompting (3DB-H, COCO PCK@0.05 / EMDB MPJPE): 0 prompts 86.7 / 63.3; 1 prompt 90.2 / 60.1; 2 prompts 93.0 / 58.9. With noise on one prompt (as a fraction of the box): 0.01 → 90.2 / 60.3; 0.03 → 89.5 / 61.5; 0.05 → 87.6 / 63.3; 0.1 → 80.9 / 67.8. So noisy keypoint prompts (10% of box) make results worse than no prompt. — [arXiv 2602.15989 HTML, Table 7](https://arxiv.org/html/2602.15989)
- Mask prompting (3DB-DINOv3, PVE / MPJPE without → with mask): Hi4D 91.4 / 76.4 → 58.3 / 47.0; Harmony4D 42.7 / 35.6 → 36.5 / 30.1; SA1B-Hard multi-person subset avg-PCK 67.9 → 72.3. — [arXiv 2602.15989 HTML, Table 8](https://arxiv.org/html/2602.15989)
- A user study with about 7,800 participants gave roughly a 5:1 preference win rate over prior methods. — [papernotes (secondary)](https://en.papernotes.org/CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery/)
- Independent fitness benchmark (Fit3D): SAM 3D Body MPJPE 70.65 vs GVHMR 71.94 and PromptHMR 75.09, on a protocol the authors note does not reward temporal consistency (as reported in a search summary; I did not open the paper itself). — [Fit3D benchmark paper, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13413046/)
- Resolution sensitivity: in the Fast SAM 3D Body ablation, reducing the model input to 128 px gives MPJPE 86.78 versus 58.96 at 512 px. — [Fast SAM 3D Body, arXiv 2603.15603](https://arxiv.org/html/2603.15603)
- SAM-Body4D authors report that when a target is small, blurred or heavily occluded the 3DB pipeline's detector may miss the person entirely, and that heavy occlusion yields distorted or hallucinated meshes. — [SAM-Body4D, arXiv 2512.08406](https://arxiv.org/html/2512.08406v1)
- Closest sports-domain evidence (does not include 3DB): CalTennis (11M frames, 60 Hz, 1080p phone cameras at 1.65 m, 90% of poses 13.4-16.7 m from camera) benchmarks TRAM, GVHMR, GENMO, WHAM and PromptHMR; mean translation error is 0.9-3.6 m, multi-view pose disagreement about 11 cm, predictions "degrade significantly over distances exceeding half the court length", and feet and hands disagree most. SAM 3D Body appears only in related work. — [CalTennis, arXiv 2606.20542](https://arxiv.org/html/2606.20542)
- Stated model limitation: body shapes for all age groups are not modelled, so children may be reconstructed poorly. — [arXiv 2602.15989 HTML, Appendix C](https://arxiv.org/html/2602.15989)

### Inferences
- The two near players (several hundred px tall) are squarely in the regime where the benchmark numbers apply. The two far players at 80-150 px are upsampled 3.4-6.4x to fill the 512 px crop; the 128 px ablation (MPJPE about +28 mm) is the best available proxy and suggests clearly degraded but usable body pose, with hands and wrists unlikely to be recoverable at that size (a hand is roughly 8-15 px).
- YOLO11 keypoints on the far players are probably noisier than 3-5% of the box; per Table 7 that is the range where prompting stops helping or hurts. Prompt with keypoints only where confidence is high (or use masks, which showed the larger gains in crowded/overlap cases such as doubles partners crossing).
- The existing calibrated camera plus court-plane foot positions already solve what CalTennis identifies as the weakest part of all monocular methods (metre-level translation/depth error); 3DB would then only be asked for root-relative pose and shape, where it is strongest.

### Gaps
- No published 3DB evaluation stratified by person pixel height, and no evaluation on racket-sport broadcast footage, was found. A direct test on the user's own footage is needed.
- No quantitative comparison of ground-truth versus MoGe-2 intrinsics was reported in the paper.
- The Fit3D numbers come from a search snippet, not a full read of the paper.

## 3. Is it per-image only? Temporal smoothness and world-grounded trajectories on video

### Takeaway
Yes, 3DB is strictly per-image and per-person; Meta has released no video version or successor as of November 2026. Temporal consistency and world grounding are supplied by third-party wrappers (SAM-Body4D, a world-coordinate retargeting pipeline, ComfyUI's native nodes, Fast SAM 3D Body), none of which reports quantitative temporal metrics.

### Cited Findings
- Official statement: 3DB "processes each individual separately, without considering multi-person or human-object interactions". — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)
- The official repo's only dated update is the 11/19/2025 release; no video mode is documented. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- SAM-Body4D (Gao, Miao, Han; 9 Dec 2025), training-free: SAM 3 video masklets for identity-consistent tracking → Diffusion-VAS occlusion-aware mask completion (frame flagged occluded when completed mask is larger and IoU < 0.7) → 3DB with masks as prompts, batched across people → Kalman smoothing of pose and hand parameters in MHR space, with shape and scale frozen to the first visible frame. Evaluation is qualitative only. On an A100 80 GB, a 480x854, 90-frame, 5-person clip ran about 2x faster than sequential per-frame inference at batch size 32. Code: github.com/gaomingqi/sam-body4d. — [arXiv 2512.08406](https://arxiv.org/html/2512.08406v1)
- SAM-Body4D documents the per-frame failure modes: temporal jitter, identity mix-ups, missed detections of small/blurred/occluded people, distorted meshes under occlusion. It does not claim world-coordinate output. — [arXiv 2512.08406](https://arxiv.org/html/2512.08406v1)
- "World-Coordinate Human Motion Retargeting via SAM 3D Body" (Tu, Su, Zhu, Zheng; 25 Dec 2025): frozen per-frame 3DB → SORT-style tracking (Kalman on boxes + Hungarian IoU) → shape/scale averaged per track and locked so bone lengths are constant → sliding-window optimisation in MHR latent space (fidelity to 3DB + Charbonnier velocity/acceleration terms) → a ground optimiser using soft foot-contact probabilities, foot-sliding and ground-penetration terms and a soft camera prior to recover the global root trajectory → retarget to a Unitree G1. No quantitative results, runtime or code link reported. — [arXiv 2512.21573](https://arxiv.org/html/2512.21573v1)
- That paper states per-frame 3DB "often suffer[s] from temporal fluctuations" in shape, skeletal scale and pose, and that monocular depth ambiguity makes global root translation noisy (shaking, foot sliding); it lists multi-person relative depth as an open limitation. — [arXiv 2512.21573](https://arxiv.org/html/2512.21573v1)
- ComfyUI core integrated SAM 3D Body natively on 23 Aug 2026 (Kijai's PR #14370): nodes `SAM3DBody_Loader/Predict/Render/Smooth/FaceExpression`, a temporal smoothing node for video, SAM 3.1 video tracks for multi-person, MoGe FOV nodes, GLB and BVH export, and repackaged bf16 and INT8 weights. Smoothing method, VRAM and speed are not documented on the news page. — [ComfyUI wiki news (secondary)](https://comfyui-wiki.com/en/news/2026-08-23-sam-3d-body-native-comfyui)
- Fast SAM 3D Body notes single-view self-occlusion causes jitter during large body rotations (they added a two-camera extension for teleoperation). — [arXiv 2603.15603](https://arxiv.org/html/2603.15603)
- A targeted search for a Meta successor ("SAM 3D Body 2", "SAM 4D", an official video HMR release in 2026) found none. — [Meta AI blog (latest official announcement found)](https://ai.meta.com/blog/sam-3d/); [alphaxiv listing of the 2026 paper](https://www.alphaxiv.org/abs/2602.15989)
- Dedicated video HMR methods remain weak exactly where the user's pipeline is strong: on CalTennis, translation error of TRAM/GVHMR/GENMO/WHAM/PromptHMR is 0.9-3.6 m with depth-axis "pose drifting". — [CalTennis, arXiv 2606.20542](https://arxiv.org/html/2606.20542)

### Inferences
- The community recipe is consistent across sources and maps cleanly onto the existing pipeline: (a) track identities (already have), (b) lock MHR shape/scale per player across the match, (c) smooth pose in MHR parameter space (Kalman or windowed optimisation), (d) replace 3DB's noisy camera-space translation with a ground-contact solve. Here step (d) can use the calibrated camera and court-plane foot positions directly rather than a learned contact model, which should be better than anything in the published wrappers.
- At 50 fps there is ample temporal redundancy for smoothing; fast swings (wrist/paddle) are the place where low-pass smoothing will damage real signal, so smoothing strength should be joint-dependent (the retargeting paper already uses per-joint weights).
- Because no wrapper publishes temporal metrics (jitter, acceleration error, foot slide), the claim that "3DB + smoothing" matches purpose-built video models such as GVHMR/GENMO on temporal quality is unverified.

### Gaps
- No quantitative temporal-consistency benchmark of 3DB-based video pipelines was found.
- No code release was found for the world-coordinate retargeting paper.
- I did not exhaustively survey every 2026 follow-up; other 3DB-based video papers may exist beyond the three found.

## 4. Hand and wrist quality (paddle orientation)

### Takeaway
3DB's hands are close to, but below, dedicated hand models (WiLoR, HaMeR) on FreiHAND, and both Meta and the paper say so explicitly; the hand branch needs a usable hand crop, so it helps for near players only.

### Cited Findings
- FreiHAND (PA-MPVPE / PA-MPJPE / F@5 / F@15): WiLoR 5.1 / 5.5 / 0.825 / 0.993; MaskHand 5.4 / 5.5 / 0.801 / 0.991; HaMeR 5.7 / 6.0 / 0.785 / 0.990; 3DB-DINOv3 6.2 / 5.5 / 0.737 / 0.988; 3DB-H 6.3 / 5.5 / 0.735 / 0.988. The specialists were trained on FreiHAND; 3DB was not. — [arXiv 2602.15989 HTML, Table 4](https://arxiv.org/html/2602.15989)
- Meta: hand pose accuracy is improved but "doesn't surpass specialized hand-only pose estimation methods". — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)
- Paper limitations: hand estimation from the body decoder alone is suboptimal (attributed to limited high-quality full-body training data); the separate hand decoder is what brings quality up. — [arXiv 2602.15989 HTML, Appendix C](https://arxiv.org/html/2602.15989)
- The hand decoder works on hand crops, and its wrist estimate is fed back to the body decoder; hand-detection/wrist-localisation failures limit the gains. — [papernotes (secondary)](https://en.papernotes.org/CVPR2026/human_understanding/sam_3d_body_robust_full-body_human_mesh_recovery/)
- Hand refinement is optional in the released demo. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- Independent evidence that extremities are the weak point in racket sport: on CalTennis, feet and hands show the highest cross-view disagreement, especially under partial occlusion (for other methods, not 3DB). — [CalTennis, arXiv 2606.20542](https://arxiv.org/html/2606.20542)
- 3DB does not model human-object interaction. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)

### Inferences
- FreiHAND measures root-aligned finger articulation on clean hand crops. Paddle orientation depends mostly on global wrist/forearm rotation under motion blur with the hand wrapped around (and partly occluded by) a handle; none of the cited numbers measures that, so they are an optimistic upper bound.
- For far players (hand about 8-15 px) the hand decoder has essentially no signal; expect a prior-driven default hand. Wrist rotation there must come from elsewhere (paddle detection, forearm direction, motion priors).
- For near players, 3DB's wrist rotation is a reasonable initial estimate to be fused with a direct paddle observation (paddle-face ellipse/quad in the image), rather than a sole source of paddle orientation.

### Gaps
- No benchmark of 3DB wrist rotation, or of hands holding rackets/paddles, was found.
- No evaluation of hand quality under motion blur at 50 fps broadcast shutter speeds was found.

## 5. Inference cost; Apple Silicon (MPS / MLX / CPU); ports; newer versions

### Takeaway
3DB is heavy (631M-840M parameters; roughly 0.5 s per person crop on a workstation NVIDIA GPU, about 1 fps end-to-end) and officially CUDA-oriented; on Apple Silicon it runs only via workarounds (MHR head pinned to CPU, unmerged MPS patches), with no trustworthy published Mac timings. SAM 3D Objects officially requires Linux and an NVIDIA GPU with at least 32 GB VRAM, though an unofficial MLX port exists.

### Cited Findings
- Model sizes: DINOv3-H+ 840M parameters; ViT-H 631M parameters. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- The paper itself reports no inference speed. — [arXiv 2602.15989 HTML](https://arxiv.org/html/2602.15989)
- Third-party measurement of stock 3DB on one NVIDIA RTX 6000 Ada (PyTorch 2.5.1, batch 1): 1.9-2.0 fps per pre-cropped person ("oracle" boxes) and 0.6-1.0 fps per frame for the full automatic pipeline (detector + FOV + body + hands). — [Fast SAM 3D Body, arXiv 2603.15603, Table 1](https://arxiv.org/html/2603.15603)
- Fast SAM 3D Body (Yang et al., 16 Mar 2026; code at github.com/yangtiming/Fast-SAM-3D-Body), training-free: YOLO11-pose instead of ViTDet-H, hand boxes from wrist keypoints, batched body+hand crops, pruned decoder passes (removes the keypoint-prompted second pass and pose correctives), 384 px input, MoGe-2 small, TensorRT FP16 + CUDA Graphs. Result: 6.3-8.3 fps automatic (up to 10.9x), about 8.3-9.7 fps per crop, 152 ms/frame on 3DPW; about 65 ms end-to-end on an RTX 5090. — [arXiv 2603.15603](https://arxiv.org/html/2603.15603)
- Fast variant accuracy cost is small but not zero (3DB vs Fast, oracle, PA-MPJPE / MPJPE / PVE): 3DPW 33.8 / 54.8 / 63.6 vs 30.4 / 59.5 / 68.9; EMDB 38.2 / 61.7 / 72.5 vs 37.3 / 64.3 / 74.4; RICH 30.9 / 53.7 / 60.3 vs 33.8 / 58.8 / 64.2; FreiHAND F@5 0.737 vs 0.714. — [arXiv 2603.15603, Table 1](https://arxiv.org/html/2603.15603)
- Fast SAM 3D Body also replaces iterative MHR→SMPL fitting (0.24 fps) with a 2.5M-parameter MLP (about 2,000 fps), at near-identical accuracy (3DPW 31.1 / 59.4 / 68.6). — [arXiv 2603.15603](https://arxiv.org/html/2603.15603)
- TensorRT and CUDA Graphs are NVIDIA-only; the Fast paper mentions no Apple/MPS/CPU path. — [arXiv 2603.15603](https://arxiv.org/abs/2603.15603)
- Apple Silicon, official repo: issue #93 (opened 15 Dec 2025, still open when read) — on macOS/MPS `SAM3DBodyEstimator.process_one_image` crashes in `MHRHead.mhr_forward` with "Placeholder storage has not been allocated on MPS device!" because the MHR TorchScript asset is not MPS-compatible. Reporter's workaround: run the MHR head on CPU and keep the rest on MPS; outputs are produced. A fork commit "feat: add Apple Silicon (MPS) backend support" (8 Apr 2026) and PR #125 reference it; neither was shown as merged. — [sam-3d-body issue #93](https://github.com/facebookresearch/sam-3d-body/issues/93)
- A community fork's issue tracker states autocast is hard-coded to `device_type="cuda"` (so MPS runs full precision), that the TorchScript `mhr_model.pt` is pinned to CPU on Apple Silicon causing sync stalls, and cites about 0.49 s/frame on MPS as a baseline with projected (not measured) 1.5-2x gains from MPS autocast and batching. Hardware for the 0.49 s figure is not stated. — [PoisedLabs fork issue #1](https://github.com/PoisedLabs/sam-3d-body/issues/1); [PoisedLabs fork issue #2](https://github.com/PoisedLabs/sam-3d-body/issues/2)
- MHR itself runs on CPU or GPU through PyTorch and is compatible with a fast CPU solver via PyMomentum. — [MHR README](https://github.com/facebookresearch/MHR)
- ComfyUI's native integration (Aug 2026) ships repackaged `sam_3d_body_dinov3_bf16.safetensors` and `..._int8_convrot.safetensors` weights and says rendering uses only PyTorch; Mac/MPS support is not stated. — [ComfyUI wiki news (secondary)](https://comfyui-wiki.com/en/news/2026-08-23-sam-3d-body-native-comfyui)
- SAM 3D Objects official prerequisites: 64-bit Linux and "an NVIDIA GPU with at least 32 GB of VRAM"; install uses CUDA 12.1 builds of PyTorch, Kaolin and PyTorch3D. — [sam-3d-objects setup.md](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- SAM 3D Objects updates in 2026: encoder weights released 1 Jun 2026; "3D Artist Object Set" and a Hugging Face leaderboard released 2 Jun 2026. — [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)
- Unofficial Apple Silicon ports of SAM 3D Objects: an MLX 8-bit bundle (4.62 GB vs 13.7 GB full-precision MLX) running on `mlx-spatial` without Torch/CUDA, producing Gaussian-splat PLY and optional GLB; no speed benchmark given. A separate "Sam3D-Objects-MLX" project rewires the pipeline to PyTorch MPS, without Gaussian splatting or colour baking. — [appautomaton MLX 8-bit model card](https://huggingface.co/appautomaton/sam-3d-objects-mlx-8bit); [Sam3D-Objects-MLX README mirror](https://1clonegithubbyxjn.1934672944.workers.dev/ZimengXiong/Sam3D-Objects-MLX/blob/main/README.md)
- Related, if SAM 3/3.1 is wanted for player masks: an open PR reports SAM 3.1 tracking 720p video on an M5 at about 1.3 fps with about 7 GB peak memory, CPU about 50x slower. — [facebookresearch/sam3 PR #627](https://github.com/facebookresearch/sam3/pull/627)
- No newer official version or successor of SAM 3D Body in 2026 was found; the 11/19/2025 checkpoints are current. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)

### Inferences
- Rough offline budget (my arithmetic from the unverified 0.49 s/crop MPS figure): 4 players x 50 fps = 200 crops per second of video, so about 100 seconds of compute per second of video, i.e. a 10-minute game would take on the order of 16-17 hours at full frame rate. Running 3DB at 10-25 fps and interpolating/smoothing in MHR space, skipping the hand decoder for far players, and batching crops would cut this by 2-10x. Treat all of this as an estimate to be measured.
- 64 GB unified memory is comfortably enough for an 840M-parameter model (about 3.4 GB in fp32) plus batched 512 px crops; memory is not the constraint, throughput is.
- The Fast SAM 3D Body speedups that are algorithmic (YOLO-pose boxes, wrist-derived hand boxes, single batched backbone pass, dropped second decoder pass, 384 px input) should port to MPS; the TensorRT/CUDA-Graphs share will not. The existing pipeline already has the YOLO11 pose boxes and a calibrated camera, so the two slowest auxiliary stages in stock 3DB (ViTDet-H detector, MoGe-2 FOV) can be skipped outright.
- NVIDIA required, stated plainly: official SAM 3D Objects (Linux + 32 GB VRAM NVIDIA GPU); Fast SAM 3D Body's TensorRT path; SAM-Body4D as published (A100, plus Diffusion-VAS). 3DB itself is "works on Mac with patches", not officially supported.

### Gaps
- No measured, hardware-attributed 3DB benchmark on any Apple Silicon chip (MPS, MLX or CPU) was found; the 0.49 s/frame figure is an unattributed claim in a fork issue.
- No MLX or Core ML port of SAM 3D Body (as opposed to SAM 3D Objects and SAM 3) was found.
- VRAM/RAM footprint of 3DB at various batch sizes is not documented in the sources read.
- Whether upstream `main` has merged MPS support since the issue page was rendered could not be confirmed.

## 6. License of code, weights and MHR; commercial restrictions

### Takeaway
3DB and SAM 3D Objects code and checkpoints are under Meta's custom "SAM License" (royalty-free, redistribution and modification allowed, use restrictions around military/weapons/trade controls, Meta can change terms); MHR is Apache-2.0. Neither is a research-only licence, which is a material advantage over SMPL/SMPL-X.

### Cited Findings
- sam-3d-body and sam-3d-objects: code and checkpoints are under the SAM License. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body); [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)
- SAM License (last updated 19 Nov 2025): grants a "non-exclusive, worldwide, non-transferable and royalty-free limited license" to use, reproduce, distribute and modify; redistribution must carry the same agreement; publications must acknowledge use; prohibits ITAR-controlled and military/warfare, nuclear, espionage and weapons uses; licensee indemnifies Meta; Meta may modify the agreement at any time with continued use as acceptance. The text does not use the word "commercial". — [sam-3d-body LICENSE](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)
- Roboflow's licence summary characterises the SAM License (for SAM 3) as royalty-free and permitting commercial use subject to those conditions. — [Roboflow licence page (secondary)](https://roboflow.com/model-licenses/segment-anything-3)
- MHR is Apache-2.0; Meta describes it as shared under a "permissive commercial license". — [MHR README](https://github.com/facebookresearch/MHR); [Meta AI blog](https://ai.meta.com/blog/sam-3d/)
- Checkpoints are gated: access must be requested on Hugging Face. — [sam-3d-body README](https://github.com/facebookresearch/sam-3d-body)
- Derived bundles inherit the SAM License (the MLX SAM 3D Objects port is distributed under it; its bundled MoGe ViT-L checkpoint is Apache-2.0). — [appautomaton MLX model card](https://huggingface.co/appautomaton/sam-3d-objects-mlx-8bit)
- A third-party site (sam3d.org) claims Apache-2.0 for SAM 3D Objects; this conflicts with the official repo and should be disregarded. — [sam3d.org mirror](https://backiee.wasmer.app/https_sam3d_org); contradicted by [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)

### Inferences
- For a personal/offline digital-twin project, and likely for a commercial product, the licence is workable; the practical risks are the unilateral-modification clause and the indemnity, not a non-commercial restriction. This is not legal advice.
- Converting outputs to SMPL/SMPL-X via the MHR tool reintroduces SMPL's own (non-commercial by default) body-model licence for the SMPL assets; staying in MHR avoids that.

### Gaps
- The licence of the SAM 3D Body dataset was not checked.
- Licences of the default auxiliary models (ViTDet detector weights, MoGe-2 variant used, DINOv3 backbone terms) were not individually verified; DINOv3 weights are folded into the 3DB checkpoint but may carry their own terms.

## 7. Could SAM 3D Objects reconstruct or pose a hand-held paddle?

### Takeaway
In principle it accepts any masked object and returns shape plus pose/layout, and it can be aligned with a 3DB body, but nothing found supports using it per-frame for a small, thin, motion-blurred, hand-occluded paddle; it is single-image, generative, officially needs a 32 GB NVIDIA GPU, and does not reason about contact. It is a poor fit for paddle pose tracking; a known rigid paddle model fitted to image evidence is the more defensible route.

### Cited Findings
- Input is an image plus an object mask; outputs "pose, shape, texture, and layout" per object; no video support is mentioned. — [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)
- Meta's stated limitations: moderate output resolution limits detail in complex objects; objects are predicted one at a time and the model "isn't trained to reason about physical interactions, such as contact or interpenetration". — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)
- 3DB likewise does not consider human-object interaction. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)
- The repo claims robustness to small objects, occlusion and clutter (a marketing-level claim without per-size numbers on the README). — [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)
- A port's model card lists mask-quality sensitivities: background leakage, broad masks, soft edges, reflections "and thin structures can reduce reconstruction quality"; and notes single-view reconstruction cannot determine unseen geometry with certainty. — [appautomaton MLX model card](https://huggingface.co/appautomaton/sam-3d-objects-mlx-8bit)
- Speed: Meta describes full textured reconstructions "within a few seconds" per object. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/)
- Hardware: Linux + NVIDIA GPU with at least 32 GB VRAM officially; unofficial MLX/MPS ports exist for Apple Silicon with reduced features or quantisation. — [sam-3d-objects setup.md](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md); [appautomaton MLX model card](https://huggingface.co/appautomaton/sam-3d-objects-mlx-8bit)
- A notebook (`demo_3db_mesh_alignment.ipynb`) aligns SAM 3D Objects and SAM 3D Body outputs into one frame. — [sam-3d-objects README](https://github.com/facebookresearch/sam-3d-objects)

### Inferences
- A pickleball paddle is a near-planar plate about 1.5 cm thick, often edge-on to the camera, motion-blurred during swings, partly hidden by the hand, and only about 10-40 px across even for near players at broadcast framing. That combines every listed weakness (thin structure, soft/blurred mask edges, occlusion, unseen geometry), and a generative model will hallucinate a new, slightly different shape and scale every frame, giving no temporally stable pose.
- At "a few seconds" per object on a 32 GB NVIDIA GPU, per-frame use at 50 fps for four paddles is not practical on a Mac mini even with the MLX port.
- The plausible limited use is one-off asset creation: reconstructing a textured paddle mesh from a clean close-up still. Since paddle dimensions are regulated and known, a hand-made rigid model is simpler and more accurate.
- Paddle pose is better posed as rigid 6-DoF fitting of a known paddle model, constrained by the 3DB wrist/hand frame (grip prior), the 2D paddle mask or keypoints, the calibrated camera and temporal continuity.

### Gaps
- No report of anyone using SAM 3D Objects on rackets, paddles, bats or similar hand-held sports implements, or on per-frame video of fast-moving objects, was found.
- No quantitative SAM 3D Objects results by object size or thinness were found in the pages read (the CVPR supplemental exists but was not opened).
- Whether SAM 3D Objects' predicted pose is metric and stable enough to track rigid motion across frames is undocumented.
