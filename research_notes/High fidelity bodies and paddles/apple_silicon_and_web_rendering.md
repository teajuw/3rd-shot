# Apple Silicon practicalities for human mesh recovery, and web rendering of animated body meshes plus paddle (as of November 2026)

Research date: 2026-11-12. Roughly 30 search/fetch calls. Most Apple Silicon evidence comes from community forks and pull requests, not upstream repos; each item says which. Anything I could not source is in Gaps, not in Cited Findings.

## 1. Which human mesh recovery models run on Apple Silicon (MPS / MLX / Core ML), blockers, workarounds, speeds

### Takeaway
Two model families have documented, measured Apple Silicon paths: SAM 3D Body (an MLX port and an MPS-optimised fork, roughly 0.2-0.5 s per person-frame) and GVHMR (two MPS forks, about 3.9 frames/s on an M1 Max, with DPVO the only CUDA-only part). InstantHMR, a small distilled model that outputs MHR, advertises a Core ML device option. I found no Apple Silicon reports for PromptHMR, CameraHMR, NLF, WHAM, TRAM, Multi-HMR, SMPLest-X, HaMeR or WiLoR.

### Cited Findings

SAM 3D Body (Meta, output is MHR)
- SAM 3D Body is a promptable single-image full-body mesh recovery model; Meta states both it and the MHR mesh format are open source. Accepted at CVPR 2026 (poster and oral). — [Meta AI](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/); [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/37714)
- MLX port exists: "Add SAM 3D Body — monocular 3D body mesh on Apple Silicon" (mlx-vlm PR #922), merged 6 May 2026. It ports the DINOv3 ViT-H+ backbone (32 blocks, 1280 dim), the 6-layer decoder, and the MHR body model (forward kinematics, linear blend skinning, pose correctives, blend shapes). Output: 18,439-vertex mesh, 70 3D keypoints, camera parameters, from one RGB image plus a person box. — [mlx-vlm PR #922](https://github.com/Blaizzy/mlx-vlm/pull/922)
- Speed on an M3 Max (36 GB mentioned in a later comment): MLX median 488 ms per inference versus 667 ms for PyTorch on MPS (about 1.37x); with mesh render 701 ms versus 882 ms; model load 5.0 s versus 9.7 s. — [mlx-vlm PR #922](https://github.com/Blaizzy/mlx-vlm/pull/922)
- MLX port limitations: body only (hand model inference not supported, deferred); `scatter_add` done through a NumPy round trip; no detector or multi-person wrapper described. MLX MHR matches the PyTorch TorchScript reference within 0.0001 mm over all 18,439 vertices; end-to-end parity for the image pipeline is not reported. — [mlx-vlm PR #922](https://github.com/Blaizzy/mlx-vlm/pull/922)
- A second MLX porting effort is tracked at lablup/mlxcel issue #2432, pointing back to the mlx-vlm code. — [lablup/mlxcel #2432](https://github.com/lablup/mlxcel/issues/2432)
- MPS-optimised PyTorch fork (PoisedLabs/sam-3d-body PR #3, merged 25 Oct 2026). Blockers found in upstream code on MPS: autocast hardcoded to CUDA (so full precision on MPS); the MHR head forced to CPU (`self.mhr.to("cpu")`) because of one hardcoded `.to(torch.float64)` cast in the skeleton code; a 33-67 MB CPU-to-MPS meshgrid transfer on every batch; bfloat16-to-NumPy conversion failure; RoPE frequencies recomputed in every block. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3); [PoisedLabs issue #1](https://github.com/PoisedLabs/sam-3d-body/issues/1)
- Fixes in that fork: autocast enabled for MPS with bfloat16/float16 (camera projection and MHR geometry kept in float32); a native float32 MPS MHR path; sparse skinning in intermediate decoder layers; cached RoPE and meshgrid. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3)
- Measured on an unspecified Apple Silicon Mac, DINOv3 ViT-H/16+ at 512x512: `forward_step` per frame went from 455.2 ms to 348.7 ms at batch 1 (2.87 frames/s), 281.2 to 224.7 ms at batch 8 (4.45 frames/s), 261.0 to 216.8 ms at batch 16 (4.61 frames/s). MHR head: 94.3 ms to 16.7 ms at batch 1. Backbone at batch 8: 369.4 ms to 267.9 ms per batch. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3)
- Accuracy cost of those changes: native MPS float32 MHR versus CPU float64 differs by at most 0.056 mm per vertex; end to end versus the fork's previous main, 2.33 mm mean joint difference and 2.38 mm per-vertex difference. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3)
- The headline "36-39 frames/s" figures in that fork are pipeline throughput where only 565 keyframes out of 6,362 frames were run through the model; they are not per-frame model speed. Batch 16 gave about 37.5-39 effective frames/s. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3); [PoisedLabs PR #4](https://github.com/PoisedLabs/sam-3d-body/pull/4)
- No Core ML port of SAM 3D Body was found; search only surfaced a proposal for Core ML on a different SAM model. — [avencera/sam3rs #13](https://github.com/avencera/sam3rs/issues/13)
- Related but different models, easy to confuse: SAM 3 (segmentation) has an MPS pull request for image inference with video/tracking remaining CUDA-only in one report, and a Triton dependency blocks it on M4 in another; SAM 3D Objects has MLX ports. — [facebookresearch/sam3 PR #627](https://github.com/facebookresearch/sam3/pull/627); [HF facebook/sam3 discussion 11](https://huggingface.co/facebook/sam3/discussions/11); [mlx-spatial](https://github.com/appautomaton/mlx-spatial)

InstantHMR (distilled from SAM 3D Body, output is MHR)
- InstantHMR recovers pose, shape, 70 2D/3D keypoints and a full mesh for every person in an image, outputs Meta's MHR, and ships ONNX graphs and PyTorch checkpoints with device options `cuda`, `tensorrt`, `cpu` and `coreml`. Model sizes are 20.0M to 32.2M parameters. — [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)
- Published speeds are 0.70-0.98 ms per person on an RTX 4070 with TensorRT and 4.60-5.49 ms on a Galaxy S23 NPU; no Apple Silicon benchmark is given. Medium model accuracy: PA-MPJPE 48.8 mm on EMDB-1, 40.0 mm on 3DPW. — [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)
- Licence: `nano`, `tiny`, `small`, `medium` are SAM License and non-commercial only (because of training data); `medium-commercial` permits commercial use under the SAM License conditions. — [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)

GVHMR (output is SMPL/SMPL-X, world-grounded)
- Fork westnt/gvhmr-pr claims end-to-end Apple Silicon (MPS) support including a moderngl mesh renderer; device auto-select CUDA then MPS then CPU, override with `GVHMR_DEVICE`. Only DPVO is CUDA-only. Mac-friendly camera options are `--camera vggt` and `--camera dust3r` (scene-aware, recover camera translation); the default moving-camera path SimpleVO recovers rotation only. Body models are registration-gated and not redistributed. — [westnt/gvhmr-pr](https://github.com/westnt/gvhmr-pr)
- A second macOS port (fontanierh/atelier PR #199) measured on an M1 Max (32-core GPU): a 300-frame 1600x1000 clip took 76 s video-to-motion (about 3.9 frames/s), peak memory 8.4 GB; checkpoints total 5.6 GB. — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- Patches that port needed: pytorch3d has no macOS wheel, so it uses pytorch3d's pure-Python rotation transforms and lazily loads k-NN (inference needs none of the compiled operators); MPS `conv2d` rejects a strided view so the HMR2 crop is made contiguous; SimpleVO adapted to pycolmap 4 and run on six CPU threads alongside the GPU stages. MPS versus CPU predictions differ by about 1.1 mm of translation. — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- Tuning results from that port: best was float32, unfused attention, batch 16, PyTorch 2.5.1. float16 slowed ViTPose by 20-25% and sped HMR2 by only 7%. Batches of 32/64 and float16 YOLO were no faster or slower. PyTorch 2.14.1 was slower and its OpenMP runtime conflicted with pycolmap's. — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)

WHAM, 4D-Humans and others
- WHAM's global trajectory step depends on an external SLAM such as DPVO or DROID-SLAM, so it inherits the CUDA constraint; the 4D-Humans repo documents only a Linux/CUDA setup. (From a search summary of the repos, not a Mac test.) — [4D-Humans](https://github.com/shubham-goel/4D-Humans); [westnt/gvhmr-pr](https://github.com/westnt/gvhmr-pr)
- General blocker pattern: research code that checks only for CUDA silently runs on CPU on a Mac; one unrelated project documents "no MPS branch, so every Apple Silicon Mac silently takes the CPU path". — [villa-apple-silicon](https://github.com/AndreasHad04/villa-apple-silicon)

### Inferences
- For four players at 50 frames/s, one minute of video is 12,000 person-frames. At the measured 0.22-0.49 s per person-frame for SAM 3D Body on Apple Silicon, that is about 44-98 minutes of compute per minute of video if every frame is processed; keyframing (as the PoisedLabs pipeline does) or a 25 frames/s pass with interpolation cuts this proportionally. GVHMR at about 3.9 frames/s per track on an M1 Max would be about 51 minutes per minute of video for four tracks. A Mac mini with a newer chip should be faster, but no source gives numbers for it.
- The DPVO blocker is irrelevant for a fixed pickleball camera: the static-camera path of GVHMR needs no visual odometry.
- InstantHMR via Core ML or ONNX is likely the fastest route on a Mac by a wide margin (tens of millions of parameters versus a ViT-H+), at lower accuracy; it is untested on Apple Silicon in its own documentation.
- Models with detectron2, xformers, or custom CUDA extensions (commonly PromptHMR, TRAM, Multi-HMR, HaMeR) would need the same class of patches as GVHMR (swap detector, pure-Python rotation utilities, drop SLAM); this is extrapolation, not a report.

### Gaps
- No Apple Silicon reports found for PromptHMR, CameraHMR, NLF, TRAM, Multi-HMR, SMPLest-X, HaMeR or WiLoR. One extended search across all of them returned nothing relevant; per-repo issue trackers were not individually checked.
- Upstream facebookresearch/sam-3d-body issue tracker was not searched directly for MPS threads; all SAM 3D Body evidence is from forks/ports.
- The PoisedLabs benchmark does not name the chip or RAM.
- Hand quality on Apple Silicon: the MLX port skips the hand decoder; whether the MPS fork runs the hand refinement path was not stated.
- No measured numbers for any of these on an M4/M4 Pro Mac mini specifically.

## 2. Memory and throughput for ViT-H-class models on a 64 GB machine; float16 and batch-size guidance; state of PyTorch MPS in 2026

### Takeaway
Memory is not the constraint on 64 GB: the one measured full pipeline peaked at 8.4 GB. Throughput is: ViT-H-class HMR runs at roughly 3-5 person-frames/s on MPS, batching beyond 8-16 gives little, and float16 on MPS is inconsistent (helps one model, hurts another), so each stage must be benchmarked.

### Cited Findings
- PyTorch 2.15 MPS notes: the `mps` device requires macOS 15.0 or later; float64 and complex128 are not supported because Metal has no double type; creating or moving a float64 tensor to MPS raises a TypeError rather than downcasting; `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help for double precision; such computation must run on CPU. — [PyTorch 2.15 MPS notes](https://docs.pytorch.org/docs/2.15/notes/mps.html)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` falls back to CPU kernels for unsupported operators (slower but lets models run). — [Hugging Face Apple Silicon docs](https://huggingface.co/docs/transformers/apple_silicon)
- Recent PyTorch releases moved many MPS operators from Apple's MPSGraph to hand-written Metal kernels (2.13, continuing in 2.14: index_add, index_select, argmin/argmax, conv3d, median, linspace, arange). — [PyTorch 2.14 release blog](https://pytorch.org/blog/pytorch-2-14-release-blog/); [dev.to on 2.13](https://dev.to/prabhakar_chaudhary_7afe4/pytorch-213-brings-flexattention-to-apple-silicon-and-its-faster-than-youd-expect-lpk)
- A SAM 2 MPS fork notes PyTorch >= 2.14.1 with TorchVision >= 0.29.1 provides native MPS pooling for Hiera and bicubic interpolation. — [sealad886/sam2 PR #1](https://github.com/sealad886/sam2/pull/1)
- Mixed precision on MPS: a 2024 PyTorch issue reported "BFloat16 is not supported on MPS"; by October 2026 a fork ran SAM 3D Body under `torch.amp.autocast` on MPS with bfloat16/float16. — [pytorch #141864](https://github.com/pytorch/pytorch/issues/141864); [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3)
- Measured float16 effect is modest and model-dependent: SAM 3D Body backbone 1.38x at batch 8 and whole step 1.20-1.31x (the pre-merge estimate of 1.8-2.0x was a projection that was not reached); GVHMR port saw ViTPose 20-25% slower and HMR2 7% faster under fp16. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3); [PoisedLabs issue #1](https://github.com/PoisedLabs/sam-3d-body/issues/1); [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- Batch size: SAM 3D Body per-frame time 348.7 ms (batch 1), 224.7 ms (batch 8), 216.8 ms (batch 16), so nearly flat beyond 8. GVHMR port: batch 16 best, 32 and 64 no faster. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3); [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- Memory: GVHMR full pipeline peak 8.4 GB on an M1 Max; SAM 3D Body MLX ran on a 36 GB M3 Max; a third-party note says SAM 3D Body "runs in 12GB VRAM" (CUDA context, weak source). — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199); [mlx-vlm PR #922](https://github.com/Blaizzy/mlx-vlm/pull/922); [mer.vin summary](https://mer.vin/2026/06/sam-3d-body-promptable-full-body-3d-mesh-from-one-image-cvpr-2026-mhr/)
- Version regressions are real: PyTorch 2.5.1 was faster than 2.14.1 for the GVHMR port. — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- MLX beat PyTorch MPS by 1.37x on the same model and chip. — [mlx-vlm PR #922](https://github.com/Blaizzy/mlx-vlm/pull/922)

### Inferences
- On 64 GB unified memory, several stages (detector, 2D pose, mesh recovery) can stay loaded at once and batch 16 is safe; there is no memory reason to quantise ViT-H-class models.
- Any pipeline code with float64 (rotation conversions, skeleton kinematics, camera maths) will either crash on MPS or get pushed to CPU; grep for `.double()` / `float64` is the first porting step, as the single cast in MHR's skeleton code showed.
- Because float16 and PyTorch version effects flip sign between models, pin the PyTorch version per stage and benchmark float32 versus autocast before committing.

### Gaps
- The PyTorch 2.15 MPS notes page fetched says nothing about float16/bfloat16 autocast, `torch.compile`, or memory watermark settings; I found no official statement of autocast support status on MPS for 2026 releases.
- One article suggested PyTorch 2.15 was at release-candidate stage when written; its stable status as of November 2026 was not confirmed. — context only: [mind-verse.de](https://www.mind-verse.de/news/pytorch-215-mps-fp8-metal-c2b4b68d7e00)
- No peak-memory figure for SAM 3D Body on Apple Silicon.

## 3. Body model licensing and what may be redistributed in a web page

### Takeaway
SMPL, SMPL-X (and by the same family terms SMPL-H/MANO) forbid any copying or distribution of the model, so embedding the template mesh and skinning weights in a web page is not permitted without a Meshcapade/Max Planck commercial licence. MHR (Apache-2.0 repo) and Anny (Apache-2.0 code, CC0 MakeHuman assets) are the redistributable choices, with one unresolved wrinkle: the MHR file bundled with SAM 3D Body is labelled SAM License by some sources.

### Cited Findings
- SMPL-X licence: use "for the sole purpose of performing non-commercial scientific research", non-commercial education and artistic projects; "any use for commercial, pornographic, military, or surveillance, purposes is prohibited"; "shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part" (one archive copy excepted); modification and derivative works are allowed; training commercial networks with it is prohibited; commercial licensing via Meshcapade, contact smpl@max-planck-innovation.de. — [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html)
- SMPL licence: "single-user, non-exclusive, non-transferable, free of charge" right for non-commercial research, education or artistic projects; same no-distribution wording; commercial use prohibited; the page links to a separate "SMPL-Body" licence whose terms were not retrieved. — [SMPL model licence](https://smpl.is.tue.mpg.de/modellicense.html)
- Meshcapade's commercial tiers: an R&D licence covers internal research only (not products, customer-facing use or redistribution); a Commercial Use licence covers integration into commercialised products; pricing is not public. — [Meshcapade body models](https://meshcapade.com/assets/body-models)
- Unverified secondary claim: Epic Games acquired Meshcapade (announced February 2026, closing April), leaving the future of SMPL commercial licensing unclear. — [dev.to / clad.you post](https://dev.to/arkadiuss/a-3d-body-scan-for-nine-cents-without-smpl-4me1)
- MHR: the repository states it is "licensed under the Apache Software License 2.0"; no separate asset licence is stated in the README. Assets are per-LOD FBX rigs (`lod?.fbx`, with identity and expression blendshapes), a TorchScript `mhr_model.pt` (LOD 1 only), a parameterisation file and `.npz` pose correctives. 7 levels of detail; 45 identity parameters (20 body, 20 head, 5 hands), 204 pose parameters, 72 expression parameters. — [facebookresearch/MHR](https://github.com/facebookresearch/MHR)
- Conflict on MHR asset licence: InstantHMR's README says "the MHR body model itself is Apache-2.0" but lists its shared `mhr-body.onnx` under the SAM License; a Hugging Face mirror of SAM 3D Body lists `assets/mhr_model.pt` under the SAM License; an independent project's licence spike calls the "Apache-2.0 code AND model" reading unverified. — [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR); [Blendcap/sam-3d-body mirror](https://huggingface.co/Blendcap/sam-3d-body); [QtMeshEditor #850](https://github.com/fernandotonon/QtMeshEditor/issues/850)
- SAM License as described by a mirror: commercial use permitted royalty-free, redistribution must carry the licence text, with trade-control and military/weapons use restrictions; the original repo is gated behind licence acceptance. — [Blendcap/sam-3d-body mirror](https://huggingface.co/Blendcap/sam-3d-body)
- MHR ships a `tools/mhr_smpl_conversion` tool (MHR to/from SMPL/SMPL-X) and a visualisation notebook; requires Python >= 3.11 and pymomentum >= 0.1.90; pip install is "experimental" and may not resolve on all platforms; macOS/arm64 support is not stated. — [facebookresearch/MHR](https://github.com/facebookresearch/MHR)
- Anny (NAVER LABS Europe): code Apache-2.0; MakeHuman-derived assets CC0 1.0; SOMA topology Apache-2.0; the optional "smplx" topology is non-commercial only and downloaded separately. Differentiable PyTorch model; default rig 104 bones, MakeHuman rig 163 bones; parameters cover age, height, weight, muscle mass, children to elderly. Warning: the default install may download non-commercial-only assets when needed. — [naver/anny](https://github.com/naver/anny); [NAVER LABS announcement](https://www.naverlabs.com/en/storyDetail/351)
- The `body-models` Python library auto-downloads ANNY and MHR on first use, while SMPL and SMPL-X require registration, which reflects the same licensing split. — [body-models on PyPI](https://pypi.org/project/body-models/0.14.0/)

### Inferences
- A self-contained page that performs skinning in the browser must embed the template vertices, faces and skinning weights, i.e. the body model itself. That is "copied/distributed" under the SMPL/SMPL-X terms, and the licence makes no public/private distinction, so even a private page shared with others is outside the licence. Shipping only pose/shape parameters is not restricted by these clauses but is useless without the model in the page.
- Baked per-frame vertex positions of a posed SMPL mesh are a derivative rather than the model file; the licence text fetched does not say whether such outputs may be shared. Treat as unclear and avoid for anything public.
- Practical route: do recovery in whatever model is most accurate, then convert to MHR (or Anny) for display, and ship the MHR LOD FBX-derived mesh under Apache-2.0 with the licence text. If the pipeline is SAM 3D Body or InstantHMR, the output is already MHR and no conversion is needed.
- To stay clear of the SAM License ambiguity, take the rig from the facebookresearch/MHR release (Apache-2.0) rather than from the copy bundled with SAM 3D Body checkpoints.

### Gaps
- SMPL-H and MANO licence pages were not fetched; I assume the same Max Planck family terms but did not verify.
- The "SMPL-Body" licence (reported elsewhere as Creative Commons for the SMPL body model) could not be confirmed: the Max Planck page did not include its text and the Meshcapade Notion page did not load.
- SKEL and ATLAS licences were not found in searches.
- The MHR LICENSE file and asset download manifest were not read verbatim; the asset licence conflict above is unresolved.
- The Epic/Meshcapade acquisition has no primary source here.
- MHR vertex counts per LOD and joint count are not on the README; only the 18,439-vertex figure (the LOD used by SAM 3D Body) is sourced, from the ports.

## 4. Rendering: compact formats, exporters, data size, and existing web viewers

### Takeaway
Ship one skinned mesh per player plus per-frame joint transforms and let three.js skin on the GPU: this is megabytes per minute, where per-frame vertices are hundreds of megabytes. No ready MHR-to-glTF exporter was found; SMPL-to-GLB exporters exist. The closest prior art for browser-side skinning of SMPL/MHR/Anny is the `body-models-viser` plugin, but viser, aitviewer and Rerun are server- or desktop-backed, not self-contained pages.

### Cited Findings
- SMPL(-X) animation is a set of parameters that a generic web viewer cannot parse directly; it must be converted to a standard skinned or baked format first. — [XRViewer docs](https://xrviewer.readthedocs.io/en/latest/getting_started/basic_usage.html)
- Exporters: a ComfyUI `SMPLToGLB` node converts SMPL motion `.npz` to an animated GLB with skinned mesh and skeletal animation for Blender and three.js (vendor documentation, untested); the Rust crate `smpl-core` has a glTF/GLB codec. — [ComfyAI SMPLToGLB](https://comfyai.run/documentation/SMPLToGLB); [smpl-core gltf codec](https://docs.rs/smpl-core/0.8.0/src/smpl_core/codec/gltf.rs.html)
- MHR's repo lists FBX rigs and a TorchScript model but no glTF/GLB export tool. A third-party summary claiming FBX and glTF export is AI-generated and unverified. — [facebookresearch/MHR](https://github.com/facebookresearch/MHR); contradicted/unverified: [emergentmind topic page](https://api.emergentmind.com/topics/momentum-humanrig-mhr)
- The GVHMR macOS port writes a `motion.json` of parent-relative rotation deltas, root positions and a rest skeleton in a gravity-aligned world frame, with no GLB/BVH export. — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)
- In three.js, glTF skeletal animation is loaded with GLTFLoader and played with an AnimationMixer over `gltf.animations` clips; glTF needs a skeleton (and in practice a mesh bound to it) to carry skinning animation. — [three.js forum: export gltf with only animation](https://discourse.threejs.org/t/export-gltf-containing-only-animation-data/55776); [three.js forum: skeleton only](https://discourse.threejs.org/t/how-to-export-just-skeleton-with-animations/2364)
- Compression: `KHR_mesh_quantization` allows 8/16-bit vertex attributes and needs no decoder; three.js loads it from r111. `EXT_meshopt_compression` also compresses indices, morph targets and animation; three.js supports it from r122 via `GLTFLoader.setMeshoptDecoder`. — [meshoptimizer gltfpack README](https://github.com/flexcompute/meshoptimizer/blob/master/gltf/README.md); [cinevva guide](https://app.cinevva.com/guides/optimize-glb-for-web)
- One example: a 28.7 MB model became 22.8 MB with quantisation and 8.1 MB with meshopt (single unverified example). — [cinevva guide](https://app.cinevva.com/guides/optimize-glb-for-web)
- MHR at the SAM 3D Body level of detail has 18,439 vertices and 51,337 skinning entries (vertex-joint weights). — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3)
- `body-models-viser` (v0.0.1): evaluates body models in the browser inside a viser viewer; Python sends unskinned vertices, sparse weights, joint indices and bone transforms, and browser JavaScript applies skinning with all weights, specifically to avoid a fixed maximum number of bones per vertex. Supports unbatched SMPL, SMPL-H, SMPL-X, MHR, ANNY and SOMA. — [body-models-viser on PyPI](https://pypi.org/project/body-models-viser/0.0.1/)
- viser is a Python library with a web client (a running Python server feeds the page). aitviewer loads SMPL/-H/-X, MANO, FLAME sequences but is an OpenGL desktop viewer with a remote-server mode. — [viser on PyPI](https://pypi.org/project/viser/0.0.6); [aitviewer on PyPI](https://pypi.org/project/aitviewer/1.6.0/)
- InstantHMR's demo uses the Rerun viewer and can save `.rrd` recordings. — [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)
- A separate desktop/web tool, body-model-visualizer, exists for interactive SMPL-family inspection. — [mkocabas/body-model-visualizer](https://github.com/mkocabas/body-model-visualizer)
- GVHMR fork renders with moderngl on the Mac GPU (offline video, not web). — [westnt/gvhmr-pr](https://github.com/westnt/gvhmr-pr)

### Inferences
Data-size arithmetic (my calculation; 50 frames/s, 60 s = 3,000 frames; raw binary before base64, which adds about 33%):

| Representation | Per body per minute | Four bodies per minute |
|---|---|---|
| SMPL, 24 joint quaternions float32 + root translation (396 bytes/frame) | 1.19 MB | 4.8 MB |
| Same, quaternions as int16 (204 bytes/frame) | 0.61 MB | 2.4 MB |
| SMPL-X, 55 joint quaternions float32 + root (892 bytes/frame) | 2.68 MB | 10.7 MB |
| MHR, 204 pose parameters float32 (816 bytes/frame) | 2.45 MB | 9.8 MB |
| Per-frame vertices, SMPL 6,890 vertices float32 (82.7 kB/frame) | 248 MB | 992 MB |
| Per-frame vertices, MHR 18,439 vertices float32 (221 kB/frame) | 664 MB | 2.65 GB |
| Per-frame vertices, MHR, int16 quantised | 332 MB | 1.33 GB |

- Paddle as a rigid body is one position plus one quaternion per frame: 28 bytes/frame, about 84 kB per minute. Ball position only: 12 bytes/frame, about 36 kB per minute. Both are negligible.
- The static mesh is paid once per player: MHR 18,439 vertices with positions, normals, 4 joint indices and 4 weights is on the order of 1 MB before quantisation; a lower MHR level of detail would shrink it. (The 24- and 55-joint counts and 6,890-vertex count for SMPL/SMPL-X are from general knowledge of those models, not from a fetched source.)
- Joint-rotation skinning therefore fits comfortably in an embedded page (a few MB per minute, base64 or binary in a script tag); per-frame vertices do not.
- MHR complications for a plain glTF skinned mesh: (1) 51,337 weights over 18,439 vertices averages 2.8 influences per vertex, so some vertices exceed the 4-influence limit of standard glTF/three.js skinning and must be pruned and renormalised, or skinned in custom JavaScript as body-models-viser does; (2) MHR's 204 pose parameters are not joint quaternions, so the exporter should run MHR forward kinematics offline and write per-joint local transforms per frame; (3) pose correctives are per-frame vertex offsets and would be dropped (small visual loss) or approximated; (4) per-player identity shape should be baked into the rest mesh and rest skeleton once, offline.
- Given the page constraint (CDN scripts only, embedded data), a custom compact format (typed arrays of joint transforms, decoded into `THREE.SkinnedMesh` bones each frame) is simpler than building a GLB with animation tracks, and avoids needing Draco/meshopt decoders; a GLB embedded as base64 and parsed with `GLTFLoader.parse` is the alternative if interchange with Blender matters.
- Subsampling animation to 25 frames/s with quaternion interpolation in the viewer halves every joint-rotation figure above.

### Gaps
- No MHR-to-glTF/GLB exporter was found; whether pymomentum itself can write glTF was not verified (not stated on the MHR README).
- three.js documentation for `SkinnedMesh` (4-influence limit, bone texture) and Draco specifics were not fetched; the 4-influence statement is from general knowledge.
- No source gives measured file sizes for SMPL/MHR motion as GLB; all sizes above are arithmetic.
- Rerun's web viewer embedding and whether an `.rrd` can be inlined into a single page were not researched.
- Number of joints in MHR and vertex counts of its lower levels of detail were not found.

## 5. Smoothing and retargeting tools for joint-rotation sequences on CPU

### Takeaway
SmoothNet is the documented learned option and runs on CPU; beyond that, the search returned no sourced material on one-euro filters, rotation interpolation or foot locking, so those remain design choices from general practice rather than cited findings.

### Cited Findings
- SmoothNet (ECCV 2022) is a temporal-only, plug-and-play refinement network that models long-range temporal relations of each joint independently; it supports 2D pose, 3D pose and SMPL parameters (sequence length x 72 pose values) and supports both GPU and CPU inference (tested on PyTorch 1.10.1). — [cure-lab/SmoothNet](https://github.com/cure-lab/SmoothNet)
- Reported gains: 86.88% lower acceleration error and 8.82% better MPJPE relative to raw estimator output. — [SmoothNet ECCV paper page](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1848_ECCV_2022_paper.php)
- Caveat from the authors: if the estimator already smooths over a sliding window, SmoothNet may over-smooth and not improve MPJPE further. It is integrated in MMPose (v0.25.0+) and MMHuman3D. — [cure-lab/SmoothNet](https://github.com/cure-lab/SmoothNet)
- Keyframe-plus-interpolation is used in practice for speed: the PoisedLabs pipeline ran SAM 3D Body on 565 adaptive keyframes of 6,362 frames. — [PoisedLabs PR #3](https://github.com/PoisedLabs/sam-3d-body/pull/3)
- GVHMR-style outputs are already temporally modelled and gravity-aligned world-frame motion (parent-relative rotations plus root positions). — [atelier PR #199](https://github.com/fontanierh/atelier/pull/199)

### Inferences
- SmoothNet's pretrained weights assume a 72-value SMPL pose layout; using it on MHR's 204 parameters would require retraining or smoothing in joint-rotation space with a classical filter instead.
- Per-frame models (SAM 3D Body, InstantHMR) need temporal smoothing; video models (GVHMR) need less, and stacking SmoothNet on them risks the over-smoothing the authors warn about. For fast racket-sport motion, over-smoothing swings is the main risk, so filter strength should be speed-adaptive.
- All of these operate on a few hundred numbers per frame; CPU cost is negligible next to mesh recovery.

### Gaps
- No sources retrieved for one-euro filtering of rotations, quaternion SLERP/Savitzky-Golay smoothing, foot-contact detection or inverse-kinematics foot locking tools, or for retargeting between SMPL and MHR skeletons other than MHR's own conversion tool. The single search on this returned only SmoothNet.
- No evidence on SmoothNet running under current PyTorch on Apple Silicon (its tested version is 1.10.1).

## 6. Appearance: cheap player-like colour versus full avatars, and compute on a Mac

### Takeaway
Full animatable Gaussian avatars are published as 12-45 minutes of training per person on a CUDA GPU, and none of the avatar papers found report Apple Silicon results; general Gaussian-splat training does now run natively on Metal. I found no sourced material on the cheap route (vertex or texture colours sampled from video), so that is an inference.

### Cited Findings
- 3DGS-Avatar: trains from one monocular video in under 30 minutes (about 30 GPU-minutes on ZJU-MoCap, 45 on PeopleSnapshot), renders at 50+ frames/s. — [3DGS-Avatar, arXiv](https://arxiv.org/html/2312.09228v3)
- STG-Avatar: about 25 minutes on an RTX 4090. SAGA: about 12 minutes training, 60+ frames/s rendering. — [STG-Avatar](https://arxiv.org/html/2510.22140); [SAGA](https://arxiv.org/html/2412.00845v1)
- "Animatable 3D Gaussian" targets multiple human avatars and reports results at 5 s and 30 s training budgets against InstantAvatar. — [Animatable 3D Gaussian](https://arxiv.org/pdf/2311.16482)
- Gaussian-splat training on Apple Silicon exists for static scenes: msplat (standalone Metal, no PyTorch) reports 7k iterations of full-resolution Mip-NeRF 360 scenes in about 90 s on an M4 Max (self-reported); MetalSplat is an MIT-licensed gsplat-style differentiable rasteriser with Metal kernels on PyTorch MPS; gsplat-mps is an AGPLv3 fork of gsplat 0.1.3; splat-apple reports MLX at about 38.5 iterations/s versus 10.6 for PyTorch MPS on an M4 (self-reported). — [msplat](https://github.com/rayanht/msplat); [MetalSplat v0.3.0](https://radiancefields.com/metalsplat-v0.3.0-trains-3dgs-and-2dgs-on-apple-silicon); [gsplat-mps](https://github.com/iffyloop/gsplat-mps); [splat-apple](https://github.com/ghif/splat-apple)
- Upstream gsplat MPS support was not confirmed by the search; Apple Silicon support comes from forks and separate projects. — [gsplat-mps](https://github.com/iffyloop/gsplat-mps)

### Inferences
- Avatar methods depend on the CUDA Gaussian rasteriser plus an accurate per-frame body fit and usually a subject filling much of the frame. Players in broadcast-style pickleball footage are small and motion-blurred, so avatar quality would be poor regardless of compute; and a web viewer for Gaussian avatars is far heavier than a skinned mesh.
- Porting an avatar method to a Mac would mean swapping its rasteriser for MetalSplat or similar; feasible in principle, unreported in practice.
- The cheap route fits the constraints: project the recovered mesh into a few video frames, sample median colours per body-part region (shirt, shorts, skin, shoes, hair/hat) using the body model's part segmentation (MHR ships a segmentation tool per its README), and store them as per-vertex colours or a tiny palette. This costs seconds of CPU and adds about 3 bytes per vertex once.
- A middle route is projecting video pixels into the body model's texture map over many frames and taking a per-texel median; it gives logos and stripes roughly but shows seams and needs good alignment.

### Gaps
- No sources on Gaussian or mesh avatar training on Apple Silicon for humans specifically.
- No sources retrieved on cheap texture/colour transfer onto SMPL/MHR from video, clothed-body models, or on Gaussian-avatar rendering in three.js. One search covered this whole area.
- Whether MHR ships a texture-coordinate layout usable for texture projection was not verified.
