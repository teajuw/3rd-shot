# Physics-based humanoid controllers that track reference human motion, and tooling for the simulated body (as of November 2026)

Reader note on sourcing: "Cited Findings" are facts read from pages fetched or returned by search in this research session (13 Nov 2026). Items marked "background, not re-verified" in Inferences/Gaps come from the researcher's prior knowledge of the literature (cutoff mid-2026) and were not re-read this session; treat them as leads to check, not as verified facts. Several figures came through search-result summaries rather than the primary page; these are flagged "(via search snippet)".

## 1. Method families and their state in 2026 (inputs, success on AMASS, noisy video motion, code/licence, compute, simulator, GPU need)

### Takeaway
The practical state of the art for "track any human motion in physics" is the universal-tracker line (PHC -> PHC+ / PULSE -> MaskedMimic -> ProtoMotions v3 general trackers), which reports 98.9-100% success on cleaned AMASS and is released with code and checkpoints; every one of these trains in NVIDIA GPU simulators (Isaac Gym / Isaac Lab / Newton), so training means renting NVIDIA GPUs. The one notable CPU/MuJoCo exception is Meta Motivo (a behavioural foundation model with a zero-shot tracking prompt), which is non-commercially licensed.

### Cited Findings

**PHC / PHC+ / PULSE (Luo et al.)**
- PHC is explicitly designed for "noisy input (e.g. pose estimates from video or generated from language)"; the repo ships a webcam video-to-pose demo (`scripts/demo/video_to_pose_server.py`) and a language-to-motion (MDM) demo. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- Released-model results on cleaned AMASS (11,313 sequences), columns success / global MPJPE (mm) / acceleration error: PHC 98.9% / 37.5 / 3.3; PHC-KP (keypoint-only input) 98.7% / 40.7 / 3.5; PHC+ (the tracker used in PULSE) 100% / 26.6 / 2.7; PHC-Prim (single primitive) 99.9% / 25.9 / 2.3; PHC-Fut (uses future frames) 100% / 25.3 / 2.5; PHC-X-Prim (SMPL-X body) 99.9% / 24.7 / 3.6. The README warns that evaluation protocol and Isaac Gym batch size / machine can change results. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- Simulator: Isaac Gym is the training framework; Isaac Lab is supported for inference via `scripts/eval_in_isaaclab.py` (added 21 Aug 2025); MuJoCo is used for building the XML body and for visualisation, not for training. Setup uses PyTorch with CUDA 11.6 (implies an NVIDIA GPU). — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- New SMPL motion is fed in by converting with `scripts/data_process/convert_data_smpl.py` (gravity must be along negative z). SMPL model files (`SMPL_NEUTRAL/MALE/FEMALE.pkl`) must be placed in `data/smpl`. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- SMPL-X humanoid support was released 30 Aug 2024; articulated fingers remain a TODO ("Add support for smplx/h (fingers!!!)") and several configs use `robot.freeze_hand=True`. Latest changelog entry seen: 21 Aug 2025. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- Licence: the README says to follow the licences of the upstream repositories it builds on; the fetched page did not display the licence terms. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- PULSE appendix, full AMASS (14,889 sequences): PULSE-X in Isaac Gym 99.5%, PHC-X in Isaac Lab 99.9%, PULSE-X in Isaac Lab 99.6% (via search snippet of a later paper's appendix). — [arXiv 2505.12278](https://arxiv.org/pdf/2505.12278)
- When PHC+ is used as a plain post-processor on a modern video estimator (GVHMR x PHC+), it improves height-velocity/jitter-type physical metrics but degrades pose accuracy and foot sliding relative to raw GVHMR: on EMDB2, GVHMR PA-MPJPE 40.95 / MPJPE 65.21 / FS 5.65 / HV 26.42 versus GVHMR x PHC+ 46.24 / 72.50 / 12.71 / 7.71; on AIST++, 53.43 / 79.05 / 11.54 / 10.19 versus 67.38 / 109.77 / 24.79 / 6.05. — [PhysHMR, arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)

**MaskedMimic and ProtoMotions (NVIDIA)**
- MaskedMimic on VR-style sparse tracking (head + hands) on the unseen AMASS test set: 98.1% success, versus PULSE 93.4%, ASE 37.6%, CALM 10.1% (via search snippet of the NVIDIA blog). — [NVIDIA developer blog](https://developer.nvidia.com/blog/unified-whole-body-control-for-physically-simulated-humanoids/)
- ProtoMotions is now "ProtoMotions3" (v3); news items on the repo dated 28 Oct 2026. Backends: Isaac Gym (Preview 4), Isaac Lab (badge 3.0), Newton (badge 1.0.0), plus MuJoCo on CPU for sim2sim testing; the Genesis backend has been removed. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- Characters: SMPL human (`--robot-name=smpl`), the SOMA animation character/skeleton, Unitree G1 and H1_2. SMPL-X, articulated hands and MHR are not mentioned on the repo page. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- Training cost: full AMASS (40+ hours of motion) trains "within 12 hours" on 4 A100s; the BONES run used 24 A100s with 13K motions per GPU. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- ProtoMotions docs (via search snippet): on 4x A100 with full AMASS, about 2 hours to 90% success and about 12 hours to 99% success. — [ProtoMotions AMASS SMPL tutorial](https://protomotions.github.io/tutorials/workflows/amass_smpl.html) (page returned 404 when fetched directly on 13 Nov 2026; figure is from the search index)
- A secondhand summary of a post by Chen Tessler reports full-AMASS tracker training in 24 hours plus 20 hours for the generative (MaskedMimic) model on 4 GPUs with 8k envs each; GPU model not given, source is not primary. — [community note](https://mike-gold.pages.dev/notes/x-bookmarks/ai/efficient-training-of-maskedmimic-with-ppo)
- Licence: ProtoMotions source is Apache-2.0; pretrained models are under the NVIDIA Open Model License unless a model card says otherwise. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- Motion tooling: PyRoki-based retargeting (one command for all of AMASS; earlier versions used Mink); ONNX export; an experimental C++ runtime with MuJoCo and Unitree G1 backends. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)

**Behavioural foundation model: Meta Motivo (FB-CPR)**
- Meta Motivo is described as "the first behavioral foundation model to control a virtual physics-based humanoid agent for a wide range of whole-body tasks"; it exposes a tracking prompt (`tracking_inference`) that turns a motion clip into a context vector, i.e. zero-shot tracking with no retraining. — [metamotivo GitHub](https://github.com/facebookresearch/metamotivo)
- Checkpoints: five small models metamotivo-S-1..5 (24.5M parameters each) and metamotivo-M-1 (288M parameters), on Hugging Face. Examples run with `device="cpu"`. Licence CC BY-NC 4.0. — [metamotivo GitHub](https://github.com/facebookresearch/metamotivo)
- It runs in HumEnv, a MuJoCo-backed, Gymnasium-compatible environment built on an SMPL humanoid; HumEnv is CC BY-NC 4.0; MoCap-initialised benchmarks require the user to prepare licensed SMPL/AMASS data. — [humenv GitHub](https://github.com/facebookresearch/humenv)
- Reported weakness (author of a downstream project, citing Meta): Motivo is weaker on very fast moves and ground moves; its input is body-relative so heading drifts on fast turns. — [Holo GitHub](https://github.com/Harp404/Holo)

**Robot-oriented universal trackers (2025-2026; same technique, robot bodies)**
- ProtoMotions ships Metatron-v1 (Unitree G1, on Hugging Face): one generative whole-body policy for tracking, steering, path following, sparse body goals, scene interaction and get-up; and a General Tracking Policy trained on BONES-SEED (~142K motions). No success rates are given on the repo page. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- UniTracker (2025): AMASS filtered by PHC to remove infeasible motions; a single policy tracks 8,100+ sequences on a 29-DoF G1 (via search snippet). — [arXiv 2507.07356](https://arxiv.org/html/2507.07356v3)
- Humanoid-GPT (June 2026): Transformer tracker trained on ~2B G1-retargeted frames, claims zero-shot tracking of hard dynamic motion (via search snippet). — [arXiv 2606.03985](https://arxiv.org/pdf/2606.03985)
- Other 2025-2026 tracker papers surfaced but not read: SONIC ("Supersizing Motion Tracking for Natural Humanoid Whole-Body Control") — [arXiv 2511.07820](https://arxiv.org/pdf/2511.07820); OmniTrack ("General Motion Tracking via Physics-Consistent Reference") — [arXiv 2602.23832](https://arxiv.org/pdf/2602.23832).

**Simulator platform support**
- Newton's installation guide lists macOS as CPU only (no GPU acceleration); NVIDIA Warp supports macOS arm64 (Apple Silicon) CPU only (via search snippet). — [Newton install guide](https://newton-physics.github.io/newton/1.2.0/guide/installation.html); [Warp compatibility](https://warp.readthedocs.io/en/latest/user_guide/compatibility.html)
- Isaac Gym-based code (example: vid2player3d) requires Linux, Python 3.7/3.8, Isaac Gym Preview 4 and CUDA devices. — [vid2player3d GitHub](https://github.com/nv-tlabs/vid2player3d)

### Inferences
- For four pickleball players and a handful of rallies, the cheapest first experiment is to run a released tracker on the converted motion rather than train one: PHC+/PHC-X checkpoints (Isaac Gym/Isaac Lab, rented NVIDIA GPU) or Meta Motivo (MuJoCo, plausibly runnable on the Mac's CPU since its examples use `device="cpu"` and MuJoCo itself supports macOS; not confirmed by the repo, which is silent on macOS).
- Training a universal tracker from scratch is a rented-GPU job on the order of 4 A100s for 12-24 hours (roughly 50-100 A100-hours) by the ProtoMotions figures; a single-GPU run of a distilled vision policy (PhysHMR) took about three days on one L40. Neither is feasible on Apple Silicon because Isaac Gym/Isaac Lab need NVIDIA GPUs and Newton/Warp are CPU-only on macOS.
- The GVHMR x PHC+ numbers show a real trade-off: a universal tracker as a post-filter removes floating/jitter but can add foot sliding and pose error on video motion, because the tracker balances the body in ways the noisy reference did not specify. Expect the same for SAM 3D Body output unless references are cleaned first or the tracker is fine-tuned on the clips.
- Background, not re-verified this session: the earlier families are DeepMimic (one policy per clip, phase-conditioned tracking reward; [arXiv 1804.02717](https://arxiv.org/abs/1804.02717)), AMP (adversarial style prior instead of explicit tracking; [arXiv 2104.02180](https://arxiv.org/abs/2104.02180)), ASE ([arXiv 2205.01906](https://arxiv.org/abs/2205.01906)) and CALM ([arXiv 2305.02195](https://arxiv.org/abs/2305.02195)) (latent skill spaces for reuse, not faithful tracking). AMP/ASE/CALM give style and reusable skills but are not designed to reproduce a specific clip frame-for-frame, which is consistent with their low sparse-tracking success in the MaskedMimic comparison (ASE 37.6%, CALM 10.1%). Per-clip DeepMimic-style tracking remains the most reliable way to reproduce one specific short clip with high fidelity; BeyondMimic-style per-motion pipelines in ProtoMotions are the modern descendant.
- Background, not re-verified: PULSE ([arXiv 2310.04582](https://arxiv.org/abs/2310.04582)) distils PHC+ into a latent action space for downstream tasks; MaskedMimic ([arXiv 2409.14393](https://arxiv.org/abs/2409.14393)) distils a full-body tracker into a masked-conditioning transformer-VAE policy accepting sparse joints, text, or scene constraints. Diffusion-policy controllers (e.g. PDP, CLoSD, UniPhys, Diffuse-CLoC) and model-based trackers (e.g. SuperTrack/world-model approaches, sampling-based MPC in MuJoCo) exist but were not surveyed this session.

### Gaps
- MaskedMimic's own full-body AMASS tracking table and official compute (paper appendix) were not reached; only the VR-tracking figure and secondhand training times were found.
- ProtoMotions v3 documentation pages (quickstart, AMASS tutorial) returned 404 on direct fetch, so the exact list and paths of released SMPL checkpoints (search snippets mention `data/pretrained_models/masked_mimic/smpl/last.ckpt` and an "SMPL AMASS general motion tracker") and their success rates are unverified. — snippet source: [ProtoMotions quickstart](https://protomotions.github.io/getting_started/quickstart.html)
- No sourced success numbers for diffusion/latent controllers or model-based methods were collected.
- PHC and Meta Motivo tracking success on AMASS as stated in the Motivo paper was not retrieved.
- Exact PHC licence terms not read.

## 2. Pretrained universal controllers: can a released checkpoint track new motion without retraining, and in which simulators?

### Takeaway
Yes: PHC/PHC+ (Isaac Gym, with Isaac Lab inference), ProtoMotions SMPL trackers/MaskedMimic (Isaac Gym / Isaac Lab / Newton), and Meta Motivo (MuJoCo via HumEnv, CPU) all accept unseen motion at inference. Only Meta Motivo is documented as CPU-run on MuJoCo; the others need an NVIDIA GPU in practice.

### Cited Findings
- PHC releases pretrained models (table in section 1) and a data-conversion script for arbitrary SMPL motion; evaluation runs in Isaac Gym, and since Aug 2025 in Isaac Lab. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- PHC also released an offline action dataset (PHC_Act, 10 Dec 2024) and retargeting documentation (9 Dec 2024) and Unitree H1/G1 support (8 Oct 2024). — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- Meta Motivo checkpoints do zero-shot tracking via a tracking prompt; runs in HumEnv (MuJoCo). — [metamotivo GitHub](https://github.com/facebookresearch/metamotivo); [humenv GitHub](https://github.com/facebookresearch/humenv)
- A working end-to-end example of exactly the reader's front half exists: "Holo" chains video -> SAM 3D Body (70 MHR keypoints in metres + camera) -> MuJoCo inverse-kinematics retarget onto the HumEnv/SMPL humanoid (keeps bone directions, applies the humanoid's bone lengths, about 2 cm per joint fit) -> Meta Motivo M-1 tracking prompt. Self-reported results: taekwondo side kick 9.1 cm per-joint posture error, no fall, 75 degrees heading error; kettlebell squats 5.3 cm, no fall, 16 degrees heading error. Hardware named: an RTX 4090 box (SAM 3D Body inference and EGL rendering). No repository licence stated; personal project. — [Holo GitHub](https://github.com/Harp404/Holo)
- ProtoMotions provides ONNX export with observation computation baked in and a C++ runtime with a MuJoCo backend, i.e. trained policies can be replayed in CPU MuJoCo. — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- vid2player3d (tennis) released code but not checkpoints: "the trained models are currently not available due to license issues of the video footage we used", so its demos cannot run. — [vid2player3d GitHub](https://github.com/nv-tlabs/vid2player3d)
- CRISP (ICLR 2026) shows off-the-shelf tracking of video-derived motion fails often without cleanup: motion-tracking failure rate fell from 55.2% to 6.9% on EMDB and PROX once scene geometry was reconstructed as clean, simulation-ready primitives. — [arXiv 2512.14696](https://arxiv.org/abs/2512.14696)
- FlowHMR (Oct 2026) reports that only 62.82% of GVHMR's outputs on ~4K internet videos (Wild-4K) can be tracked by a physics-based controller, versus 82.47% for its own physics-aware generator. — [arXiv 2610.03691](https://arxiv.org/abs/2610.03691)

### Inferences
- "Zero-shot" works for mocap-like motion (98.9-100% on AMASS) but video-derived motion from the wild is materially harder: independent 2025-2026 reports put raw trackability of state-of-the-art video estimates at roughly 45-63% of clips before any physics-aware cleanup (CRISP's 55.2% failure on scene-interaction benchmarks; FlowHMR's 62.82% success for GVHMR). Flat-court pickleball has no scene geometry problem, so it should sit at the easier end, but fast lunges, split-steps and occlusion-induced pose glitches are the likely failure cases.
- A realistic plan is two-stage: (1) zero-shot with a released tracker to see which segments fail; (2) fine-tune the tracker (or train per-clip policies) on the failing pickleball segments on a rented GPU.
- Mac route: Meta Motivo in MuJoCo, or an ONNX-exported ProtoMotions policy replayed in MuJoCo, are the two paths that plausibly run inference locally on Apple Silicon; both are unverified on macOS by their own docs.

### Gaps
- No source confirmed any of these checkpoints running on Apple Silicon.
- Whether a released ProtoMotions SMPL tracker checkpoint can be exported to ONNX and replayed in MuJoCo for the SMPL body (as opposed to the G1 robot) was not confirmed.
- Tracking success of released checkpoints specifically on racket-sport motion is unreported.

## 3. Building the simulated body (SMPL / SMPL-X tools; MHR equivalent or route; licensing)

### Takeaway
SMPLSim (BSD-3) is the standard tool that turns SMPL / SMPL-H / SMPL-X into a MuJoCo XML or Isaac Gym humanoid, and it is what PHC builds on; no MHR-native physics-humanoid generator was found. The practical routes for SAM 3D Body output are (a) Meta's official MHR -> SMPL/SMPL-X converter (Apache-2.0 tool, but it needs the non-commercial SMPL model files) then SMPLSim/PHC, or (b) skip SMPL and IK-retarget MHR keypoints directly onto an existing MuJoCo humanoid, as Holo and "Biomechanical 3D Body" do.

### Cited Findings
- SMPLSim is a pip-installable library that creates the SMPL humanoid in MuJoCo (v3+) and Isaac Gym, supports SMPL, SMPL-H and SMPL-X, writes XML model files for MuJoCo, includes a simple PPO example for MuJoCo, and is BSD-3-Clause licensed. — [SMPLSim GitHub](https://github.com/ZhengyiLuo/SMPLSim)
- The PHC/PHC+ SMPL humanoid has 24 rigid bodies, 23 actuated; PhysHMR runs it at zero-shape (mean) SMPL size with physics at 60 Hz and control at 30 Hz. — [PhysHMR, arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PHC requires the SMPL .pkl model files and uses SMPL to adjust humanoid height to avoid ground penetration at data load. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- MHR is Apache-2.0. It has 45 identity parameters (20 body, 20 head, 5 hands), 204 model parameters for joint angles and scalings, 72 FACS-style expression blendshapes, ships an FBX rig (`lod?.fbx`) and a TorchScript model, and integrates with PyMomentum's CPU solver. Its repo page mentions no physics, URDF, MJCF, simulation or robotics tooling. — [MHR GitHub](https://github.com/facebookresearch/MHR)
- MHR has 127 joints and 18.4k vertices versus SMPL's 24 joints and 6,890 vertices (via search snippet of a secondary summary). — [emergentmind summary of arXiv 2511.15586](https://www.emergentmind.com/papers/2511.15586)
- Official conversion tool (`tools/mhr_smpl_conversion`): supports SMPL/SMPL-X -> MHR, MHR -> SMPL/SMPL-X, and SAM 3D outputs (MHR) -> SMPL/SMPL-X. Two solvers: PyMomentum (CPU, hierarchical Gauss-Newton, robust, supports temporal tracking) and PyTorch (GPU-accelerated, edge + vertex loss, batch, frame-independent). `single_identity=True` (default) fits one shape for the whole sequence. Requires the official SMPL / SMPL-X model files. No accuracy or speed numbers are given; per-frame fitting errors can be returned. — [MHR conversion README](https://github.com/facebookresearch/MHR/blob/main/tools/mhr_smpl_conversion/README.md)
- Direct (no-SMPL) route in practice: Holo fits the HumEnv humanoid to 70 MHR keypoints with MuJoCo IK at about 2 cm per joint. — [Holo GitHub](https://github.com/Harp404/Holo)
- "Biomechanical 3D Body" (arXiv 2608.29928) adds a head to SAM 3D Body that regresses joint angles and segment scales for a MuJoCo humanoid model, and ports the backbone and MHR to JAX/Equinox to use MuJoCo MJX (via search snippet). — [arXiv 2608.29928](https://arxiv.org/html/2608.29928)
- "World-Coordinate Human Motion Retargeting via SAM 3D Body" uses SAM 3D Body as a frozen backbone and MHR as the intermediate representation for retargeting to a Unitree G1; a related paper argues SMPL's coupling of skeleton and surface complicates temporally consistent kinematics for retargeting (via search snippet). — [arXiv 2512.21573](https://arxiv.org/pdf/2512.21573); [arXiv 2604.01447](https://arxiv.org/html/2604.01447v2)
- ProtoMotions offers PyRoki-based retargeting and a non-SMPL human character (SOMA skeleton). — [ProtoMotions GitHub](https://github.com/NVlabs/ProtoMotions)
- Licensing: the MPI SMPL-X model licence covers non-commercial scientific research, education or artistic projects; commercial, military, surveillance and pornographic use are prohibited, as is using the software "to train methods/algorithms/neural networks/etc. for commercial purposes"; commercial licences are sold via Meshcapade. — [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html)
- Meshcapade lists paid commercial tiers for the body models (figures of EUR 1,500/year and EUR 25,000/year appeared in search snippets; page states pricing may change) and gates SMPL-fitted datasets behind an SMPL licence. — [Meshcapade body models](https://meshcapade.com/assets/body-models)
- HumEnv and Meta Motivo are CC BY-NC 4.0. — [humenv GitHub](https://github.com/facebookresearch/humenv); [metamotivo GitHub](https://github.com/facebookresearch/metamotivo)

### Inferences
- The lowest-friction path to a PHC/ProtoMotions-compatible body is: SAM 3D Body -> MHR params -> official MHR->SMPL converter (PyMomentum CPU solver should run on the Mac; `single_identity=True` gives one beta vector per player) -> SMPLSim to generate a per-player MJCF with those betas -> PHC/ProtoMotions data conversion. Each player gets their own body proportions; note PhysHMR and most pretrained trackers were evaluated on the mean-shape body, so per-player shapes may reduce zero-shot success unless the checkpoint was trained with shape variation.
- A licence-clean path avoids SMPL entirely: MHR (Apache-2.0) joint positions/segment lengths -> hand-written or scripted MJCF (capsules/boxes per segment, masses from segment volumes and standard anthropometric density) -> train a tracker in ProtoMotions (Apache-2.0 code) using MuJoCo/Newton. This forgoes every SMPL-body pretrained checkpoint (those are trained on AMASS, itself SMPL-licensed data) and means training from scratch on own data or permissively licensed mocap (ProtoMotions' BONES-SEED / SOMA route).
- Any checkpoint trained on AMASS with the SMPL body (PHC, PULSE, MaskedMimic SMPL, Meta Motivo) carries non-commercial constraints from SMPL/AMASS and, for Motivo/HumEnv, from CC BY-NC directly. For a personal/research digital twin this is fine; for a commercial product it is a blocker requiring a Meshcapade licence or the SMPL-free route. This is a reading of licence texts, not legal advice.
- MHR's 127-joint skeleton must be reduced to ~24 simulated bodies either way; fingers, face and twist joints are dropped or frozen.

### Gaps
- No published tool that generates MJCF/URDF directly from MHR identity parameters was found; "Biomechanical 3D Body" is the closest but its code availability and humanoid model were not checked.
- MHR->SMPL conversion error and runtime are not documented; needs a local measurement.
- Whether MJCF files derived from SMPL joint regressors legally inherit the SMPL licence is not addressed by any source found.
- A search snippet (a developer blog, unverified) claimed Epic Games acquired Meshcapade in 2026, which could change SMPL licensing; no primary confirmation was found. — [dev.to post](https://dev.to/arkadiuss/a-3d-body-scan-for-nine-cents-without-smpl-4me1)
- AMASS's own licence terms were not retrieved.

## 4. Hands and held objects (articulated hands, rigidly attached racket/paddle)

### Takeaway
Racket sports in physics simulation are almost always done by rigidly welding the racket to the wrist in place of the hand, not by simulating a grip; this is directly applicable to a pickleball paddle. Articulated fingers are still unfinished in the mainstream trackers (PHC lists them as TODO).

### Cited Findings
- SMPLOlympics (humanoid compatible with SMPL and SMPL-X; includes tennis and table tennis, 1v1 and 2v2 competitive modes): "each humanoid is equipped with a circular racket with a 15cm radius, positioned 35cm away from the wrist, replacing the right hand"; for table tennis the paddle has an 8 cm radius, positioned 12 cm from the wrist (via search snippet of the paper). — [arXiv 2407.00187](https://arxiv.org/pdf/2407.00187)
- SMPLOlympics sports list: golf, javelin, high jump, long jump, hurdling; competitive table tennis, tennis, fencing, boxing, soccer, basketball; baseline state and reward designs provided; project page smplolympics.github.io/SMPLOlympics. — [arXiv abstract 2407.00187](https://arxiv.org/abs/2407.00187)
- Vid2Player3D (SIGGRAPH 2023, NVIDIA): learns simulated tennis skills from broadcast video with a low-level imitation policy and a high-level motion-planning policy over a learned motion embedding; "correct estimated motion with physics-based imitation"; a hybrid control policy overrides erroneous parts of the embedding with high-level corrections; produces serves, forehands, backhands, topspin and slice, and two-player rallies with simulated racket and ball dynamics. — [Vid2Player3D project page](https://research.nvidia.com/labs/toronto-ai/vid2player3d/)
- Vid2Player3D's video annotation pipeline: YOLOv4 (detection/tracking), ViTPose (2D keypoints), HybrIK (3D pose/mesh), a foot-contact detector, GLAMR (global root trajectory optimisation), and a tennis court line detector. Runs on Isaac Gym Preview 4 (environment adapted from ASE), requires SMPL model files; low-level policy training code released 28 Nov 2023; trained models not released. — [vid2player3d GitHub](https://github.com/nv-tlabs/vid2player3d)
- "Strategy and Skill Learning for Physics-based Table Tennis Animation" (2024): two-level controller (strategy picks skills from agent/opponent/ball state; skill-level controller executes strokes); evaluated agent-vs-agent and human-vs-agent in VR (via search snippet). — [arXiv 2407.16210](https://arxiv.org/pdf/2407.16210)
- PHC: SMPL-X body supported, fingers a TODO; `freeze_hand=True` in configs. — [PHC GitHub](https://github.com/ZhengyiLuo/PHC)
- Dexterous whole-body work exists on the PULSE-X/PHC-X line ("Emergent Active Perception and Dexterity of Simulated Humanoids from Visual Reinforcement Learning", which reports PULSE-X/PHC-X numbers in Isaac Gym and Isaac Lab). — [arXiv 2505.12278](https://arxiv.org/pdf/2505.12278)
- Real-robot racket work: CyboRacket (2026) runs onboard ball-trajectory prediction with a pretrained whole-body motion executor on a physical humanoid (via search snippet). — [arXiv 2603.14605](https://arxiv.org/pdf/2603.14605)

### Inferences
- For pickleball, copy the SMPLOlympics/Vid2Player3D recipe: add a paddle rigid body (pickleball paddles are roughly 40 cm long and 20 cm wide; verify against equipment rules in the adjacent research thread) fixed to the dominant wrist body in the MJCF, optionally replacing the hand geometry. Because the paddle is welded, the tracker never has to learn grasping, and existing 24-body SMPL trackers apply with only a changed hand link (extra mass/inertia may require fine-tuning).
- Video cannot recover wrist roll/paddle face angle reliably; Vid2Player3D's answer was to let a high-level policy override the wrist/arm so that the ball actually goes to the target. Expect the same need if the twin must reproduce shot outcomes, not just body motion.
- Two-handed backhands with a welded paddle require either ignoring the off-hand contact or adding a soft constraint; no source describing a solution was read (Vid2Player3D's page lists two-handed backhands as a supported style but gives no mechanism).

### Gaps
- How Vid2Player3D attaches the racket and handles the wrist is in its paper/supplement, not in the pages fetched.
- No pickleball-specific physics-humanoid work was found in this session's searches (searches were not pickleball-targeted; an adjacent researcher may cover this).
- No physics-based badminton character controller surfaced.
- MaskedMimic/ProtoMotions object-in-hand support was not confirmed.

## 5. Physics-based refinement of video motion specifically (PhysHMR, PhysPT, SimPoE, PhysDiff, 2024-2026 work)

### Takeaway
Reported gains are mainly in physical plausibility (foot sliding, floating/height velocity, jitter, trackability), usually at flat or slightly worse joint accuracy; the best 2025-2026 systems (PhysHMR, FlowHMR, CRISP) match kinematic accuracy while cutting artefacts, but cost days of NVIDIA GPU training and none is confirmed to ship a ready-to-run checkpoint.

### Cited Findings
- PhysHMR (arXiv Oct 2025; Feng, Huang, Wang, Gu, Liu): learns a vision-to-action policy that controls a simulated humanoid directly from monocular video; lifts 2D keypoints to 3D rays in global space ("pixel-as-ray"); distils action supervision from a PHC+ expert trained on AMASS, combined with RL. Isaac Gym, SMPL humanoid (24 bodies, 23 actuated), 1,536 parallel envs on a single NVIDIA L40, converging after about three days. — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysHMR EMDB2 results (PA-MPJPE / W-MPJPE / MPJPE / foot sliding / height velocity; lower is better): TRAM 35.51 / 148.05 / 56.74 / 11.76 / 22.97; GVHMR 40.95 / 228.67 / 65.21 / 5.65 / 26.42; PhysPT (on CLIFF) 48.40 / 762.78 / 77.00 / 11.02 / 6.54; GVHMR x PHC+ 46.24 / 193.01 / 72.50 / 12.71 / 7.71; PhysHMR 39.34 / 189.26 / 55.48 / 4.60 / 5.04. — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysHMR AIST++ results (same columns): TRAM 50.18 / 189.44 / 76.30 / 23.18 / 7.93; GVHMR 53.43 / 175.64 / 79.05 / 11.54 / 10.19; PhysPT 70.72 / 260.07 / 108.57 / 13.68 / 3.71; GVHMR x PHC+ 67.38 / 193.23 / 109.77 / 24.79 / 6.05; PhysHMR 50.40 / 187.42 / 63.94 / 9.14 / 3.10. — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysHMR ablation: success rate with PPO only 65.5%, distillation only 72.0%, PPO + distillation 88.4% (H36M). User study (26 participants): PhysHMR preferred 66.3%, PhysPT 19.9%, GVHMR x PHC+ 13.8%. — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysHMR limitations: a real-to-sim gap persists; no human-scene interaction (EMDB2 skateboard and stairs sequences excluded); offline because the GVHMR image encoder is not real-time; paper does not state a code release. — [arXiv 2510.02566](https://arxiv.org/html/2510.02566v1)
- PhysPT (CVPR 2024; Zhang, Kephart, Cui, Ji): a Transformer encoder-decoder trained self-supervised on mocap only, with force, contact and Euler-Lagrange losses; applied on top of any kinematic estimator without fine-tuning; uses no physics engine. — [arXiv 2404.04430](https://arxiv.org/abs/2404.04430)
- FlowHMR (arXiv Oct 2026): video-conditioned flow-matching motion generator, post-trained with Group Relative Policy Optimization using a video-fidelity reward and a reward for being trackable by a physics-based controller; on Wild-4K (~4K internet videos) physical tracking success 82.47% versus 62.82% for GVHMR; code repo listed at github.com/flowhmr/flowhmr. — [arXiv 2610.03691](https://arxiv.org/abs/2610.03691)
- CRISP (ICLR 2026): recovers simulatable human motion plus planar-primitive scene geometry from monocular video and validates with an RL humanoid tracker; tracking failure 55.2% -> 6.9% on EMDB/PROX; 43% faster RL simulation throughput; project page crisp-real2sim.github.io/CRISP-Real2Sim. — [arXiv 2512.14696](https://arxiv.org/abs/2512.14696)
- PBHC / KungfuBot (2025) uses a four-step pipeline for video motion destined for a Unitree G1: SMPL motion estimation from monocular video, physics-based motion filtering, contact-aware motion correction, retargeting (via search snippet). — [arXiv 2506.12851](https://arxiv.org/html/2506.12851v2)
- Vid2Player3D corrects broadcast-video tennis motion by physics-based imitation, then learns a motion embedding from the corrected motion. — [Vid2Player3D project page](https://research.nvidia.com/labs/toronto-ai/vid2player3d/)
- GRIP (CVPR 2026; IMUs + pressure insoles, not video) reports physics-based approaches suppress foot sliding more effectively than kinematic ones (via search snippet). — [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Hori_Ground_Reaction_Inertial_Poser_Physics-based_Human_Motion_Capture_from_Sparse_CVPR_2026_paper.pdf)
- PhyMotion (2026) retargets video-recovered SMPL-X motion to a MuJoCo humanoid to score kinematic plausibility, contact/balance consistency and dynamic feasibility (as a reward for video generation, not a reconstruction method) (via search snippet). — [arXiv 2605.14269](https://arxiv.org/pdf/2605.14269)

### Inferences
- Quantified benefit to expect from simulation-in-the-loop refinement on video motion, by PhysHMR's tables: height-velocity/floating artefacts cut roughly 3-5x versus kinematic estimators (EMDB2 HV 26.42 -> 5.04; AIST++ 10.19 -> 3.10), foot sliding modestly reduced versus the best kinematic method (5.65 -> 4.60; 11.54 -> 9.14), joint accuracy about equal or better. Naive post-hoc tracking (GVHMR x PHC+) gets the floating fix but loses 6-30 mm MPJPE and doubles foot sliding.
- PhysPT-style learned physics priors need no simulator and no RL (so are Mac-friendly in principle) but are clearly less accurate in global trajectory in this comparison (W-MPJPE 762.78 on EMDB2), and they produce plausible-looking motion plus force estimates, not a controllable simulated body.
- If the goal is a controllable simulated player (not just cleaned kinematics), a tracker policy is unavoidable; the refinement literature's value is as a pre-filter that raises the fraction of clips the tracker can follow.
- Background, not re-verified: SimPoE ([arXiv 2104.00683](https://arxiv.org/abs/2104.00683)) trains a video-conditioned RL policy in MuJoCo with residual forces; PhysDiff ([arXiv 2212.02500](https://arxiv.org/abs/2212.02500)) projects diffusion-generated motion through a physics tracker during sampling to cut penetration, floating and sliding. Neither was re-read this session, so no numbers are given.

### Gaps
- PhysPT's own foot-sliding/acceleration numbers and code status were not retrieved.
- PhysHMR code/checkpoint availability unconfirmed; FlowHMR repo existence is taken from the arXiv listing, contents and licence unchecked; CRISP code not confirmed.
- No evaluation of any of these on SAM 3D Body / MHR input or on racket sports.
- FlowHMR and CRISP simulators and compute are not stated on their abstract pages.

## 6. Lighter alternatives that do not need RL (trajectory optimisation with contact, differentiable physics, inverse dynamics with foot-contact constraints)

### Takeaway
Non-RL options exist and run on CPU MuJoCo (hence on the Mac), but the sources found are either older research code (2021-2022) or general toolboxes rather than maintained video-to-physical-human pipelines; they are enough for per-clip cleanup and force/torque estimates, not for an interactive, perturbation-robust controllable body.

### Cited Findings
- "Trajectory Optimization for Physics-Based Reconstruction of 3d Human Pose from Monocular Video" (2022) estimates a physical human model and motor-control trajectories so the simulated motion matches the video, with contact-, mass- and inertia-aware optimisation (via search snippet). — [arXiv 2205.12292](https://arxiv.org/abs/2205.12292)
- "Differentiable Dynamics for Articulated 3d Human Motion Reconstruction" (2022) and "Physics-based Human Motion Estimation and Synthesis from Videos" (2021) are earlier differentiable-physics / physics-optimisation reconstructions (titles surfaced by search; not read). — [arXiv 2205.12256](https://arxiv.org/pdf/2205.12256); [arXiv 2109.09913](https://arxiv.org/pdf/2109.09913)
- MuJoDiCo is an MJX-based toolbox for trajectory optimisation through differentiable contact, with algorithms for contact-rich systems and a suggested `contact_smoothing=1e-5` starting point (via search snippet; general toolbox, not a human-video pipeline). — [mujodico GitHub](https://github.com/wei-chen-li/mujodico)
- "Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting" (2026) is a recent contact-implicit trajectory-optimisation retargeting method (title surfaced by search; not read). — [arXiv 2608.03116](https://arxiv.org/pdf/2608.03116)
- Kinematic IK retargeting onto a MuJoCo humanoid without any learning reaches about 2 cm per-joint fit from SAM 3D Body keypoints (Holo). — [Holo GitHub](https://github.com/Harp404/Holo)
- PhysPT is the learned, simulator-free analogue: physics-inspired losses (force, contact, Euler-Lagrange), plug-in on any kinematic estimator, infers forces. — [arXiv 2404.04430](https://arxiv.org/abs/2404.04430)
- PBHC's "physics-based motion filtering" and "contact-aware motion correction" steps are non-RL preprocessing stages placed before tracking (via search snippet). — [arXiv 2506.12851](https://arxiv.org/html/2506.12851v2)
- Vid2Player3D's pre-physics cleanup is likewise non-RL: foot-contact detection plus GLAMR global root trajectory optimisation. — [vid2player3d GitHub](https://github.com/nv-tlabs/vid2player3d)
- A search for a 2025 MJX-based, RL-free, video-to-SMPL physical reconstruction pipeline found none. — search performed 13 Nov 2026; closest hits listed above

### Inferences
- When the lighter options are enough: (a) the deliverable is cleaned kinematics (no foot skate, no ground penetration, plausible root trajectory) for replay or visualisation; (b) one wants joint torques / ground-reaction estimates via inverse dynamics on a scaled MuJoCo model; (c) clips are short and few (four players, selected rallies), so per-clip optimisation time is acceptable. Kinematic IK with foot-contact locking plus inverse dynamics in MuJoCo is entirely CPU work and fits a 64 GB Apple Silicon machine.
- When they are not enough: the body must stay upright under closed-loop simulation, respond to perturbations, be steered to new shots, or interact with a simulated ball. Open-loop torque trajectories from trajectory optimisation fall over within a fraction of a second without a feedback policy; that is the gap RL trackers fill.
- A sensible staged plan on the reader's hardware: Mac-only stage = MHR->SMPL (or direct IK) + contact-aware kinematic cleanup + MuJoCo inverse dynamics + (optional) Meta Motivo zero-shot tracking on CPU; rented-GPU stage = PHC+/ProtoMotions zero-shot evaluation, then fine-tuning on pickleball clips (order of tens of A100-hours by ProtoMotions' full-AMASS figures; fine-tuning on a few minutes of motion should cost far less, but no source quantifies it).
- Background, not re-verified: sampling-based predictive control (MuJoCo MPC / MJPC-style) can track mocap references on CPU without learning and runs on macOS, and differentiable simulators (MJX, Warp, Brax, DiffTaichi-style) allow gradient-based fitting; no sourced humanoid full-body video-tracking success numbers were gathered for these.

### Gaps
- No quantitative head-to-head between trajectory optimisation and RL trackers on video-derived human motion was found.
- Runtime per clip for the 2022 trajectory-optimisation methods, and whether their code is released, were not checked.
- MJX on Apple Silicon (JAX has no Metal GPU backend of production quality as far as the researcher recalls; unverified) and MuJoCo Warp on macOS (Warp is CPU-only there per its docs) mean differentiable/GPU-batched MuJoCo variants give no GPU speedup on the Mac; exact performance unmeasured.
- No maintained 2025-2026 open-source "video -> contact-constrained inverse dynamics" human pipeline was identified; biomechanics tools (e.g. OpenSim/OpenCap-style, "Biomechanical 3D Body") were surfaced but not investigated.
