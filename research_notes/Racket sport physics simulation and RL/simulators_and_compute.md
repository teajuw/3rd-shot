# Simulators, RL stacks and compute for simulated humanoid athletes (as of 13 Nov 2026)

Reader context: Apple Silicon Mac (M5 Pro class, 64 GB), macOS; open to renting cloud GPUs. Target: four humanoid pickleball players + paddles + ball, first driven by motion recovered from video, later by RL agents.

Note on sourcing: most facts below come from primary pages fetched this session (MuJoCo docs/changelog, GitHub READMEs, Unity release page). Where a fact came only from a search-result summary or a secondary site, it is flagged "(search summary)" or "(secondary)". Items I could not source this session are in Gaps, some marked as "unverified background knowledge" so the writer can decide whether to use them with a caveat.

## 1. MuJoCo, MJX, MuJoCo Warp, MuJoCo Playground

### Takeaway
MuJoCo (CPU, C engine) is the one first-class simulator that runs natively on macOS and is actively released (4.0.1 on 11 Nov 2026); it now has a built-in multi-threaded batch/rollout API. The GPU variants are effectively NVIDIA-only in practice: MuJoCo Warp / MJX-Warp target NVIDIA GPUs (Warp on macOS is CPU-only, no Metal), and MJX-JAX lists "Apple Silicon" as a platform but Apple-GPU JAX (jax-metal) appears unmaintained, so on a Mac MJX-JAX realistically means CPU.

### Cited Findings
- MuJoCo release cadence: 3.5.0 (12 Feb 2026), 3.6.0 (10 Mar 2026), 3.7.0 (14 Apr 2026), 3.8.1 (11 May 2026), 3.9.0 (27 May 2026) (search summary of changelogs) — [MuJoCo 3.9.0 changelog source](https://mujoco.readthedocs.io/en/3.9.0/_sources/changelog.rst.txt); 3.16.0 (14 Oct 2026), 3.17.0 (22 Oct 2026), 4.0.0 (6 Nov 2026), 4.0.1 (11 Nov 2026) — [MuJoCo changelog, latest](https://mujoco.readthedocs.io/en/latest/changelog.html)
- MuJoCo 4.0.0 adds "speculative contacts", which prevent fast-moving convex geoms from tunnelling through thin obstacles (directly relevant to a fast ball hitting a thin paddle or net), plus elasto-plasticity; removes Python 3.10 support; splits the Python distribution into `mujoco-core`, `mujoco-studio`, `mujoco`. The RK4 integrator change and several deformable changes are breaking — [MuJoCo changelog, latest](https://mujoco.readthedocs.io/en/latest/changelog.html)
- 4.0.0 also notes "MuJoCo Warp still requires `solver="CG"` for such models" (a class of models in that release note), i.e. Warp feature parity with the C engine is not complete — [MuJoCo changelog, latest](https://mujoco.readthedocs.io/en/latest/changelog.html)
- 4.0.1 refers to "the 4.0.0 macOS and Linux archives", confirming macOS binaries ship for the 4.0 line — [MuJoCo changelog, latest](https://mujoco.readthedocs.io/en/latest/changelog.html)
- MuJoCo 3.17.0 added a batch API: "many simulations of one model stepped on a thread pool of its own", bit-identical to a plain loop at any thread count; Python `mujoco.batch.Batch(model, nsim)` with `step`, `forward`, `reset`, `rollout`, with `rollout` run with the GIL released. This is the native route to CPU-parallel environments on a Mac — [MuJoCo changelog, latest](https://mujoco.readthedocs.io/en/latest/changelog.html)
- MuJoCo Warp was declared officially released in MuJoCo 3.5.0 (12 Feb 2026) (search summary) — [MuJoCo 3.5.0 changelog source](https://mujoco.readthedocs.io/en/3.5.0/_sources/changelog.rst.txt)
- MuJoCo Warp README: "MuJoCo Warp requires an NVIDIA GPU for fast simulation but supports CPU for development and debugging." Maintained "by Google DeepMind and NVIDIA as part of the Newton project". Differentiability "not yet available"; flex is "experimental". macOS is not mentioned — [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp)
- NVIDIA Warp itself: macOS arm64 (Apple Silicon) supported with CPU-only acceleration; no Metal; Intel Macs discontinued (use Warp 1.9.x or earlier); GPU builds unavailable on macOS because CUDA is unavailable there (search summary) — [Warp compatibility and support](https://nvidia.github.io/warp/latest/user_guide/compatibility.html)
- MJX docs: MJX-JAX runs on "Nvidia and AMD GPUs, Apple Silicon, and Google Cloud TPUs"; MJX-Warp "optimizes performance specifically for NVIDIA GPUs". MJX-JAX is recommended for "small scenes" and "roughly supports gradients"; MJX-Warp "does not support automatic differentiation" — [MJX docs (stable)](https://mujoco.readthedocs.io/en/stable/mjx.html)
- MJX docs performance: MJX-Warp Humanoid throughput 3.35M steps/s (pure Warp), 2.96M (JAX FFI), 2.67M (WARP_STAGED); hardware not named on the page as fetched. MJX-JAX single-humanoid figures "650K, 1.8M, 950K and 2.7M" across four timed architectures: Apple M3 Max, 64-core AMD 3995WX, Nvidia A100, 8-chip v5 TPU. The fetch tool could not map numbers to devices unambiguously; the Apple M3 Max and AMD entries are CPU MuJoCo runs (native C engine, multi-threaded), not MJX on an Apple GPU. If order matches, an M3 Max does about 650K humanoid steps/s on CPU with native MuJoCo — [MJX docs (stable)](https://mujoco.readthedocs.io/en/stable/mjx.html)
- MJX-JAX limits: "MJX-JAX can be 10x slower than MuJoCo" for a single scene; works best with "thousands or tens of thousands of scenes in parallel"; throughput falls faster than MuJoCo as humanoids per scene increase (relevant: four humanoids in one scene); convex mesh collision needs roughly ≤200 vertices (mesh–primitive) or <32 vertices (convex–convex); ELLIPSOID and CYLINDER only collide with primitives in MJX-JAX. MJX-Warp "fully supports mesh collisions" — [MJX docs (stable)](https://mujoco.readthedocs.io/en/stable/mjx.html)
- JAX on Apple GPU: Apple's page still describes a Metal plug-in for JAX, labelled experimental; a Dec 2025 Apple forum post reports JAX maintainers closed issues citing no active development on jax-metal; other posts call it effectively broken (forum reports, not an official deprecation) — [Apple: Accelerated JAX on Mac](https://developer.apple.com/metal/jax/); [Apple forum: Is Jax for Apple Silicon still supported](https://developer.apple.com/forums/thread/815139)
- A community port "MJX-MLX" replaces JAX with Apple MLX for Metal-accelerated MJX on M-series Macs; not official, maturity unknown (search summary) — [RobotFlow-Labs/Mujoco-mlx](https://gh.r-mj.com/RobotFlow-Labs/Mujoco-mlx)
- mjlab (Isaac-Lab-style API on MuJoCo Warp, used for BeyondMimic-style G1 motion imitation): "requires an NVIDIA GPU for training (via MuJoCo Warp)"; macOS supported only for evaluation, "significantly slower"; example is G1 motion imitation with 4096 environments on an NVIDIA GPU (from a fork's README; verify upstream) — [mjlab (fork README)](https://github.com/Vertax42/mjlab)
- MuJoCo Playground (MJX-based) paper claims policies train in "minutes"; a separate MJX+Brax PPO humanoid walking policy (REEM-C) trained in about 56 minutes on an RTX 4090 with a collision-reduced model; a later paper reports humanoid locomotion in about 15 minutes on one RTX 4090 with FastSAC/FastTD3 (search summaries) — [MuJoCo Playground paper](https://arxiv.org/abs/2502.08844); [REEM-C MJX paper](https://www.arxiv.org/pdf/2407.05148); [Learning Sim-to-Real Humanoid Locomotion in 15 minutes](https://arxiv.org/html/2512.01996v1)

### Inferences
- For the first project phase (replaying motion recovered from video on four humanoids, kinematic or PD-tracked, plus a ball), native MuJoCo on the Mac is sufficient and is the only mainstream option with first-class macOS binaries and current releases.
- On the Mac, the practical parallel-RL route is native MuJoCo with `mujoco.batch` (CPU threads), not MJX/Warp. If the ~650K steps/s M3 Max figure is right for a single plain humanoid, an M5 Pro should be in the same order of magnitude for simple humanoid scenes; a four-SMPL-humanoid scene with paddles and ball contact will be several times slower per step. That is enough for single-clip motion tracking and small-scale RL, but 1–2 orders of magnitude short of a single NVIDIA GPU running MuJoCo Warp (millions of steps/s).
- Because mjlab, MuJoCo Warp and Newton share the MJCF model format with CPU MuJoCo, a model built and debugged on the Mac can be moved to a rented NVIDIA GPU for heavy training without changing simulator family. This is the strongest argument for standardising on MuJoCo.
- MuJoCo 4.0's speculative contacts reduce the main ball-physics risk (tunnelling of a fast, small ball through a thin paddle). Aerodynamic drag and Magnus lift on the ball would still need to be applied as custom forces (MuJoCo has fluid-force options, but spin lift for a perforated pickleball is not built in; unverified).

### Gaps
- Exact device-to-number mapping in the MJX-JAX table and the hardware for the 3.35M Warp figure were not recoverable through the fetch tool; read the MJX page directly before quoting.
- No measured M4/M5 `testspeed` number for a humanoid was found. A local benchmark is trivial to run and should replace the M3 Max figure.
- No source found on ball-bounce fidelity (coefficient of restitution tuning via `solref`/`solimp`) for pickleball specifically.
- MuJoCo Playground humanoid-specific timing tables were not retrieved.

## 2. NVIDIA Isaac Gym / Isaac Sim / Isaac Lab, Newton, Genesis, Brax, PyBullet, Drake, RaiSim, ProtoMotions

### Takeaway
The NVIDIA stack (Isaac Lab on PhysX or Newton, ProtoMotions on top) is where most published humanoid motion-tracking and sports work lives, and none of its GPU paths run on macOS; Newton and Warp install on macOS but only on CPU. Genesis is the one GPU-parallel simulator that claims an Apple Metal backend, but no Metal humanoid throughput numbers were found.

### Cited Findings
- Isaac Lab supported platforms (current docs): Ubuntu 22.04+ x86_64, Linux aarch64 (including DGX Spark), Windows 11 x86_64; no macOS. Full Isaac Sim workflows: at least 32 GB RAM and 16 GB VRAM recommended. "Headless Newton simulation does not require RTX hardware." (search summary of official docs) — [Isaac Lab installation (release 3.0.0)](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/isaaclab/installation.html)
- Isaac Lab 3.0 is built on Isaac Sim 6.0 with multi-backend physics (PhysX and Newton); the 3.0 beta was Ubuntu-first with Windows later; Newton 1.0 GA was announced at GTC 2026 (search summary) — [Isaac Lab Newton integration docs](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html); [Isaac Lab development update #4339](https://github.com/isaac-sim/IsaacLab/discussions/4339)
- Isaac Sim 6.2 release notes exist and reference Newton 1.6, so Isaac Sim is at 6.2 and Newton at about 1.6 by late 2026 (search summary) — [Isaac Sim 6.2.0 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html)
- Newton: Apache-2.0 code, Linux Foundation project initiated by Disney Research, Google DeepMind and NVIDIA; MuJoCo Warp is the primary backend; Python 3.11+. Platform text: "macOS (CPU only)" and "No local CUDA Toolkit is required; macOS runs on CPU." GPU acceleration needs an NVIDIA GPU (Turing or newer, driver R580+ with Warp 1.18+) — [newton-physics/newton](https://github.com/newton-physics/newton)
- Newton v1.5.0 moved its sim extra to MuJoCo and MuJoCo Warp 3.11 (secondary) — [humanoid.guide on Newton v1.5.0](https://humanoid.guide/newton-v1-5-0-adds-vectorized-joint-control-for-robot-simulation/)
- ProtoMotions 3 (NVlabs): Apache-2.0 code; backends IsaacGym (Preview 4), IsaacLab 3.0, Newton 1.0.0, MuJoCo 3.0+; "the Genesis backend has been removed"; characters include SMPL, Unitree G1, H1_2, SOMA; MuJoCo backend is CPU and described as the "smallest reference implementation", used for sim2sim testing; latest news item dated 28 Oct 2026. Supports MaskedMimic. No sports tasks listed — [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- ProtoMotions training claim: full public AMASS (40+ hours of motion) tracked "within 12 hours on 4 A100s"; multi-GPU run "with 24 A100s with 13K motions on each GPU" on BONES; a general tracker on BONES-SEED (~142K motions) — [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- Isaac Gym paper: an AMP humanoid policy learned with about 39 million samples in about 6 minutes with 4096 environments on one GPU; the PyBullet reference needed about 30 hours on 16 CPU cores for a similar sample count (search summary) — [Isaac Gym paper](https://arxiv.org/pdf/2108.10470)
- SMPLOlympics (CMU + NVIDIA, 2024): SMPL/SMPL-X-compatible humanoids in golf, javelin, high jump, long jump, hurdling and competitive 1v1 / 2v2 table tennis, tennis, fencing, boxing, soccer, basketball. The closest published precedent for four-player racket sport simulation — [SMPLOlympics abstract](https://arxiv.org/abs/2407.00187)
- PHC (ICCV 2023, Isaac Gym): tracks 98.9% of AMASS without external forces; 96.4% success on AMASS-Test; PHC+ (in PULSE) claims 100% on AMASS. A community port "puffer-phc" runs a simplified PHC with PufferLib — [PHC paper](https://ar5iv.labs.arxiv.org/html/2305.06456); [ZhengyiLuo/PHC](https://www.github.com/ZhengyiLuo/PHC); [puffer-phc](https://gh.r-mj.com/kywch/puffer-phc)
- Genesis: `genesis-world` on PyPI at 1.1.x; compile targets listed as CUDA, AMD ROCm, Apple Metal, Vulkan, x86, ARM64; README documents an Apple Silicon install path with standard PyTorch. Headline "43 million FPS" is a Franka arm on an RTX 4090, not a humanoid and not Apple hardware. A community repo trains a Unitree G1 with Genesis on Apple Silicon (Metal), last pushed Dec 2025 (search summary) — [genesis-world 1.1.1 on PyPI](https://pypi.org/project/genesis-world/1.1.1/); [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world); [deltartificial/g1 listing](https://awesome.ecosyste.ms/projects/github.com%2Fdeltartificial%2Fg1)
- Brax: `brax/training` (PPO/SAC in JAX) is still maintained as of 0.13.0, but for physics the project recommends MJX or MuJoCo Warp rather than Brax's own pipelines (search summary, from a fork README) — [Brax README (fork)](https://github.com/GTLIDAR/brax)
- Drake: native macOS arm64 support since v1.9.0; the Rosetta page is historical (search summary) — [Drake Rosetta 2 note](https://drake.mit.edu/rosetta2.html)

### Inferences
- Platform matrix for the reader's Mac: MuJoCo (yes, native), Drake (yes, native arm64), Newton / Warp / MuJoCo Warp (install and run, CPU only, intended for debugging), Genesis (yes, claims Metal GPU backend; unbenchmarked), Brax/MJX-JAX (CPU JAX only in practice), Isaac Sim / Isaac Lab / Isaac Gym (no), ProtoMotions (only via its MuJoCo CPU backend, i.e. evaluation and sim2sim, not large-scale training), mjlab (evaluation only).
- Typical GPU-simulator scale in the cited work is about 4096 environments per GPU, with millions of physics steps per second; that is the regime the published humanoid results assume.
- ProtoMotions already ships an SMPL humanoid, a motion tracker, and a MuJoCo backend. That makes "ProtoMotions on a rented NVIDIA GPU for training, MuJoCo on the Mac for playback and evaluation" a coherent split that keeps one character definition.
- Genesis is the only candidate for local GPU-parallel RL on the Mac, but ProtoMotions dropping its Genesis backend is a negative maturity signal, and its Metal path has no published humanoid numbers.

### Gaps
- Isaac Gym status: I found no source this session for its deprecation. Unverified background knowledge: Isaac Gym Preview 4 (2022) was the final release, is Linux + NVIDIA only, is closed-source binary under an NVIDIA licence, and NVIDIA directs users to Isaac Lab. ProtoMotions' "Preview_4" badge is consistent with that.
- PyBullet: no current release information found. Unverified background: runs on macOS on CPU, permissive (zlib) licence, largely in maintenance mode since about 2022; the Isaac Gym paper's 30-hour comparison above indicates its speed class.
- RaiSim: no licence or macOS information found this session. Unverified background: closed-source, licence-key based (free for academic use), CPU-only, with macOS builds historically available.
- SMPLOlympics, PHC, PULSE and MaskedMimic training compute (GPU type, envs, hours) were not retrieved; the abstracts do not state them. Unverified background: PHC-family papers report multi-day training on a single high-end NVIDIA GPU (A100 class).
- Genesis: no humanoid steps/s on Metal, and no confirmation that all rigid-body solver features work on the Metal backend.
- Licences for Isaac Sim / Isaac Lab were not checked this session (unverified background: Isaac Lab BSD-3; Isaac Sim open-sourced under Apache-2.0 in 2025 with some proprietary Omniverse components).

## 3. Unity + ML-Agents (and Sentis / Inference Engine)

### Takeaway
ML-Agents is alive but in low-intensity maintenance: the last GitHub "Release" is Release 23 (package 4.0.0, Aug 2025), with point releases through 4.0.3 and a reported 4.1.0 (29 Jul 2026) on the Unity registry whose only changes are a gym-to-gymnasium migration and a sensor bug fix. It runs on Apple Silicon and supports self-play, but it is a CPU, tens-of-environments system, and no recent published humanoid motion-tracking or sports research found here used it; it is a reasonable renderer or game front end, not a competitive training simulator for torque-controlled humanoids.

### Cited Findings
- GitHub releases: Release 23 = package 4.0.0, dated 2025-08-28, minimum Unity 6000.0, upgraded to Inference Engine 2.2.1, extensions package merged into main package. Release 22 = 3.0.0 (2024-09-02; Unity 2023.2, Sentis 2.0.0, PyTorch 2.1.1). Release 21 = 3.0.0-exp.1 (2023-10-09; Barracuda replaced by Sentis). Roughly one tagged release per year — [ML-Agents releases](https://github.com/Unity-Technologies/ml-agents/releases)
- 4.1.0 on the Unity package registry: gym to gymnasium migration (#6309) and a `StackingSensor` compressed-observation fix (#6306); minimum Unity 6000.0 (search summary of registry) — [packages.unity.com: com.unity.ml-agents](https://packages.unity.com/com.unity.ml-agents); 4.0.3 upgraded to Inference Engine 2.5.0 — [ML-Agents 4.0 changelog](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/changelog/CHANGELOG.html)
- A Japanese secondary article (Sep 2026) states 4.1.0 was published 29 Jul 2026, Python side remained `mlagents==1.1.0`, repository commits continued to 29 Sep 2026, licence Apache-2.0, and that the inference package name reverted from "Inference Engine" to "Sentis" at 2.4.0 with 4.1.0 bundling Sentis 2.6.1 (`com.unity.ai.inference`). Single secondary source; the naming claim conflicts with Unity's 4.0.3 changelog wording ("Inference Engine 2.5.0") — [issoh.co.jp article](https://www.issoh.co.jp/md/tech/4905); contradicted in part by [ML-Agents 4.0 changelog](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/changelog/CHANGELOG.html)
- Community concern about abandonment is on record: a Unity forum thread titled "Is ML-Agents being deprecated?"; Unity then announced 4.0.0 — [Unity Discussions: Is ML-Agents being deprecated?](https://discussions.unity.com/t/is-ml-agents-being-deprecated/1643681); [ML-Agents 4.0.0 is now available](https://discussions.unity.com/t/ml-agents-4-0-0-is-now-available/1681770/1)
- ML-Agents remains listed as a package in the Unity 6.4 (6000.4) manual — [Unity Manual: ML-Agents](https://docs.unity3d.com/Manual/com.unity.ml-agents.html)
- Humanoid research on Unity is old and small-scale: Marathon Environments (2019) trained Unity ragdoll humanoids with ML-Agents; the backflip agent took about 25 hours wall-clock with 64 agents in parallel, and the auto-generated ragdoll had self-collision problems giving lower motion quality than DeepMimic. A 2020 paper built a physics character controller on Unity ArticulationBody joints with PPO — [Marathon Environments](https://arxiv.org/pdf/1902.09097); [ArticulationBody character controller paper](https://arxiv.org/pdf/2006.07508v1.pdf)
- A 2022 survey of RL for character animation describes ML-Agents as a Unity plugin with PPO and SAC, and Isaac Gym as the GPU-parallel alternative — [Survey on RL methods in character animation](https://arxiv.org/pdf/2203.04735)
- Scale contrast: Marathon used 64 parallel agents and about 25 hours for one skill; Isaac Gym trains an AMP humanoid in about 6 minutes with 4096 environments — [Marathon Environments](https://arxiv.org/pdf/1902.09097); [Isaac Gym paper](https://arxiv.org/pdf/2108.10470)
- MuJoCo ships an official Unity plug-in (documented in the MuJoCo docs navigation), which lets Unity render and host MuJoCo physics instead of PhysX — [MuJoCo changelog page (navigation: Unity Plug-in)](https://mujoco.readthedocs.io/en/latest/changelog.html)

### Inferences
- Is Unity + ML-Agents a good fit? For the RL-training core, no: it is CPU PhysX with tens to low hundreds of agents, has no motion-imitation tooling (no AMP/PHC-style tracker, no SMPL humanoid, no AMASS pipeline), and the modern humanoid literature (PHC, PULSE, MaskedMimic, SMPLOlympics, ProtoMotions, BeyondMimic/mjlab) is entirely on Isaac Gym / Isaac Lab / MuJoCo-family simulators. Choosing Unity means re-implementing that stack alone.
- Where Unity does fit: as a presentation layer (render the match, cameras, UI) fed by trajectories from MuJoCo, or as a host for trained policies via Sentis / Inference Engine (ONNX). The MuJoCo Unity plug-in is a possible bridge.
- Maintenance risk is moderate: releases continue (4.1.0 in 2026) but are dependency bumps, not new algorithms; the Python trainer has barely changed.

### Gaps
- Not verified this session (unverified background knowledge): ML-Agents supports self-play with ELO tracking and the MA-POCA cooperative trainer; trainers are PPO, SAC, MA-POCA plus GAIL/BC imitation; the Unity Editor and players run natively on Apple Silicon; the PyTorch trainer can use CPU or MPS. Unity's Walker/Crawler examples use ConfigurableJoint ragdolls with target-rotation (PD-like) control rather than raw torques, and ArticulationBody (PhysX reduced-coordinate, Featherstone) is the more accurate option.
- DReCon (Ubisoft La Forge, 2019) is often cited as a Unity humanoid controller; I could not confirm its engine or whether it used ML-Agents (background: it did not use ML-Agents).
- No head-to-head training-speed benchmark of ML-Agents versus a GPU simulator for humanoids was found.
- Official 4.1.0 changelog page and the exact Sentis / Inference Engine version it bundles were not retrieved from a primary source.

## 4. SMPL and SMPL-X Unity packages; MHR / Momentum assets

### Takeaway
The Max Planck SMPL / SMPL-X Unity packages are rendering and animation assets (rigged FBX meshes with shape and pose-corrective blend shapes plus set-up scripts), released under non-commercial research licences that require registration and restrict redistribution of the model; nothing found describes colliders, rigid bodies or joints, so they are not physics-ready. They are worth considering only as a skin for visualisation, not as the simulated body.

### Cited Findings
- SMPL site: provides "example FBX files with animated models and code for using SMPL in Python, Maya, and Unity"; Maya and Unity scripts help set up the model for animation. SMPL has 6890 vertices and 23 body joints plus root (search summary) — [SMPL site](https://smpl.is.tue.mpg.de/)
- "SMPL-X for Blender and Unity" released 17 Jun 2021 by Perceiving Systems, licence type listed as "PS:License 1.0"; SMPL-X 1.1 is added as "a textured skeletal mesh with a shape specific rig, as well as shape keys (blend shapes) for shape, expression and pose correctives"; code is in the downloads section of the SMPL-X site for registered users — [SMPL-X for Blender and Unity](https://ps.is.mpg.de/code/smpl-x-for-blender-and-unity)
- SMPL-X model licence is for "non-commercial scientific research purposes" — [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html); SMPL licence page and commercial contact (smpl@max-planck-innovation.de) — [SMPL licence](https://smpl.is.tue.mpg.de/license.html)
- Generated SMPL bodies (meshes output by the model) fall under a separate "SMPL Body" licence, CC BY 4.0, distinct from the model licence (search summary) — [SMPL licence](https://smpl.is.tue.mpg.de/license.html)
- Commercial SMPL licensing is handled through Meshcapade / Max Planck Innovation — [Meshcapade wiki: SMPL](https://meshcapade.wiki/SMPL)
- bmlSUP (BioMotionLab SMPL Unity Player) plays AMASS/SMPL animations in Unity and falls under the same licence agreement as SMPL; it is a player, not a physics rig — [bmlSUP paper](https://biomotionlab.github.io/SUP/docs/assets/bmlSUP_IEEEVR2021_Paper.pdf); [bmlSUP site](https://www.biomotionlab.ca/sup/)
- A community "Realtime SMPLX Unity" project exists for driving SMPL-X in Unity in real time — [Realtime_SMPLX_Unity](https://github.com/thinhlpg/Realtime_SMPLX_Unity)
- Generic Unity active-ragdoll tooling exists but is not SMPL-specific: PhysAnim (open source, motor- and keyframe-based ragdoll pose matching) — [PhysAnim-Unity](https://github.com/lenoctambule/PhysAnim-Unity)
- ProtoMotions ships an SMPL humanoid for physics simulation (`--robot-name=smpl`) with SMPL/SMPL-H attribution and licence notices, i.e. a physics-ready SMPL-shaped body already exists in the MuJoCo/Isaac ecosystem — [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- MHR (Momentum Human Rig, Meta) has an official project page — [MHR project page](https://facebookresearch.github.io/MHR)

### Inferences
- The SMPL Unity package does not shorten the path to simulated players: a physics humanoid needs capsule/box collision geoms, masses, joint limits and actuators derived from SMPL shape parameters. That is what the SMPL-to-MJCF converters used by PHC / ProtoMotions provide, in MuJoCo format, not Unity format.
- If video-to-motion recovery outputs SMPL parameters, the natural pipeline is SMPL params → SMPL-shaped MJCF humanoid in MuJoCo (per-player body shape) → tracking controller, with the SMPL mesh used only as a visual skin.
- Licence implication for a public web page: redistributing the SMPL model files (or a Unity package containing them) is not permitted under the research licence; pre-posed output meshes fall under the more permissive body licence, but a skinned, animatable SMPL in the browser effectively ships the model. A capsule/primitive humanoid or a freely licensed rig avoids the issue.

### Gaps
- I could not retrieve the actual file listing of the SMPL Unity package or the full text of its licence clauses on redistribution; the non-commercial and registration requirements are confirmed, the exact redistribution wording is not.
- No source confirms or denies physics components in the Max Planck Unity packages; "rendering-only" is inferred from their descriptions (skeletal mesh, blend shapes, animation set-up scripts).
- MHR: the project page returned no licence or export details. Unverified background: MHR code and model were released by Meta in late 2025 under Apache-2.0, built on the Momentum library, and used by SAM 3D Body; I found no official Unity or Unreal asset package for it.

## 5. Other engines: Unreal Learning Agents, Godot RL Agents, Madrona

### Takeaway
None of these is a credible primary simulator for torque-controlled humanoid athletes; they are game-AI tools. Unreal Learning Agents is still in use on UE 5.7/5.8 per forum activity, Godot RL Agents is a small MIT-licensed bridge to standard RL libraries, and nothing was found on Madrona.

### Cited Findings
- Unreal Learning Agents: active forum threads into late 2026 (including a 3 Nov 2026 thread about RTX 5000-series support, and UE 5.7 issue threads); no official 2026 roadmap statement found — [Unreal forums: Learning Agents tag](https://forums.unrealengine.com/tag/Learning-Agents); [Learning Agents issues thread](https://forums.unrealengine.com/t/learning-agents-issues/2708583)
- Unreal Engine 5.8 was available by mid-2026 (secondary) — [Edge: State of Unreal 2026 takeaways](https://knowledge.edge-online.com/p/ai-goes-native-in-unreal-engine-the-five-main-takeaways-from-state-of-unreal-2026)
- Godot RL Agents: MIT licence; `pip install godot-rl`; backends StableBaselines3, Sample Factory, Ray RLlib, CleanRL; original paper Dec 2021; PyPI shows 0.6.x — [godot_rl_agents](https://github.com/edbeeching/godot_rl_agents); [Godot RL Agents paper](https://arxiv.org/abs/2112.03636); [godot-rl on PyPI](https://pypi.org/project/godot-rl/0.6.2/)

### Inferences
- Both Unreal and Godot would have the same structural problems as Unity for this project: game-grade physics, CPU-side stepping, no motion-imitation ecosystem. Their role, if any, is rendering.

### Gaps
- Madrona: no results. Unverified background: Madrona is a research ECS engine for GPU batch simulation (CUDA for the GPU backend, CPU backend runs on macOS) whose built-in physics is basic; its batch renderer is used by MuJoCo Playground for vision tasks. Not a humanoid dynamics engine in its own right.
- Whether Learning Agents has been used for physically simulated full-body humanoids was not established.

## 6. RL libraries and Apple Silicon (MPS) support

### Takeaway
Sourced evidence here is thin: PyTorch MPS works on macOS with CPU fallback for unimplemented ops, and PufferLib has at least a community Apple Silicon path, but I found no official MPS statements for rl_games, RSL-RL or skrl. In practice the policy network for humanoid RL is small, so the physics, not the learner, is the bottleneck on a Mac.

### Cited Findings
- PyTorch MPS backend requires macOS 12.3+; some ops are not implemented on MPS; `PYTORCH_ENABLE_MPS_FALLBACK=1` falls back to CPU kernels — [Hugging Face: PyTorch training on Apple Silicon](https://huggingface.co/docs/transformers/perf_train_special); [Apple: Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch)
- PufferLib: a community pull request (Aug 2025) states stock PufferLib failed to import on Mac, makes the C extension import optional, adds device auto-selection, and reports 235K+ steps/s training on an M4 Mac mini; merge status unverified (search summary) — [PufferLib PR 318 (mirror)](https://tgfed.nhyvvfeijnh.workers.dev/PufferAI/PufferLib/pull/318); [PufferAI/PufferLib](https://github.com/PufferAI/pufferlib)
- PufferLib 3.0 advertises training at about 4,000,000 steps/s (on its own lightweight C environments, not humanoid physics) — [PufferLib blog](https://puffer.ai/blog.html)
- puffer-phc: a simplified PHC humanoid tracker trained with PufferLib (on Isaac Gym) — [puffer-phc](https://gh.r-mj.com/kywch/puffer-phc)
- Brax's training module (JAX PPO/SAC) is maintained as of 0.13.0 and is the standard learner paired with MJX / MuJoCo Playground — [Brax README (fork)](https://github.com/GTLIDAR/brax)
- ML-Agents Release 22 pinned PyTorch 2.1.1 — [ML-Agents releases](https://github.com/Unity-Technologies/ml-agents/releases)
- Godot RL Agents pairs with StableBaselines3, Sample Factory, RLlib, CleanRL — [godot_rl_agents](https://github.com/edbeeching/godot_rl_agents)

### Inferences
- On the Mac, any pure-PyTorch learner (Stable-Baselines3, CleanRL, RSL-RL, skrl with a non-Isaac environment) should run on CPU or MPS because they are device-agnostic PyTorch; for the small MLPs typical of humanoid control, CPU is often as fast as MPS once host-to-device copies are counted. The libraries tied to GPU simulators (rl_games and RSL-RL as shipped in Isaac Lab / ProtoMotions / mjlab) assume observations arrive as CUDA tensors and are only useful locally if the environment is swapped for a CPU one.
- JAX learners (Brax PPO) run on the Mac on CPU only, given the jax-metal status in section 1.

### Gaps
- No primary source found for MPS support or macOS status of rl_games, RSL-RL, skrl, Stable-Baselines3 or CleanRL. Unverified background: SB3 and CleanRL accept `device="mps"`; skrl supports PyTorch and JAX back ends; RSL-RL is pip-installable and device-agnostic; rl_games is the default in Isaac Gym-era humanoid code (PHC, AMP).
- No benchmark of MPS versus CPU for small-MLP PPO on M-series was found.

## 7. Realistic compute budgets and cloud cost

### Takeaway
Published numbers span four orders of magnitude: a single-skill humanoid policy trains in minutes on one consumer NVIDIA GPU with about 4096 environments, a general motion tracker over all of AMASS takes about 48 A100-hours (roughly $60–90 rented), and frontier foundation-scale trackers use 9k–32k GPU-hours. A single Mac is adequate for playback, model building and small single-clip tracking experiments; anything involving thousands of parallel environments is cheaper and far faster on a rented NVIDIA GPU at roughly $0.35–$2 per hour.

### Cited Findings
- Isaac Gym: AMP humanoid, about 39M samples, about 6 minutes, 4096 environments, one GPU; PyBullet equivalent about 30 hours on 16 CPU cores — [Isaac Gym paper](https://arxiv.org/pdf/2108.10470)
- MJX + Brax PPO humanoid walking (REEM-C): about 56 minutes on an RTX 4090 (AMD 7950X, 64 GB RAM), with a collision-reduced model (search summary) — [REEM-C MJX paper](https://www.arxiv.org/pdf/2407.05148)
- FastSAC / FastTD3 humanoid locomotion: about 15 minutes on a single RTX 4090 (search summary) — [Learning Sim-to-Real Humanoid Locomotion in 15 minutes](https://arxiv.org/html/2512.01996v1); [FastTD3](https://arxiv.org/html/2505.22642v3)
- mjlab motion imitation example: Unitree G1, 4096 environments on an NVIDIA GPU (fork README) — [mjlab (fork README)](https://github.com/Vertax42/mjlab)
- ProtoMotions: all of AMASS (40+ h of motion) in "within 12 hours on 4 A100s" = about 48 A100-hours; large runs on 24 A100s — [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)
- SONIC (NVIDIA, Nov 2025): 100M frames of motion; the paper gives inconsistent compute figures in different places: 9k GPU-hours, 21k GPU-hours, and 32,000 GPU-hours (128 GPUs for 3 days). It also remarks that state-of-the-art humanoid controllers are "typically small neural networks trained on a single GPU for days" — [SONIC paper](https://arxiv.org/html/2511.07820v1)
- SuperMimic (Oct 2026, authors' claim, not hardware-matched): in 3 hours on a single GPU surpasses a tracker baseline trained 7 days on 128 GPUs (secondary summary) — [SuperMimic summary](https://www.emergentmind.com/papers/2610.28480)
- Unity-era reference: Marathon backflip about 25 hours with 64 parallel agents; DeepMimic about 48 hours per skill on CPU — [Marathon Environments](https://arxiv.org/pdf/1902.09097)
- Native MuJoCo CPU humanoid throughput on Apple M3 Max is one of the four figures (likely about 650K steps/s) in the MJX docs table — [MJX docs (stable)](https://mujoco.readthedocs.io/en/stable/mjx.html)
- 2026 cloud prices (trackers disagree; treat as ranges): RunPod RTX 4090 about $0.34/h community, $0.69–0.74/h secure (some list $0.89); RunPod A100 80GB about $1.19–1.79/h; RunPod H100 about $1.99/h (PCIe) to $2.69–3.99/h; RunPod L40S about $0.79–0.99/h; Lambda H100 about $3.29–3.78/h, A100 about $1.29–2.79/h, no RTX 4090 (search summary across sites) — [RunPod pricing](https://www.runpod.io/gpu-cloud/pricing); [tech-insider: RunPod vs Lambda vs Vast 2026](https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/); [Thunder Compute comparison](https://www.thundercompute.com/blog/Runpod-pricing-vs-thunder-compute); [DeployBase: RunPod GPU pricing](https://deploybase.ai/articles/runpod-gpu-pricing); [Hackceleration RunPod pricing](https://www.hackceleration.com/labs/runpod-pricing)

### Inferences
- Rough cost per run (arithmetic on the figures above): single-skill or single-clip policy (15–60 min on an RTX 4090) ≈ $0.10–$0.90; a multi-hour single-GPU experiment (say 8 h on a 4090 or L40S) ≈ $3–8; full-AMASS general tracker (48 A100-hours) ≈ $57–86 at $1.19–1.79/h; "single GPU for days" (3 days on an A100) ≈ $85–130; SONIC-scale (9k–32k GPU-hours) ≈ tens of thousands of dollars, out of scope.
- Mac versus rented GPU: if an M-series CPU delivers a few hundred thousand simple-humanoid steps/s and an NVIDIA GPU with MuJoCo Warp or Isaac delivers low millions, the Mac is roughly 5–20x slower before accounting for the heavier four-player scene. A 1-hour GPU job becomes an overnight Mac job; a 12-hour 4xA100 job is impractical locally (weeks).
- For this project the cheapest path is probably not to train a general tracker at all: reuse a pretrained tracker (PHC / PULSE / ProtoMotions checkpoints exist; ProtoMotions' pretrained models are under the NVIDIA Open Model License) and fine-tune or train task policies for pickleball, which sits in the "$1–$100 per run" band.
- Multi-agent self-play for four players will multiply sample requirements relative to single-humanoid tracking; SMPLOlympics is the reference to consult for what was achievable, but its compute was not retrieved.

### Gaps
- No published compute for racket-sport humanoid policies was retrieved (SMPLOlympics, Vid2Player3D tennis, table-tennis and badminton humanoid papers). These are the most relevant comparators and should be read directly.
- PHC / PULSE / MaskedMimic training time and GPU type not confirmed.
- Cloud prices are from aggregator sites with conflicting numbers and various dates (June–October 2026); check live provider pages before budgeting.
- No measured M5 Pro throughput; the M3 Max figure is the nearest proxy and its mapping in the table is uncertain.

## 8. Getting the result into a web page

### Takeaway
MuJoCo has official WebAssembly/JavaScript bindings maintained by Google DeepMind (npm `@mujoco/mujoco`, README at 3.9.0, self-described as work in progress), so a MuJoCo scene can be both replayed and stepped live in a browser, typically rendered with three.js; a Python `mujoco-studio-wasm` wheel and a Studio Web Viewer also exist as of 4.0.1.

### Cited Findings
- `@mujoco/mujoco` on npm: "canonical JavaScript and TypeScript bindings for the MuJoCo physics engine", exposing the core engine compiled to WebAssembly, "developed and maintained by Google DeepMind"; README warns the bindings are still a work in progress — [@mujoco/mujoco README (3.9.0)](https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.9.0/README.md); [MuJoCo wasm README](https://raw.githubusercontent.com/google-deepmind/mujoco/main/wasm/README.md)
- MuJoCo 3.17.0 changelog mentions generated accessors in the JavaScript bindings; 4.0.1 says the Studio Web Viewer now works on Windows via a `mujoco-studio-wasm` wheel — [MuJoCo changelog, latest](https://mujoco.readthedocs.io/en/latest/changelog.html)
- Community `mujoco_wasm` (zalo, with contributions from Kevin Zakka) provides a three.js viewer on top of the WASM build and is linked from the MuJoCo GitHub page; it now pulls the official bindings — [zalo/mujoco_wasm (indexed)](https://docsearch.algolia.com/mcp/docs/repo/zalo/mujoco_wasm); [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- Other packages: an older unscoped `mujoco` npm package (3.1.16), `mujoco-js` (publisher unverified), `@vuer-ai/mujoco-ts` (React component), and MuJoCo-GS-Web (MuJoCo plus Gaussian splatting in the browser) — [mujoco npm 3.1.16](https://cdn.jsdelivr.net/npm/mujoco@3.1.16/README.md); [mujoco-js stats](https://www.pkgstats.com/pkg:mujoco-js); [@vuer-ai/mujoco-ts](https://cdn.jsdelivr.net/npm/@vuer-ai/mujoco-ts@0.0.79/README.md); [MuJoCo-GS-Web](https://gh.r-mj.com/Vector-Wangel/MuJoCo-GS-Web)
- Unity's inference runtime (Sentis / Inference Engine) is what ML-Agents uses to run trained ONNX policies inside a Unity build — [ML-Agents releases](https://github.com/Unity-Technologies/ml-agents/releases)

### Inferences
- Two web delivery modes are available with the MuJoCo route: (a) replay, export joint trajectories (qpos per frame) and animate a three.js scene, with no physics in the browser, which is the cheapest and most robust; (b) live simulation, load the same MJCF in MuJoCo WASM and step it in the page, optionally running the trained policy with an ONNX runtime in the browser. A scene of four humanoids and a ball is well within single-threaded WASM real-time for MuJoCo-class dynamics (inference from the CPU throughput figures; not measured).
- The same MJCF model therefore serves Mac development, cloud GPU training (MuJoCo Warp / mjlab / Newton / ProtoMotions MuJoCo backend) and the browser. No Unity/Unreal/Isaac option offers that continuity; Unity WebGL builds can run PhysX and Sentis in a browser but with a much heavier payload.
- GPU physics in the browser via WebGPU is not something any source here confirmed; treat browser simulation as CPU/WASM.

### Gaps
- Current version of `@mujoco/mujoco` on npm (whether it tracks 4.0.x) not confirmed; the indexed README is 3.9.0.
- No measured WASM steps/s for a humanoid scene found.
- No source found on running policies in-browser alongside MuJoCo WASM (ONNX Runtime Web is the obvious candidate; unverified).
- Whether Unity 6 Web builds support Sentis / Inference Engine on WebGPU was not checked.
