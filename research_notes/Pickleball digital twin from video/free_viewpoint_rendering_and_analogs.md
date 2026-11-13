# Free-viewpoint sports replay: multi-camera reference systems, monocular reconstruction frontier, and Mac viewer/physics tooling

Notes compiled 2026-11-11. Conventions used throughout:

- **[V]** = verified this session by fetching the page or reading a search result drawn from it.
- **[PK]** = from the researcher's prior knowledge; the URL is the canonical location but was NOT re-fetched this session. Treat as likely correct but unverified.
- **NEW** = published after mid-2025; exact date given where known.
- arXiv abstracts were read, not full papers, unless stated. Hardware and limitation claims not in abstracts are flagged as gaps.

---

## 1. What was the "Ultra-Perspective" (超视角) system at China's 15th National Games, and how do comparable systems work? Which are "tracking data rendered in a game engine" rather than volumetric capture?

### Takeaway
"超视角" was a true multi-camera system: 36 custom cameras ringing the court, built by China Media Group (CMG/央视频) with Peng Cheng Laboratory and Peking University Shenzhen Graduate School, delivered as a draggable 360-degree view in the 央视频 app for badminton and basketball; no source states the reconstruction algorithm (no confirmation of Gaussian splatting). The closest analog to what one broadcast camera can feed is the other family entirely: Hawk-Eye skeletal tracking rendered by Sony's Beyond Sports or Tennis Australia's "AO Animated", which are tracking data driving rigged characters in a real-time renderer, not captured pixels.

### Cited Findings

**15th National Games "超视角" (NEW, November 2025)**
- [V] Built by CMG's 央视频 with "国重实验室" (a state key laboratory); system name "立体视频赛事直播系统" (stereo/volumetric video live event system); based on a "real-time stereo video capture array" plus self-developed deep-learning and 3D reconstruction/rendering algorithms doing "real-time 3D modelling and seamless multi-view video fusion". Article dated 2025-11-08. — [Tencent News / 央视频, 2025-11-08](https://news.qq.com/rain/a/20251108A052GL00)
- [V] 36 precision custom cameras around the venue give 360-degree coverage; the same ring powers "bullet time" replays. — [Tencent News, 2025-11-08](https://news.qq.com/rain/a/20251108A052GL00); [CGTN, 2025-11-20](https://news.cgtn.com/news/2025-11-20/360-degree-live-streaming-tech-enhances-National-Games-experience-1IsFXdXo6Yg/p.html)
- [V] Viewer: continuous 360-degree drag (not preset angles), rotate, zoom, overview or close-up, on phone/tablet in the 央视频 client. Stated view-switch response 0.1 s; claimed smooth "even on 4G". No end-to-end stream latency figure is given. — [Tencent News, 2025-11-08](https://news.qq.com/rain/a/20251108A052GL00)
- [V] Venues: Dongguan Sports Center (basketball), Shenzhen Sports Center (badminton). First trialled at the 2025 Li-Ning China Masters badminton in Shenzhen (September 2025). — [Tencent News, 2025-11-08](https://news.qq.com/rain/a/20251108A052GL00); [Tencent News, 2025-09-20](https://news.qq.com/rain/a/20250920A037HD00)
- [V] Peng Cheng Laboratory (鹏城实验室) supplied the 3D immersive video capture and coding technology; CMG's State Key Laboratory of UHD Audio/Video Production and Presentation ran production and user access; Peking University Shenzhen Graduate School co-developed. Delivery relies on national standard GB/T 44115.2-2024 "Information technology — Virtual reality content representation — Part 2: Video", led by Peng Cheng Lab and PKU Shenzhen, covering 3D immersive video representation, compression and real-time transmission. The article claims "tens of millions" of online participations. Dated 2025-11-24. Camera count, codec name, rendering method, bitrate and resolution are not stated. — [ScienceNet, 2025-11-24](https://news.sciencenet.cn/htmlnews/2025/11/555757.shtm)
- [V] Listed among CMG's "ten technology innovation applications" for the Games (2025-11-11), alongside 8K and vertical-screen production. — [Tencent News, 2025-11-11](https://news.qq.com/rain/a/20251111A010CJ00)
- [V] English secondary coverage calls it "ultra-perspective" and attributes details to QQ. — [Dexerto](https://www.dexerto.com/entertainment/china-revolutionizes-live-sports-streaming-by-letting-viewers-fully-control-the-camera-3283469/)
- Caveat: all sources are CMG, partner labs or outlets repeating them. No independent technical evaluation was found.

**Volumetric / free-viewpoint multi-camera systems (real pixels, many cameras)**
- [V] Intel True View: roughly 38 5K cameras around a stadium perimeter (sources vary: "~40", "38", "dozens"); voxel-based volumetric reconstruction for replays. Intel shut down its sports division in 2021, selling part to Verizon. — [SVG, 2020-11-16](https://www.sportsvideo.org/2020/11/16/svg-tech-insight-immersive-media-experiences-with-intel-true-view-delivers-new-reality-for-sports/); [CTech](https://www.calcalistech.com/ctech/articles/0,7340,L-3915635,00.html)
- [V] Canon Free Viewpoint Video System: 125 cameras at Rugby World Cup 2019 with near-real-time virtual-camera replays at 1080p; about 100 Cinema EOS-based cameras at Cleveland's arena ("C-View") and at Barclays Center ("Netaverse"); an ESPN+ Nets–Mavericks game in March 2022 was produced almost entirely from the virtual camera. Microsecond shutter sync across 100+ cameras is a stated requirement. — [SVG, 2019-10-25](https://www.sportsvideo.org/2019/10/25/live-from-the-rugby-world-cup-canon-free-viewpoint-system-offers-near-realtime-volumetric-capture/); [SVG, 2022-03-21](https://www.sportsvideo.org/2022/03/21/canon-free-viewpoint-tech-used-for-espn-full-nba-game-production); [Canon](https://global.canon/en/technology/frontier18.html)
- [V] Apple Vision Pro "Spectrum Front Row in Apple Immersive" (NEW, live from 2026-01-09): select Lakers games shot with Blackmagic URSA Cine Immersive Live cameras placed courtside and under each basket. This is multi-angle 180-degree stereo video from fixed positions, not free viewpoint. The NBA app separately offers a "Tabletop" view. — [Apple Newsroom, January 2026](https://www.apple.com/newsroom/2026/01/spectrum-front-row-tips-off-january-9-on-apple-vision-pro/); [NBA support](https://support.watch.nba.com/hc/en-us/articles/20891550821783)

**Tracking data rendered in a game engine (the single-camera-relevant category)**
- [V] Hawk-Eye SkeleTRACK: optical, markerless, 29 skeletal points per athlete. — [Hawk-Eye](https://hawkeyeinnovations.com/news/4243365/skeletrack-a-new-era-of-data-in-tennis)
- [V] Beyond Sports (Sony): ingests multiple tracking sources, "analyze, validate, enhance, translate to a 3D environment" and stream in real time with under one second of delay. Its CTO noted raw ankle and toe points produce unnatural movement, so an "enhancement system" cleans the motion per sport. — [Hawk-Eye: data-driven virtual recreations](https://hawkeyeinnovations.com/news/4216051/defining-the-sports-experiences-of-tomorrow-datadriven-virtual-recreations); [AWS NAB25 demo](https://aws.amazon.com/media/nab25-demos/data-driven-alternative-programming)
- [V] Products built this way: ESPN "Dunk the Halls" (NBA, 2024-12-25, first real-time animated NBA altcast, Hawk-Eye optical tracking); "The Simpsons Funday Football" (NFL, 2024-12-09); NHL "Inside Out Classic" (NEW, reported 2026-04-02), which fused NHL EDGE puck/player sensors (up to 40 Hz) with Hawk-Eye optical limb and stick tracking. — [SVG, 2024-12-20](https://www.sportsvideo.org/2024/12/20/sonys-beyond-sports-teams-up-with-espn-for-first-real-time-animated-alternate-presentation-of-an-nba-game/); [SVG, 2026-04-02](https://www.sportsvideo.org/2026/04/02/coming-of-age-espn-and-nhls-inside-out-classic-marks-another-step-forward-for-the-animated-alternative-broadcast/)
- [V] Tennis Australia "AO Animated": 12 cameras, 29 skeletal points per player at 50 fps plus ball tracking, pushed through graphics render engines into cartoon avatars on a roughly two-minute delay; debuted January 2025 on YouTube; NEW: ESPN carried it for 2026 semifinals and finals (reported 2026-01-29). The format also sidesteps broadcast-rights restrictions. — [SVG, 2025-01-24](https://www.sportsvideo.org/2025/01/24/tennis-australia-innovates-at-ao-with-ao-animated-as-it-renews-tech-deal-with-emg-gravity-media/); [NPR, 2025-01-17](https://www.npr.org/2025/01/17/nx-s1-5261006/australian-open-wii-animation); [SVG, 2026-01-29](https://www.sportsvideo.org/2026/01/29/espn-debuts-tennis-australias-ao-animated-altcasts-for-semis-and-finals/)
- [V] MLB Statcast (Hawk-Eye) tracks 18 skeletal points at 30 Hz. — [Built In](https://builtin.com/articles/mlb-statcast-tech-update-hawk-eye-integration)

### Inferences
- The 超视角 pipeline is most plausibly multi-view video plus depth/geometry with view synthesis on the server or client (the "immersive video" lineage that GB/T 44115.2 standardises), not per-frame 4D Gaussian splatting. This is an inference from the standard's scope and the 0.1 s view-switch claim; no source names the algorithm. With 36 cameras at 10-degree spacing, intermediate views need only small-baseline interpolation, which is why it looks seamless. One camera offers none of that.
- The user's goal ("modelled court, generic body meshes driven by recovered motion, simulated ball") is architecturally the Beyond Sports / AO Animated design, with the 12-camera skeletal tracker replaced by monocular pose estimation. AO Animated is the single best public reference: same sport family, cartoon avatars, visibly imperfect (dropped rackets, jittery limbs were widely noted in coverage), and still enjoyable to watch.
- Beyond Sports' "enhancement" step is direct evidence that even 12-camera skeletal data needs a motion clean-up layer before it looks natural. A monocular pipeline needs that layer more, which supports the user's instinct to prioritise smooth physically grounded motion over accuracy.

### Gaps
- No source confirms the 超视角 reconstruction method, codec, end-to-end latency, stream resolution, or whether rendering is client-side or cloud-side.
- No evidence was found of Tencent, Huawei or Migu building the National Games system; the named parties are CMG, Peng Cheng Laboratory and PKU Shenzhen. Migu and others have run separate free-view products, but these were not researched.
- No source names the render engine behind Beyond Sports (Unreal vs Unity vs proprietary).
- Arcturus (volumetric video tooling; reportedly took over Microsoft's Mixed Reality Capture Studios technology) and Wimbledon 3D recreations were not found in searches this session; nothing verified.
- NFL altcast pipeline details (Next Gen Stats RFID vs optical skeletal) were not verified.

---

## 2. Monocular dynamic reconstruction research (2024 to November 2026): how well does it handle large viewpoint changes, small distant humans and long clips, and what hardware does it need?

### Takeaway
Photoreal free viewpoint of real players from one broadcast camera is still not feasible in November 2026: optimisation-based monocular 4D Gaussian methods score around 16 dB PSNR at any substantial angular offset even on synthetic single objects, and 2026 generative "camera redirection" papers still describe large-angle changes as the open failure case. Everything in this family assumes CUDA GPUs; none is a practical route on an M5 Pro.

### Cited Findings

**Optimisation-based monocular 4D Gaussian splatting**
- [PK] Shape of Motion (July 2024): 4D reconstruction from a single video using SE(3) motion bases, relying on monocular depth and long-range 2D tracks. — [arXiv 2407.13764](https://arxiv.org/abs/2407.13764)
- [PK] MoSca (May 2024): "4D motion scaffolds" lifting foundation-model priors to dynamic Gaussians from casual video. — [arXiv 2405.17421](https://arxiv.org/abs/2405.17421)
- [V] NEW: USplat4D (arXiv October 2025; ICLR 2026) states Shape of Motion "struggle[s] on extreme novel views far from the input trajectory", including opposite-side views, and that SoM and MoSca do not model motion behind occluders. On a synthetic Objaverse benchmark of 6 articulated objects (121-frame orbit), reported novel-view PSNR by horizontal offset: SoM 16.09 / 15.58 / 16.45 dB at 0–60 / 60–120 / 120–180 degrees; MoSca 16.18 / 15.74 / 15.89; USplat4D raises these by under 1 dB. (Table values came through a lossy extraction; check the original before quoting.) Real-video extreme views (DyCheck, DAVIS) are evaluated qualitatively only, because no ground truth exists. — [arXiv 2510.12768v3](https://arxiv.org/html/2510.12768v3); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007618)
- [V] The original 4D-GS paper notes deformation fields struggle with large motion and overfit training views. — [arXiv 2310.08528](https://arxiv.org/pdf/2310.08528)
- [V] "4D Gaussian Splatting in the Wild": casual monocular video lacks multi-view information, so methods overfit; fast motion breaks initialisation. — [arXiv 2411.08879](https://arxiv.org/html/2411.08879v1)
- [V] NEW: A February 2026 paper on multi-scale motion fields restates that monocular 4D reconstruction is "fundamentally ill-posed". — [arXiv 2602.13806](https://www.arxiv.org/pdf/2602.13806)

**Feed-forward 4D geometry (pointmaps, not renderable avatars)**
- [PK] MonST3R (October 2024), CUT3R (January 2025), St4RTrack (April 2025) predict per-frame pointmaps, camera poses and, for St4RTrack, 3D tracks from video in a feed-forward pass. They output partial visible-surface point clouds, not complete geometry. — [MonST3R 2410.03825](https://arxiv.org/abs/2410.03825); [CUT3R 2501.12387](https://arxiv.org/abs/2501.12387); [St4RTrack 2504.13152](https://arxiv.org/abs/2504.13152)

**Animatable human Gaussian avatars**
- [PK] GaussianAvatar, HUGS (both late 2023) and ExAvatar (July 2024) fit per-subject Gaussian avatars on SMPL/SMPL-X from a monocular video in which the person is large in frame and seen from many sides. — [GaussianAvatar 2312.02134](https://arxiv.org/abs/2312.02134); [HUGS 2311.17910](https://arxiv.org/abs/2311.17910); [ExAvatar 2407.21686](https://arxiv.org/abs/2407.21686)
- [V] LHM (ICCV 2025): feed-forward transformer producing an animatable Gaussian avatar from a single image "in seconds", anchored on SMPL-X surface points. PF-LHM (June 2025) extends to one or a few unposed images. — [LHM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Qiu_LHM_Large_Animatable_Human_Reconstruction_Model_for_Single_Image_to_ICCV_2025_paper.pdf); [PF-LHM 2506.13766](https://arxiv.org/html/2506.13766v1)
- [V] AniGS: animatable avatar from one image in "a few minutes" on a single RTX 3090. — [arXiv 2412.02684](https://arxiv.org/pdf/2412.02684)
- [V] NEW: Dream, Lift, Animate (NVIDIA, 3DV 2026) hallucinates novel views with diffusion then lifts Gaussians into SMPL-X UV space. NEW: HumanGS (April 2026, multi-view feed-forward) and DiGS-Avatar (2026-08-21, single image in under a second) are recent preprints whose claims were not checked. — [Dream, Lift, Animate](https://research.nvidia.com/labs/dair/dream-lift-animate); [DiGS-Avatar listing](https://www.alphaxiv.org/abs/2608.20759)

**Generative novel-view video with camera control**
- [PK] ReCamMaster, TrajectoryCrafter, GEN3C (all March 2025) and CAT4D (November 2024) re-render an input video along a new camera path using video diffusion, some conditioned on a point-cloud render of estimated geometry. — [ReCamMaster 2503.11647](https://arxiv.org/abs/2503.11647); [TrajectoryCrafter 2503.05638](https://arxiv.org/abs/2503.05638); [GEN3C 2503.03751](https://arxiv.org/abs/2503.03751); [CAT4D 2411.18613](https://arxiv.org/abs/2411.18613)
- [V] NEW: FreeOrbit4D (arXiv 2026-01-26, v2 2026-05-19, SIGGRAPH 2026 conference paper) states that large-angle redirection is underconstrained and that diffusion-based methods "often break down under large-angle viewpoint changes far from the original trajectory". Its fix: split static background from foreground, complete the foreground with an object-centric multi-view diffusion model, then guide a video diffusion model with the completed 4D proxy. No orbit-angle limit or hardware is stated in the abstract. — [arXiv 2601.18993](https://arxiv.org/abs/2601.18993)
- [V] NEW 2026 follow-ups found by title/abstract only: RealCam (May 2026, autoregressive real-time interactive camera control), Vista4D and Reshoot-Anything (April 2026), SierpinskiCam (June 2026), FlashRender (September 2026, few-step), an AAAI 2026 paper on trajectory editing via generative rendering from estimated geometry, PostCam (November 2025), GRVS (March 2026). — [RealCam](https://www.alphaxiv.org/abs/2605.06051); [Vista4D](https://arxiv.org/pdf/2604.21915); [Reshoot-Anything](https://arxiv.org/pdf/2604.21776); [FlashRender](https://arxiv.org/pdf/2609.03563); [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37832); [PostCam](https://arxiv.org/pdf/2511.17185); [GRVS](https://www.alphaxiv.org/abs/2603.29734.md)

### Inferences
- Roughly 16 dB PSNR on clean synthetic single objects is far below watchable quality (static-scene splats typically score in the high 20s to 30s). A doubles pickleball rally seen from one elevated baseline camera is a harder case on every axis: four people, each perhaps 100 to 200 pixels tall, self- and mutual occlusion, fast limbs and paddles, and zero parallax because the camera is fixed. The far-side players are only ever seen from the front and the near-side players only from behind, so a 180-degree orbit must invent every unseen surface.
- Generative redirection will produce plausible pixels but will hallucinate faces, kit and paddle positions, and is not temporally stable over a multi-minute rally; published demos are short clips (seconds). It also discards the thing the user actually wants, which is an explicit, controllable scene.
- The one research piece that could add "real player look" to a parametric replay is the single-image avatar line (LHM and successors): crop a decent frame of each player, generate an animatable SMPL-X-anchored avatar, drive it with the recovered motion. Expect a recognisable kit colour and body build with a mushy or invented back and face. This is an optional cosmetic upgrade, not a foundation.
- Hardware: these codebases are built on CUDA-only components (gsplat/diff-gaussian-rasterization CUDA kernels, xformers, large video diffusion backbones typically wanting 24 to 80 GB VRAM). On the M5 Pro they are either unsupported or would need porting work. If the user wants to try them, a rented cloud GPU is the practical route.

### Gaps
- No quantitative evaluation of any of these methods on broadcast sports footage with small distant players was found.
- VRAM and runtime for Shape of Motion, MoSca, LHM, FreeOrbit4D, RealCam and peers were not verified this session.
- Genie-style interactive world models were not researched; nothing was found tying them to faithful re-rendering of a specific real video.
- 2026 papers were assessed from abstracts and titles only.

---

## 3. Sports-specific monocular free-viewpoint work: what did it achieve and what did it fake?

### Takeaway
Sports-specific work converges on exactly the parametric recipe: calibrate the camera from court lines, lift players to 3D meshes on the ground plane, and reconstruct the ball with physics between detected hits and bounces, then render in a synthetic scene. Two 2026 papers (TT4D for table tennis, Volley3D for volleyball) show this working at scale from broadcast video; none attempts photoreal novel views of the real players.

### Cited Findings
- [V] NEW: TT4D (arXiv 2026-05-02; v2 2026-09-09; UC Berkeley and Augsburg authors): a pipeline and dataset of 140+ hours of singles and doubles table tennis reconstructed from monocular broadcast video, with camera calibration, 3D ball positions, ball spin, shot segmentation and 3D human meshes over time. Key idea is "lift first": a learned network lifts the whole unsegmented 2D ball track to 3D (through occlusion), then shots are segmented in 3D. Stated applications include virtual replay and robot learning; authors say the pipeline extends to tennis or badminton. Code/data release status not stated on the abstract page. — [arXiv 2605.01234](https://arxiv.org/abs/2605.01234)
- [V] NEW: Volley3D (arXiv 2026-10-28): physics-based full-rally 3D reconstruction from monocular volleyball broadcasts with no 3D training supervision. Players: ground-plane projection corrected with jump detection and Newtonian interpolation. Ball: sub-trajectory segmentation, collision matching, and physical equations tied to reconstructed 3D hand positions. Evaluated against multi-camera benchmarks; project page has videos; code release not confirmed. — [arXiv 2610.38358](https://arxiv.org/abs/2610.38358); [project page](https://noambsf.github.io/Volley3D)
- [V] NEW: "Beyond the Broadcast" (arXiv 2025-07-26): a VR tennis viewing system that reconstructs "a simulated game" of players and ball from broadcast video data and overlays tactical visualisations; user study reports it beats conventional VR broadcast. A search-result excerpt says the authors consider direct 3D ball estimation from monocular video infeasible and use kinematic reconstruction instead. Pose model and engine not stated in the abstract. — [arXiv 2507.20006](https://arxiv.org/abs/2507.20006)
- [V] "Where Is The Ball" (June 2025): 3D ball trajectory estimation from 2D monocular tracking. Method details not read. — [arXiv 2506.05763](https://arxiv.org/pdf/2506.05763)
- [V] MonoTrack (CVPR Workshops 2022): reconstructs 3D badminton shuttle trajectories per shot from monocular broadcast video with a nonlinear optimiser over a physical flight model, using court and player pose cues. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf)
- [PK] "Soccer on Your Tabletop" (CVPR 2018): per-player depth maps predicted by a network trained on FIFA video-game data, turned into textured billboards-with-depth on a synthetic pitch for AR viewing. Players are view-dependent depth sprites, not full 3D; the ball was not reconstructed. — [arXiv 1806.00890](https://arxiv.org/abs/1806.00890)
- [V] SoccerNet Game State Reconstruction: localise and identify all players from a single moving broadcast camera to produce a "video game-like top-view minimap" (2D positions, team, jersey number), scored with GS-HOTA. Winning pipelines chain a detector, a pitch-line camera estimator, tracking with re-ID and jersey OCR. NEW: Broadcast2Pitch (WACV 2026). — [SoccerNet 2025 results](https://arxiv.org/pdf/2508.19182); [Golovkin et al., CVPRW 2025](https://arxiv.org/pdf/2504.06357); [Broadcast2Pitch](https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf)
- [V] NEW: SoccerNet 2026 challenge report (July 2026) includes a novel-view-synthesis track built on multi-camera data with 3DGS and Triangle Splatting baselines; best team gained over 3 dB PSNR. Ground-level broadcast cameras were 59% of evaluation views but under 2% of training cameras, causing artifacts. NEW: DENSER (June 2026) is a depth-guided alias-free splatting entry. These are multi-view static-scene tasks, not monocular. — [SoccerNet 2026 results](https://arxiv.org/pdf/2607.07320); [DENSER](https://arxiv.org/pdf/2606.01419)

### Inferences
- What these systems fake, consistently: (1) appearance (generic or game-asset bodies, synthetic court); (2) ball depth, which is inferred from physics plus anchor events (bounce on a known plane, contact at a reconstructed hand/racket) and not measured; (3) player depth, which comes from assuming feet on the ground plane, with special handling for jumps; (4) rackets/paddles, usually inferred from wrist pose or omitted.
- Pickleball is a favourable case relative to these papers: fixed camera (calibrate once from court lines), small court, at most four players, a ball that bounces often (many ground-plane anchors), low spin relative to table tennis, and rare high jumps. The hard parts are the small, fast, often motion-blurred ball against busy backgrounds, the far-side players' low pixel count, and paddle orientation.
- TT4D's "lift then segment" and Volley3D's "anchor to hand contacts plus Newtonian segments" are the two design patterns to borrow for the ball layer. Both support the user's framing: the output is a physically consistent trajectory that reprojects onto the 2D detections, not a measured one.
- No Gaussian-splatting paper was found that produces a monocular broadcast sports replay with real-player appearance. The absence is itself a signal about feasibility.

### Gaps
- No pickleball-specific 3D reconstruction paper was found (searches were not pickleball-targeted; a sibling researcher may cover this).
- TT4D and Volley3D code availability, licences and compute needs were not confirmed.
- Tennis-specific monocular 3D reconstruction papers (beyond "Beyond the Broadcast") were not surfaced.
- "Soccer on Your Tabletop" details are from prior knowledge, not re-read.

---

## 4. Are Meta SAM 3D Objects and other single-image-to-3D models useful for static assets or player avatars, and do they run on a Mac?

### Takeaway
Single-image-to-3D models are useful for props and set dressing (paddle, net posts, benches, signage, umpire chair) and TRELLIS.2 now runs natively on Apple Silicon; they are the wrong tool for the court itself (regulation geometry, model it by hand) and a weak tool for player avatars. SAM 3D Body is the more relevant half of Meta's release because it outputs a permissively licensed body rig (MHR).

### Cited Findings
- [V] NEW: Meta announced SAM 3D on 2025-11-19 alongside SAM 3, as two models: SAM 3D Objects (object and scene reconstruction from one image) and SAM 3D Body (human pose and shape), with checkpoints, inference code and a new parametric human model. — [Meta AI blog](https://ai.meta.com/blog/sam-3d/); [Meta Newsroom](https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/amp/)
- [V] SAM 3D Objects repo: code and checkpoints under the "SAM License"; quick-start exports a Gaussian splat PLY (`output["gs"].save_ply`); predicts shape, texture and layout; claims robustness to occlusion and small objects; includes a notebook to align outputs with SAM 3D Body meshes. README states no hardware, macOS or MPS information. — [GitHub: facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)
- [V] Reported limitations: objects are reconstructed in isolation without physical interaction reasoning; output meshes are moderate resolution with blurred fine detail. — [Roboflow blog](https://blog.roboflow.com/sam-3d/); [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/metas-sam-3-image-segmentation-suite-analyzes-and-creates-3d-bodies-and-other-objects)
- [V] SAM 3D Body outputs the Momentum Human Rig (MHR), which separates skeleton from soft-tissue shape. — [Roboflow blog](https://blog.roboflow.com/sam-3d/)
- [V] NEW: TRELLIS.2 (Microsoft) has a community Apple Silicon port, `trellis-mac`, using PyTorch MPS: needs M1 or later, Python 3.11+, at least 24 GB unified memory, about 15 GB of weights; author reports 400K+ vertex meshes from one image in about 3.5 minutes on an M4 Pro. Surfaced around April 2026. Port code is MIT, but dependencies DINOv3 (Meta custom licence) and RMBG-2.0 (CC BY-NC 4.0) carry commercial restrictions. An MLX port (`trellis2mlx`) is partial. — [Show HN](https://hn.svelte.dev/item/47828896); [hands-on, M1 Max 64 GB](https://lilting.ch/en/articles/trellis2-m1-max-hands-on); [trellis2mlx](https://github.com/lyonsno/trellis2mlx)
- [V] Hunyuan3D 2.x on Mac is community-driven: Pinokio installer (50 GB+ weights) and a `Hunyuan3D-2.1-mac` fork with MPS; "Modly" bundles Hunyuan3D 2 variants and TRELLIS.2 GGUF for local use on Apple Silicon. — [Codersera guide](https://codersera.com/blog/how-to-install-and-run-hunyuan3d-2-on-macos-a-step-by-step-guide/amp/); [Modly write-up](https://themenonlab.blog/blog/modly-image-to-3d-local-ai)
- [PK] Original papers: TRELLIS (December 2024) and Hunyuan3D 2.0 (January 2025). — [TRELLIS 2412.01506](https://arxiv.org/abs/2412.01506); [Hunyuan3D 2.0 2501.12202](https://arxiv.org/abs/2501.12202)

### Inferences
- Court, lines, kitchen and net are specified by rulebook dimensions (court 20 ft by 44 ft, 7 ft non-volley zone, net 36 in at posts and 34 in at centre; these are standard figures stated from general knowledge, check the USA Pickleball rulebook). A hand-built parametric model is more accurate, lighter and cleaner than any generated mesh, and exact geometry is required anyway for camera calibration and ball-bounce physics.
- Image-to-3D inputs from a broadcast frame are poor: props are small, compressed and partly occluded. Better results come from feeding a clean product photo of a paddle or a stock image of a net post than a crop from the match.
- For player avatars, image-to-3D models output a static, unrigged mesh in whatever pose the crop shows; it then needs auto-rigging and will not match the motion skeleton. A cleaner route is a parametric body (MHR or Anny, section 5) with shape fitted by SAM 3D Body and a simple kit-colour material, optionally upgraded later with an LHM-style avatar.
- SAM 3D Objects on Mac is unverified and likely CUDA-bound (it derives from the TRELLIS-style sparse-voxel stack). TRELLIS.2 via `trellis-mac` is the verified local option; the 64 GB machine exceeds its 24 GB floor.

### Gaps
- SAM License terms (commercial use, redistribution) were not read. SAM 3D Objects hardware requirements and macOS support are not stated in the README and were not found elsewhere.
- Whether SAM 3D Body runs on MPS was not verified (a sibling researcher on pose estimation may cover it).
- Hunyuan3D 3.x and TRELLIS.2 official release dates and licences were not verified; Mac port claims rest on community blogs and repos.

---

## 5. Viewer and engine tooling for a parametric replay on macOS: tools, file formats, and body-model licences

### Takeaway
A browser viewer (Three.js or React Three Fiber) playing glTF/GLB skinned-mesh animation is the lowest-friction "watch from any angle" target on a Mac, with viser or Rerun as the Python-side debug viewers during pipeline development. Use an Apache-2.0 body (Meta MHR or NAVER Anny) instead of SMPL/SMPL-X, whose licence is non-commercial and forbids redistribution.

### Cited Findings

**Body-model licences**
- [V] SMPL model licence: limited to non-commercial scientific research, education or artistic projects; commercial use prohibited, including training networks for commercial use; no copying, sharing, distribution or sublicensing (one archive copy allowed); commercial licences via Meshcapade. — [SMPL model licence](https://smpl.is.tue.mpg.de/modellicense.html)
- [V] NEW: Meta MHR (Momentum Human Rig; released with SAM 3D, November 2025): Apache 2.0. 45 identity parameters, 204 pose parameters, 72 expression parameters, neural pose correctives, 7 levels of detail (TorchScript model supports LOD 1 only), FBX rig assets per LOD with identity and expression blendshapes, and a `tools/mhr_smpl_conversion` tool for MHR to/from SMPL and SMPL-X. Requires Python 3.11+, PyTorch (CPU or GPU), pymomentum 0.1.90+. No glTF exporter is mentioned; supported operating systems are not listed. — [GitHub: facebookresearch/MHR](https://github.com/facebookresearch/MHR)
- [V] NEW: Anny (NAVER LABS Europe; arXiv November 2025): body model and code under Apache 2.0, built from open MakeHuman-lineage assets instead of scans, shape calibrated to WHO population statistics, 13,380 vertices, 163-bone skeleton, supports multiple rig conventions including Mixamo. Does not model clothing. — [arXiv 2511.03589](https://arxiv.org/abs/2511.03589v2)
- [PK] SKEL (biomechanical skeleton inside SMPL, MPI): research licence in the SMPL family, so not a permissive alternative. — [SKEL project page](https://skel.is.tue.mpg.de/)

**Viewers**
- [V] viser ships an official SMPL skinned-mesh example that updates bone transforms instead of re-uploading vertices; it renders in the browser via three.js; SMPL files must be downloaded separately. — [viser SMPL skinned demo](https://viser.studio/main/examples/demos/smpl_skinned/)
- [V] Rerun: the `moverseai/rerun-animation` plugin turns the Rerun viewer into a 3D animation viewer accepting BVH and SMPL-family `.npz` parameter files. — [GitHub: moverseai/rerun-animation](https://github.com/moverseai/rerun-animation)
- [V] An RSS 2026 robotics paper comparing viewers reports viser scales well on large point clouds. — [RSS 2026 proceedings](https://www.roboticsproceedings.org/rss22/p175.pdf)
- [PK] Three.js loads glTF 2.0 with skins and animation clips through `GLTFLoader` and plays them with `AnimationMixer` (scrubbing, time scale, cross-fades); `OrbitControls` gives the free camera. React Three Fiber wraps the same engine. — [three.js GLTFLoader docs](https://threejs.org/docs/#examples/en/loaders/GLTFLoader); [three.js AnimationMixer docs](https://threejs.org/docs/#api/en/animation/AnimationMixer); [React Three Fiber](https://r3f.docs.pmnd.rs/)
- [PK] Blender runs natively on Apple Silicon with a Metal backend, imports/exports glTF, FBX, BVH and USD, and is scriptable in Python; Meshcapade publishes SMPL-X Blender add-ons (under SMPL-family licence terms). — [Blender](https://www.blender.org/download/); [Blender glTF add-on manual](https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html)
- [PK] Godot 4 (MIT licence, native macOS, glTF as its preferred 3D interchange format), Unity and Unreal Engine 5 all run on Apple Silicon; Apple RealityKit consumes USDZ and is the path to a visionOS tabletop view. — [Godot](https://godotengine.org/); [Godot glTF docs](https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_3d_scenes/available_formats.html); [RealityKit](https://developer.apple.com/documentation/realitykit)

### Inferences
- Recommended interchange design (researcher's judgement): keep a neutral intermediate, for example per-frame root translation plus joint rotations per player, ball position per frame, and event markers, in NPZ or JSON at a fixed frame rate. Export from that to (a) one GLB per rally containing the court, four skinned bodies with baked animation clips and an animated ball node, for the web viewer, and (b) USD/USDZ later if a Vision Pro tabletop view is wanted. BVH is fine for skeleton-only exchange with Blender but carries no mesh or shape.
- Baking to glTF skeletal animation (joint rotations on a fixed rig) keeps files small: four players at 60 fps for a 10-minute game is on the order of a few million rotation keys, versus gigabytes for per-vertex caches. Per-player body shape is constant, so bake shape into the bind mesh once.
- Three.js/R3F fits the "even if it's just a mockup" requirement best: zero install to view, orbit/fly cameras, timeline scrubbing and slow motion from `AnimationMixer`, preset cameras (broadcast, opposite baseline, sideline, player POV, top-down) and trails or bounce markers are all small features. A game engine (Godot, Unity, Unreal) only pays off if the user wants nicer lighting, crowds or VR.
- Practical MHR path: SAM 3D Body gives MHR parameters per frame; MHR ships FBX rigs; convert FBX to glTF via Blender once, then drive the same skeleton in three.js. The absence of a first-party glTF exporter means this conversion step needs a small Blender script. Anny's Mixamo-compatible rig option is an alternative that opens up retargeting to stock Mixamo characters.
- If the pose estimator in use outputs SMPL parameters, the user can still do personal, non-commercial experiments with SMPL, but should not publish the model files in a public repo or web app. Converting to MHR or Anny, or retargeting joint rotations onto any CC0/Mixamo rig, avoids the issue.
- viser is the best fit for the development loop (Python process pushes meshes, point clouds, camera frusta and the video frame side by side to a browser tab); Rerun is the better fit for time-series inspection of the pipeline (2D detections, 3D tracks, scalar plots on one timeline).

### Gaps
- No off-the-shelf three.js viewer for SMPL/MHR sequence playback was found in search; the three.js capabilities above are from prior knowledge of the library, not verified this session.
- Current Apple Silicon status of Unreal Engine 5.x and Unity 6 editor features was not verified.
- Whether pymomentum publishes macOS arm64 wheels was not verified.
- Anny and MHR pose-corrective quality relative to SMPL-X in fast athletic poses was not assessed.

---

## 6. Physics engines for the "physically grounded" layer on a Mac: ball flight and bounce, and humanoid motion tracking

### Takeaway
The ball needs no physics engine: a hand-written point-mass ODE with gravity, drag, Magnus force and a restitution/friction bounce, fitted to the 2D detections, is what the sports papers use. For players, MuJoCo on CPU is the dependable Mac choice; NVIDIA's Newton and MuJoCo Warp are CPU-only on macOS, Genesis lists a Metal backend, and full physics-based humanoid tracking is a stretch goal compared with kinematic smoothing plus foot-contact constraints.

### Cited Findings
- [V] Newton (NVIDIA Warp-based engine; 1.0 docs, with a 1.2.1 release candidate on PyPI): supported on Linux, Windows and "macOS (CPU only)". — [Newton installation guide](https://newton-physics.github.io/newton/1.0.0/guide/installation.html); [PyPI: newton](https://pypi.org/project/newton/)
- [V] NVIDIA Warp: Apple Silicon macOS is CPU-only; GPU acceleration requires CUDA. — [Warp compatibility](https://nvidia.github.io/warp/latest/user_guide/compatibility.html)
- [V] MuJoCo Warp (MJWarp): "requires an NVIDIA GPU for fast simulation but supports CPU for development and debugging". — [GitHub: google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp); [MJWarp docs](https://mujoco.readthedocs.io/en/stable/mjwarp)
- [V] MJX (MuJoCo on JAX): official docs list Apple Silicon among supported platforms alongside NVIDIA/AMD GPUs and TPUs. A community guide says GPU use on Mac goes through jax-metal (unverified, third-party). — [MJX docs](https://mujoco.org/book/mjx.html); [community Mac playbook](https://svah-x.github.io/mac-playbooks/playbooks/mujoco)
- [V] NEW: A community `Mujoco-mlx` project ports MJX to Apple MLX for Metal acceleration; self-reported validation, not an official release. — [Mujoco-mlx](https://gh.r-mj.com/RobotFlow-Labs/Mujoco-mlx)
- [V] Genesis: Python simulator supporting Linux, macOS and Windows with NVIDIA, AMD, Apple Metal, Vulkan and CPU backends; v0.3.0 released 2025-08-05, and a `genesis-world` 1.1.2 package exists on PyPI. Backend claims come from the repo listing and a third-party wiki. — [GitHub: Genesis-Embodied-AI](https://github.com/Genesis-Embodied-AI/genesis-world); [PyPI: genesis-world](https://pypi.org/project/genesis-world/1.1.2/); [DataCamp overview](https://www.datacamp.com/blog/genesis-physics-engine)
- [V] Sports papers implement ball physics directly: Volley3D uses Newtonian sub-trajectories anchored to contacts; MonoTrack uses a nonlinear optimiser over a flight model per shot; TT4D uses a learned lifting network that also infers spin. — [Volley3D](https://arxiv.org/abs/2610.38358); [MonoTrack](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf); [TT4D](https://arxiv.org/abs/2605.01234)
- [PK] MuJoCo (Apache 2.0, DeepMind) ships native macOS arm64 builds and a Python package; classic CPU MuJoCo is single-scene fast and widely used for humanoid control. — [MuJoCo](https://mujoco.org/); [GitHub: google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- [PK] Rapier (Rust, with official JavaScript/WASM bindings) and Jolt Physics (C++, with a community WASM build) are rigid-body engines usable in the browser alongside three.js; Blender has built-in rigid-body physics (Bullet). — [Rapier](https://rapier.rs/); [Jolt Physics](https://github.com/jrouwe/JoltPhysics); [Blender rigid body manual](https://docs.blender.org/manual/en/latest/physics/rigid_body/index.html)

### Inferences
- Ball layer: fit, do not simulate forward blindly. Segment the 2D track at hits and bounces; for each segment solve for initial 3D position, velocity (and optionally spin) that minimises reprojection error under gravity plus quadratic drag plus Magnus, with constraints that segments meet at bounce points on the court plane (z = ball radius) and at paddle contacts near a player's hand. A plastic perforated pickleball has high drag, so drag must be in the model. SciPy `least_squares` on a Mac CPU is ample. The result is smooth and physically plausible by construction even where detections are missing, which matches the user's stated priority.
- A browser engine (Rapier/Jolt) is only needed for "what if" interactivity (the viewer nudging a shot). For faithful playback, bake the fitted trajectory into the GLB.
- Player layer, in rising order of effort: (1) kinematic clean-up: temporal smoothing, ground-plane foot locking, inverse kinematics to remove foot skate and floor penetration; (2) physics-based tracking, where a simulated humanoid is controlled to imitate the reference motion (the PHC/DeepMimic style of approach; not researched this session), which removes floating and jitter but needs a trained controller, normally trained on thousands of parallel GPU environments. On the Mac, running a pretrained tracking policy in CPU MuJoCo is realistic; training one locally is not, given that Newton, MJWarp and Warp are CPU-only there. Genesis-on-Metal and MJX-on-Metal are possible but less trodden paths.
- Recommendation implied by the evidence: start with (1); it is what Beyond Sports' "enhancement system" amounts to and it delivers most of the perceived smoothness. Treat (2) as an optional later experiment, possibly with cloud GPU training.

### Gaps
- No verified benchmark of Genesis or MJX GPU performance on Apple Silicon was found; the Metal claims rest on repo text and community guides. jax-metal's maturity for MJX is uncertain.
- Availability of pretrained, Mac-runnable humanoid motion-tracking controllers (and their body-model licences) was not researched.
- Pickleball-specific aerodynamic coefficients (drag, lift, restitution on court surfaces) were not researched; a sibling researcher may cover ball physics.
- Isaac Sim / Isaac Lab were not researched; they require NVIDIA RTX hardware (prior knowledge, unverified this session), so they are not Mac options.

---

## Cross-cutting assessment (inferences drawn from the sections above)

- **Realistically achievable on the M5 Pro today:** a parametric replay in the AO Animated mould. Hand-modelled regulation court; four generic or lightly personalised Apache-licensed bodies driven by monocular motion capture with kinematic clean-up; a physics-fitted ball; a three.js viewer with free camera, scrubbing and slow motion. Every layer has published precedent from monocular broadcast video (TT4D, Volley3D, MonoTrack, SoccerNet-GSR).
- **Research-grade, optional:** single-image animatable Gaussian avatars (LHM line) to give players a real-kit look; physics-based humanoid tracking for contact realism; generated props via TRELLIS.2 on MPS. Each is CUDA-first or needs porting, except TRELLIS.2.
- **Not yet feasible from one fixed broadcast camera:** photoreal free viewpoint of the actual players at 90 to 180 degrees from the source view, i.e. what the 36-camera 超视角 rig or Canon's 100+ camera arenas deliver. Published monocular 4D methods sit near 16 dB PSNR on easy synthetic cases at large offsets, and 2026 generative methods still name large-angle redirection as their failure mode.
- **The honest framing for the user:** the National Games system re-renders captured light; the achievable system re-stages the match with stand-ins. Positions, timing, shot shapes and footwork patterns can be right enough to be recognisable; faces, exact limb poses, paddle angles and ball spin will be approximations.
