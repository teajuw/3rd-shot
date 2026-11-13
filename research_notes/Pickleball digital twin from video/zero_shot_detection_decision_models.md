# Zero-shot visual detection and localization with decision models, VLMs and promptable segmentation (for pickleball broadcast video)

Research date: 2026-11-11. "NEW" marks anything published after mid-2025, with the date. Many of the decision-model sources are third-party listings or blogs; where I did not reach a primary source this is stated. Numbers are as reported by the cited source; "self-reported" means vendor numbers with no independent replication found.

## 1. What are "decision models" as a model class, and can they localize?

### Takeaway
Decision models (TypeSafe Jev and its open imitators: LiquidAI d1, StartLux Decision, Kev, Cloudflare Clef, OpenAI Decisions API) are a class that appeared in September-October 2026: they take a state (text/JSON, sometimes images) plus typed questions and return yes/no probabilities, choices or ratings in one forward pass with no generated text. Only Jev 1.14 (hosted, experimental) has a box output; none of the open/local ones has a box or point output, and I found no published visual-grounding or small-object benchmark for any of them, so their role in a ball pipeline is candidate verification ("is there a ball in this crop?"), not localization.

### Cited Findings

Jev (TypeSafe) - NEW, 2026-09-15
- TypeSafe came out of stealth on 2026-09-15 with Jev as its first model, called a "System One model"; hosted only, weights unpublished. `jev-1.14.0` was announced 2026-10-14. — [flaviocopes.com Jev deep dive](https://flaviocopes.com/jev.md)
- Question types: Noul (yes/no probability 0-1), Choice (up to 255 options, with probabilities and confidence), Score (2-10 levels), and Box (experimental; requires `jev-1.14.0` plus the image header). — [flaviocopes.com](https://flaviocopes.com/jev.md)
- Image input is experimental ("danger zone", header `TypeSafe-Danger-Zone-Images: true`), PNG/JPEG, images are rejected rather than auto-resized; TypeSafe says image input and Box are not ready for production. — [flaviocopes.com](https://flaviocopes.com/jev.md)
- Box returns `top/bottom/left/right` in 0-1 plus `confidence`, and always returns a rectangle even when the target is absent, so the docs pair it with a Noul visibility question. No accuracy or grounding benchmark for Box is given. — [flaviocopes.com](https://flaviocopes.com/jev.md)
- Latency quoted by TypeSafe: 70-500 ms end to end, most around 100 ms; price $0.042 per million input tokens, output free. — [flaviocopes.com](https://flaviocopes.com/jev.md)
- Calibration: trained with "Reinforcement Learning for Calibrated Decisions (RLCD)"; TypeSafe has published no paper and chose not to publish public benchmark results. Stated weaknesses include counting, precise arithmetic, and Score being weakly calibrated across levels. — [flaviocopes.com](https://flaviocopes.com/jev.md)
- Endpoint is `POST https://api.typesafe.ai/v1/systemone` (model, state, questions). — search summary of [aimlapi docs](https://docs.aimlapi.com/api-references/decision-models/typesafe/jev) and [OpenRouter guide](https://openrouter.ai/docs/guides/community/jev.md) (secondhand; official docs not opened)

LiquidAI d1-3B - NEW, 2026-10-07 (date from secondary source)
- d1-3B is a 3.12B-parameter decision model built on LFM2.5-VL-3B with a SigLIP2 NaFlex 400M vision encoder; "not a chat model", returns answers in one forward pass with zero output tokens. Question types are noul, choice and score only; no box/point output is mentioned. — [HF model card LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)
- Release date 2026-10-07 is reported by a secondary source, not the card. — [mer.vin](https://mer.vin/news/liquid-ai-d1-3b-a-3b-decision-model-that-answers-in-one-forward-pass/)
- Card latency on Apple M5 Pro (mps): 30 ms one question, 41 ms three questions, 62 ms for a 384 px image, 640 ms for a 3.4k-token state, 78 packed states/s. RTX 4090: 8 ms / 17 ms image. — [HF model card](https://huggingface.co/LiquidAI/d1-3B)
- Vision benchmarks on the card: CV-Bench 82.1 (base LFM2.5-VL-3B scores 87.6, i.e. the decision model is lower), BLINK 59.2, POPE 88.5, MMVP 77.0, MMStar 59.9, vision mean 74.1 vs base 73.9; ImajevBench 64.0 (base 66.8; 45.1 with images removed). No grounding/localization benchmark is listed. — [HF model card](https://huggingface.co/LiquidAI/d1-3B)
- Decision Index 0.2.1 score 48.57 ("best decision model under 10B"); self-reported. License tag `lfm1.0`; 26 quantized variants for llama.cpp/Ollama/LM Studio exist; MLX is not mentioned; requires `trust_remote_code=True` and `transformers>=5.14`. The card notes a newer `LiquidAI/d1-2B`. — [HF model card](https://huggingface.co/LiquidAI/d1-3B)
- A secondary source says the LFM Open License v1.0 has a revenue condition for commercial use. — [mer.vin](https://mer.vin/news/liquid-ai-d1-3b-a-3b-decision-model-that-answers-in-one-forward-pass/)

StartLux Decision - NEW, autumn 2026 (no primary card reached)
- Third-party listings describe a family from 0.8B to 35B that answers typed questions (choice, yes/no, rating) with per-option probabilities; 4B and 9B accept text, JSON and images. — [Featherless listing 4B](https://featherless.ai/models/startlux-models/StartLux-Decision-v1-4B)
- `StartLux-Decision-v1.1-9B-Instant` is listed as a 9B model taking text or up to 8 images, weights CC BY-NC 4.0 (non-commercial; commercial needs a separate license). — [Featherless listing 9B Instant](https://featherless.ai/models/startlux-models/StartLux-Decision-v1.1-9B-Instant)
- A 36kr article reports the 27B model scoring 63.88 vs Jev 1.13's 57.91 on Decision Index 0.2.1 (StartLux self-reported). — [36kr](https://eu.36kr.com/en/p/4011046457872258)
- No box/point output and no MLX build were mentioned in any listing found. — [Featherless listing 9B Instant](https://featherless.ai/models/startlux-models/StartLux-Decision-v1.1-9B-Instant)

Kev - NEW, 2026-09-20/25 (sources disagree)
- Kev is "Small Jev-like decision models you can train and run yourself" by Jared Palmer, Apache-2.0: Kev-0.8B/4B/9B on Qwen3.5 bases (rank-16 LoRA plus a pointer head on a frozen base) and Kev-27B on Qwen3.8-27B. Question types noul/choice/score; inputs are text; no image input and no box type are mentioned. — [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- The "pointer head" scores each option's `</opt>` hidden state against the question's `<decide>` hidden state (it is an option selector, not a spatial pointer). — [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- Runs on PyTorch (CUDA, ROCm, MPS, CPU) and MLX; Kev-4B on an Apple M5: 721 ms for five questions, 136 ms with the text cached; H100: 18.1 ms for six questions. — [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- Release date is given as 2026-09-25 by OpenRouter and 2026-09-20 by another listing. — [OpenRouter kev-4b](https://openrouter.ai/jaredpalmer/kev-4b); [gradually.ai](https://www.gradually.ai/en/ai-models/kev-4b/)

Others
- "OpenJev": no specific model found. The only reference is that community models named "Open-Jev" or "Jev-Style" on Hugging Face are independent Qwen-based projects, none of which is Jev. — [flaviocopes.com](https://flaviocopes.com/jev.md)
- Cloudflare Clef (NEW, 2026-10-01): open weights, Apache 2.0, built on Qwen, same request format as Jev; reportedly needs a single NVIDIA H200. — [flaviocopes.com](https://flaviocopes.com/jev.md)
- OpenAI Decisions API (NEW; announcement 2026-09-29, public beta 2026-10-06, GA 2026-10-22 per differing reports): `POST /v1/decisions`, only `gpt-6-luna`, text and image inputs; OpenAI claims about 150 ms vs about 1.6 s for a standard Luna call; no independent measurement and no accuracy comparison published. — [aiweekly.co](https://aiweekly.co/alerts/openai-launches-decisions-api-public-beta-on-gpt-6-luna); [unite.ai](https://unite.ai/openai-releases-decisions-api-in-public-beta-powered-by-gpt-6-luna)
- Jev Decision Index (NEW, 2026-09-22): a Hugging Face Space leaderboard posted by multimodalart scoring decision models on a frozen suite of 132,422 requests across 37 benchmarks (19 scored) in five areas: Tools & Automation, Retrieval & Classification, Language Understanding, Knowledge & Reasoning, Arts & Human Judgment. None of the named areas is visual grounding. — [aiweekly.co](https://aiweekly.co/alerts/jev-decision-index-ranks-30-open-weight-decision-models)

### Inferences
- The user's local measurements (d1-3B about 100-150 ms per image decision on MPS; CV-Bench 82.1 reproduced) are consistent with the card (62 ms for a 384 px image on M5 Pro; CV-Bench 82.1). Larger crops or more questions would explain the gap.
- Because open decision models have no coordinate output, localization requires probing (grid/tiles) or an external candidate generator. The user's 8x8 grid at about 7 s/photo is about 110 ms per cell; at 30 fps that is roughly 210 s of compute per second of video, so whole-frame grid probing is not viable for a full match. Verifying a handful of motion candidates per frame (e.g. 3-5 crops at 60-150 ms each) is roughly 0.2-0.75 s per frame, feasible offline but still far slower than a small trained detector.
- A 5-15 px ball inside a 384 px tile is the regime where even the base VLM's fine perception is weakest (CV-Bench is not a small-object benchmark); a yes/no verifier should be given tight upscaled crops around motion candidates, and its P(yes) must be validated on labeled pickleball crops before trusting calibration claims, which were established on text-style tasks.
- Jev Box is the only decision-model localizer, and it is hosted, experimental, hallucinates a box when the target is absent, and has no published accuracy. Treat it as untested for tiny balls.
- Licenses matter for the local options: Kev is Apache-2.0 but text-only; d1-3B (LFM license with a revenue condition) and StartLux (CC BY-NC) accept images.

### Gaps
- No public information found for a model literally named "OpenJev"; no Hugging Face primary card reached for StartLux (only Featherless/36kr listings), so its image handling and "8-bit MLX" build are unverified from public sources.
- No independent evaluation of any decision model on RefCOCO, pointing, small-object detection or sports imagery. The Decision Index has no stated vision track.
- No published accuracy for Jev Box; no calibration curves for image inputs for any of these models.
- Official TypeSafe docs (docs.typesafe.ai) and OpenAI developer docs were not opened; details above come from blogs and aggregators.

## 2. How good are frontier LLMs/VLMs at bounding boxes and pointing, especially on tiny objects?

### Takeaway
On saturated benchmarks (RefCOCO) modern VLMs exceed 90%, but on harder grounding tests the best model scored 45% and most frontier models scored near 0% on tiny objects and on "object absent" rejection; the belief that "the latest LLMs draw boxes extremely well" holds for clear, medium-to-large objects, not for a 5-15 px blurred ball. Small-scale practitioner evidence in soccer agrees: Gemini 3 found distant balls that Qwen3-VL-8B, InternVL3.5-8B and a fine-tuned YOLOv11 missed, at about 2 s per image.

### Cited Findings
- GroundingME (NEW, arXiv 2025-12-19; 1,005 samples, Accuracy@0.5 IoU): recent models exceed 90% on RefCOCO, yet the best model here, Qwen3-VL-235B-A22B, reaches 45.1% total; most models fall in 10-40%. — [GroundingME](https://arxiv.org/html/2512.17495v1)
- GroundingME "Limited-Small" (tiny objects in HR-Bench 8K images) scores: Gemini-2.5-Pro 0.7, Gemini-2.5-Flash 3.3, Qwen2.5-VL-72B 0.7, GLM-4.5V 10.7, Seed-1.6-Vision 15.3, Qwen3-VL-32B 7.3, Qwen3-VL-8B 16.0, Qwen3-VL-2B 36.0, Qwen3-VL-4B 48.7, Qwen3-VL-30B-A3B 52.0, Qwen3-VL-235B-A22B 35.3. — [GroundingME Table 3](https://arxiv.org/html/2512.17495v1)
- GroundingME Rejection (object absent): nearly all models score at or near 0% (Gemini-2.5-Pro 7.0, Llama-4-Maverick 6.0, Qwen3-VL all 0.0), i.e. they hallucinate a box rather than say "not present". Thinking mode adds roughly 4.7-7.4 points. — [GroundingME](https://arxiv.org/html/2512.17495v1)
- SOUBench (NEW, arXiv 2604.22884, April 2026; 18,204 VQA pairs on small objects in driving/aerial/underwater scenes, COCO small-object definition): best of 15 MLLMs is GPT-5.2 at 58.97%, trailing humans by 23.53 points; Gemini-3-flash and Qwen3-VL-8B were among those tested. — [SOUBench](https://arxiv.org/html/2604.22884v1)
- Point-Bench image pointing (from the Molmo2 paper, NEW, v4 2026-04-02): Molmo2-8B 68.7, Molmo2-4B 67.7, Poivre-7B 67.5, Gemini-Robotics-ER-1.5 67.1, Gemini-2.5-Pro 62.8, Qwen2.5-VL-72B 59.0, Qwen3-VL-235B 58.3; human 89.1. — [Molmo2 paper](https://arxiv.org/html/2601.10611v4)
- Gemini 3 Pro (NEW, Google blog 2025-12-05): claims pointing with "pixel-precise coordinates", chaining 2D points for trajectories, a `media_resolution` parameter, and tuning for fast action when sampling above 1 fps (golf-swing example at 10 fps). No pointing or grounding benchmark numbers are given in the post. — [Google blog](https://blog.google/technology/developers/gemini-3-pro-vision/)
- Practitioner test (NEW, PyTorch Conference Europe, Paris, 2026-04-08, Future Processing): on a Roboflow soccer-ball image set, a fine-tuned YOLOv11, Qwen3-8B and InternVL-3.5-8B all failed on small distant balls ("in neither of the tested cases, these balls were detected"); Gemini 3 (thinking) gave the best accuracy and found balls in simple and complex scenes at "no more than 2 seconds" per image vs about 30 ms for YOLOv11. The slides give no aggregate accuracy numbers, and the Qwen run shown returned placeholder coordinates. — [slides PDF](https://hosted-files.sched.co/pytorchconferenceeu2026/33/M_Szymkowski_PyTorch_Conference.pdf)
- Roboflow VLM comparison (NEW, 2026-08-04) on RF20-VL-FSOD (20-dataset subset of RF100-VL): ranking GPT-6 Astra, then Gemini 3.8 Flash, Qwen3.8-Max, Claude Fable 5.1; no numeric scores on the page. Caveats stated: VLM outputs have no reliable confidence scores (which hurts mAP), few-shot examples sometimes hurt Qwen and Claude, and RF100-VL is public so pre-training exposure cannot be ruled out. — [Roboflow blog](https://roboflow.com/blog/vision-language-model-benchmarks)
- Efficiency data point (NEW, January 2026): EGM-Qwen3-VL-8B reaches 91.4 IoU on RefCOCO at 737 ms average latency vs Qwen3-VL-235B needing 4,320 ms for 90.5. — [arXiv 2601.13633](https://arxiv.org/html/2601.13633)
- Zero-shot "where is the ball" from context (NEW, DEV Community, posted July 27, year not shown, presumably 2026): unnamed frontier VLMs scored 53% on off-center soccer balls from pixels (the author calls it a coin flip) and 35-38% from player tracks as text, while a 14k-parameter DeepSets model scored 82% and a fine-tuned ResNet-18 scored 74%. Note this is ball-location inference with the ball hidden, not detection. — [DEV Community](https://dev.to/javieraguilarai/wheres-the-ball-part-2-a-60-kilobyte-model-two-sports-and-the-signal-vlms-miss-5947)

### Inferences
- For a ball occupying roughly 0.002-0.01% of a 1080p frame, frontier VLMs on the whole frame will be unreliable; all the evidence above says performance collapses on tiny objects, and Gemini-family models were notably weak on GroundingME's small subset even though Gemini 3 looked best in one informal soccer test. Cropping/tiling so the ball spans tens of pixels is a prerequisite.
- The near-zero rejection scores mean a VLM asked "box the ball" on a frame where the ball is occluded or out of view will usually invent one. Any VLM labeler needs an explicit presence check and a temporal/physics filter.
- Among open models, the small Qwen3-VL sizes (4B, 30B-A3B) scored best on tiny objects in GroundingME; bigger was not better (32B: 7.3). This is one benchmark; it should be verified on pickleball frames before selecting a local model.
- At about 2 s and API cost per frame, a hosted frontier VLM is only practical as a sparse auto-labeler (hundreds to low thousands of frames), not as the per-frame detector for a full match.

### Gaps
- No benchmark found that measures box or point accuracy on 5-15 px objects with motion blur specifically; no VLM result on any racket-sport ball detection dataset.
- No ScreenSpot, ODinW or LVIS zero-shot numbers for Gemini 3.x, GPT-5.x/6 or Claude 4.x/5.x were retrieved; Roboflow's RF20-VL numbers are not in the article text.
- Gemini's segmentation-mask output quality was not found in any measured source.
- The PyTorch Conference slides report qualitative outcomes only; test split was 44 images.

## 3. Best open local grounding/detection models and Apple Silicon feasibility

### Takeaway
The locally runnable candidates with verified Apple Silicon paths are SAM 3/3.1 (MLX via mlx-vlm, measured on M3 Max), Moondream 3 (community MLX 4-bit), Molmo 2 / MolmoPoint, Qwen3-VL, Florence-2, PaliGemma 2 and LFM2-VL (all listed in mlx-vlm); the strongest open detectors by LVIS score (Grounding DINO 1.5 Pro, DINO-X Pro) are distributed via API, and several attractive new models carry non-commercial licenses. Measured Apple Silicon speeds for grounding tasks are scarce.

### Cited Findings
- Moondream 3 Preview: RefCOCO 91.1, RefCOCO+ 81.8, RefCOCOg 88.6 per Moondream docs; native skills include pointing, counting and object detection; weights under Business Source License 1.1 with an Additional Use Grant (no third-party service). Moondream 3.1 is a preliminary release under a new "Moondream Model License" and reports COCO val F1@0.5 of 81.46 (a different metric from the mAP 51.2 listed for an earlier model). — [Moondream docs](https://docs.moondream.ai/); [Moondream 3.1 model page](https://moondream.ai/models/moondream_3-1_9B_A2B); [Moondream models](https://moondream.ai/models)
- A community 4-bit MLX quantization of Moondream 3 Preview runs with mlx-vlm at about 6.5 GB peak memory. — [HF beshkenadze/moondream3-preview-mlx-4bit](https://huggingface.co/beshkenadze/moondream3-preview-mlx-4bit)
- Molmo2 (NEW; arXiv 2601.10611, v4 2026-04-02): variants Molmo2-4B, Molmo2-8B (Qwen3 LLMs), Molmo2-O-7B (OLMo); image pointing Point-Bench 67.7-68.7; video sampled at 2 fps, max 128 frames (384 in long-context mode). — [Molmo2 paper](https://arxiv.org/html/2601.10611v4)
- mlx-vlm lists Molmo / Molmo 2 / MolmoPoint, Florence2, Moondream3, Qwen3-VL (incl. MoE), LFM2-VL and PaliGemma as supported families; pointing/grounding support is called out for `molmo_point`. A spec entry for MolmoPoint (about 8B) reports about 39 GB peak memory in bf16 and about 6 tokens/s on Apple Silicon. — [mlx-vlm docs mirror](https://www.mintlify.com/yocxy2/mlx-vlm/models/supported-models); [mlx-vlm on PyPI](https://pypi.org/project/mlx-vlm)
- mlx-community hosts LFM2-VL quantizations (450M to 1.6B). — [mlx-community LFM2-VL collection](https://huggingface.co/collections/mlx-community/lfm2-vl)
- An MLX inference paper (NEW, January 2026) notes redundant vision-encoder computation adds 1.5-2 s of latency per request for VLMs such as Qwen3-VL unless cached. — [arXiv 2601.19139](https://arxiv.org/html/2601.19139)
- NVIDIA LocateAnything-3B (NEW; arXiv 2605.27365; release reported as late May 2026 or 2026-06-29): grounding model returning boxes or points with Parallel Box Decoding; a third-party article cites 12.7 boxes/s on one H100 vs 1.1 for Qwen3-VL; license restricted to non-commercial research (NVIDIA License; Qwen2.5-3B base under Qwen Research License). — [Voxel51 model zoo](https://docs.voxel51.com/model_zoo/models/nvidia_LocateAnything_3B.html); [pasqualepillitteri.it](https://pasqualepillitteri.it/en/news/7293/nvidia-locateanything-3b-object-detection-faster); [interfaze mirror of card](https://interfaze.ai/models/nvidialocateanything-3b.md)
- DINO-X Pro: 59.8 AP LVIS-minival, 52.4 AP LVIS-val, 56.0 AP COCO zero-shot; Grounding DINO 1.5 Pro: 55.7 / 47.6 / 54.3 (IDEA Research self-reported). Distribution is through the DINO-X API repository. — [DINO-X paper](https://arxiv.org/html/2411.14347v2); [dino-x-api](https://github.com/idea-research/dino-x-api)
- YOLOE: on LVIS, YOLOE-v8-S beats YOLO-Worldv2-S by 3.5 AP with 1.4x inference speedup; the paper measures CoreML latency on iPhone 12 (1.3x speedup vs YOLO-Worldv2-S). Ultralytics documents newer YOLOE-26 models. — [YOLOE paper](https://arxiv.org/html/2503.07465v2); [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- For the d1-3B decision model as a local verifier: 62 ms per 384 px image on M5 Pro mps. — [HF model card](https://huggingface.co/LiquidAI/d1-3B)

### Inferences
- For players (large objects), any of SAM 3.1, YOLOE or Moondream 3 should work zero-shot; the ball and paddles (thin, fast, often blurred) are the hard classes, and court landmarks are better solved geometrically (line detection plus homography) than by a grounding model.
- Autoregressive VLM grounders (Qwen3-VL, Molmo2, Moondream) on MLX will likely cost on the order of a second per frame or more given vision-encoder overhead and token generation; they suit sparse labeling and tile verification. Feed-forward open-vocabulary detectors (YOLOE, CoreML-exportable) are the only zero-shot class plausibly fast enough for every frame, but their small-object recall is unverified here.
- License screen for a project that might ever be commercial: Kev Apache-2.0; SAM License described as permissive for commercial use; Moondream BSL (restricts hosting); LocateAnything-3B, StartLux non-commercial.

### Gaps
- No measured Apple Silicon grounding latency found for Florence-2, OWLv2, Grounding DINO (open 1.0), YOLO-World/YOLOE (Mac, as opposed to iPhone 12), PaliGemma 2, Qwen3-VL or Molmo2. These need local benchmarking.
- No sources retrieved for Florence-2, OWLv2, PaliGemma 2, Perception Encoder / PLM, or LFM2-VL grounding accuracy in this pass; one older blog claimed Florence-2 has no MLX port, contradicted by the current mlx-vlm listing ([DEV post](https://dev.to/nivdvir/building-a-real-time-screen-reader-on-macos-that-actually-works-471) vs [mlx-vlm docs mirror](https://www.mintlify.com/yocxy2/mlx-vlm/models/supported-models)).
- Small-object AP (APs) breakdowns for YOLOE / Grounding DINO / OWLv2 were not found. YOLOE/Ultralytics license terms were not confirmed in this pass.
- Molmo2 weight licenses were not stated on the page read. LocateAnything-3B has no known MLX port (none found).

## 4. SAM 3 / SAM 3.1 and SAM 2 for concept-prompted detection and tracking

### Takeaway
SAM 3 (848M params, text/exemplar/point/box prompts, image and video) officially requires CUDA, but there is an mlx-vlm port of SAM 3 and SAM 3.1 with measured tracking speeds of about 190-230 ms/frame for 3-5 objects on an M3 Max. The one detailed practitioner report on a small sports ball found SAM 3 excellent for players and goals but returning zero ball detections on entire clips (0/117 and 0/108 frames) at 1080p/30 fps, so SAM 3 alone should not be expected to track a pickleball.

### Cited Findings
- SAM 3 has about 848M parameters with a detector and tracker sharing a vision encoder; it segments all instances of an open-vocabulary concept from a short phrase, accepts point/box/mask prompts and exemplar boxes, and supports text-prompted video tracking with point refinement. — [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Official requirements: Python 3.12+, PyTorch 2.7+, a CUDA-compatible GPU with CUDA 12.6+; MPS/CPU/Apple are not mentioned. License: SAM License; checkpoints are gated on Hugging Face. — [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Benchmarks (README/paper): SA-Co/Gold cgF1 54.1 (instance segmentation; human 72.8), LVIS zero-shot mask AP 48.5 (alphaXiv summary says 48.8); video cgF1/pHOTA: SA-V 30.3/58.0, YT-Temporal-1B 50.8/69.9, SmartGlasses 36.4/63.6; 30 ms per image with 100+ objects on an H200. — [facebookresearch/sam3](https://github.com/facebookresearch/sam3); [alphaXiv overview of arXiv 2511.16719](https://alphaxiv.org/overview/2511.16719v1)
- SAM 3.1 (NEW, 2026-03-27): shared-memory joint multi-object tracking ("Object Multiplex"), described as significantly faster without sacrificing accuracy. — [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- MLX port: `mlx-community/sam3.1-bf16` (about 873M params, 3.49 GB, converted with mlx-vlm 0.4.3) supports text-prompted detection, instance segmentation, box-guided detection, video tracking and a realtime mode; CLI `python -m mlx_vlm.models.sam3_1.generate` with tasks detect/segment/track/realtime, default resolution 1008, `--detect-every 15`, `--backbone-every 30`. — [HF mlx-community/sam3.1-bf16](https://huggingface.co/mlx-community/sam3.1-bf16)
- Measured on M3 Max (bf16): tracker ms/frame SAM 3 vs SAM 3.1: 3 objects 547 vs 227; 4 objects 608 vs 203; 5 objects 766 vs 190. Realtime mode at 224 px: about 40 ms/frame (about 25 FPS) vs about 212 ms unoptimized. — [HF mlx-community/sam3.1-bf16](https://huggingface.co/mlx-community/sam3.1-bf16)
- An open PR on the official repo adds MPS support for image inference; video/tracking remains CUDA-only there, and a HF discussion notes a hard dependency on the CUDA-only triton library. — [sam3 PR mirror](https://gh.r-mj.com/facebookresearch/sam3/pull/400); [HF discussion](https://huggingface.co/facebook/sam3/discussions/11)
- Independent small-ball evidence (NEW, Substack, 2026-06-15, football shot analyzer, YouTube re-encode at 1080p/30 fps): the author calls SAM3 "the best single model in this pipeline" but the ball is "brutally hard to capture in every frame", often 3-4 px wide and smeared; SAM3 found the ball in 0 of 117 frames on one clip (LocateAnything-3B recovered 42 of 117) and 0 of 108 on a tilted-camera clip. — [dcrey7 Substack](https://dcrey7.substack.com/p/world-fut-coach)
- In the Molmo2 paper's video tracking benchmark (Molmo2-Track, J&F), SAM 3 scores 36.3 vs Molmo2-4B 56.7 and Gemini 2.5 Pro 47.9 (Ai2's numbers; language-referred tracking). — [Molmo2 paper](https://arxiv.org/html/2601.10611v4)
- SAM 2 for balls: a Sieve write-up notes SAM 2 needs a manually prepared starting box for the ball and can produce inaccurate boxes, and that YOLOv8 struggles with the ball's small size and motion. — [Sieve blog](https://alpha.sievedata.com/blog/ball-tracking-with-sam2)

### Inferences
- At the measured M3 Max rate (about 200 ms/frame for 4-5 tracked objects at 1008 px), tracking 4 players through a 10-minute, 30 fps clip (18,000 frames) is roughly an hour on comparable hardware; an M5 Pro should be similar or better. This is feasible offline for players (and possibly paddles) but the numbers are from the model card, not independently replicated.
- SAM 3's default 1008 px input downsamples a 1080p-wide frame by about 1.9x, shrinking a 5-15 px ball to roughly 3-8 px; this is consistent with the observed zero-detection failures. Running SAM 3 on tiles or on a zoomed court crop is the obvious mitigation but is untested in the sources.
- SAM 2/2.1-style memory propagation is also a poor fit for the ball: propagation assumes appearance continuity, while the ball changes from a disc to a streak and disappears at contact and behind players.
- "pickleball", "paddle", "player" as concept prompts: "player/person" will work; "paddle" is plausible but unverified (small, thin, heavily occluded by hands); "pickleball" at broadcast scale is expected to fail often based on the soccer evidence.

### Gaps
- No independent test of SAM 3 or 3.1 on a pickleball, tennis ball, shuttle or table-tennis ball was found; the only evidence is one soccer hobby project.
- No memory figures for SAM 3.1 on Apple Silicon; no M5-class measurements; SAM 3.1 accuracy numbers vs SAM 3 not published in the README.
- SAM 2.1 Apple Silicon speed numbers were not retrieved (a Korean blog on running SAM 2.1 on Mac exists but was not read: [ice-ice-bear blog](https://ice-ice-bear.github.io/ko/posts/2026-04-08-sam2-mac/)).
- Whether SAM 3 exemplar prompts (boxing one ball instance) improve tiny-ball recall is untested in the sources.

## 5. Practical recipes: auto-label and distill, tiling, candidate verification, temporal filters

### Takeaway
The only documented 2026 sports pipeline of this kind used SAM 3 plus a grounding VLM fallback to auto-label about 1.8k frames and distill an RF-DETR student that ran about 50x faster, and its ball class still reached only about 0.08 AP@50:95 (vs 0.89 goals, 0.66 players); zero-shot teachers transfer their ball blindness to the student. I found no paper comparing a zero-shot VLM/SAM labeling pipeline against a trained TrackNet-style model, and nothing documenting motion-candidate plus VLM-verification for sports balls.

### Cited Findings
- KICKY AI / FUT-HEROS (NEW, 2026-06-15): pipeline is (1) SAM3 with text prompts "ball", "person", "goal post"; (2) where SAM3 finds no ball, run LocateAnything-3B; (3) merge as auto-labels with no hand annotation; (4) train RF-DETR-Seg-Small at 384 px. The student is about 50x faster than the SAM3+LocateAnything teacher and recovers the ball on some clips SAM3 missed. — [dcrey7 Substack](https://dcrey7.substack.com/p/world-fut-coach); [HF model card](https://huggingface.co/build-small-hackathon/kicky-ai-rfdetr-seg)
- Same project: ball AP@50:95 on validation about 0.08 vs about 0.89 for goals and 0.66 for players; the author attributes this to insufficient resolution and motion blur at 30 fps, calls resolution "the #1 lever", and suggests a larger backbone at higher resolution plus manual labeling. — [dcrey7 Substack](https://dcrey7.substack.com/p/world-fut-coach)
- SAM 3 pseudo-label distillation in another domain (NEW, May 2026, arXiv 2605.25860): SAM3 labels distilled into lightweight YOLOv8 for pig monitoring gave 79.4% mAP without human labels and about 200x lower latency than the teacher. — [paper summary](https://tldr.takara.ai/p/2605.25860)
- Tooling: an `autodistill-sam3` package exists for SAM 3-based auto-labeling. — [PyPI autodistill-sam3](https://pypi.org/project/autodistill-sam3/)
- In the soccer comparison, a YOLOv11 fine-tuned on 11,848 images reached about 0.995 mAP50 on its validation set yet still missed small distant balls that Gemini 3 found. — [slides PDF](https://hosted-files.sched.co/pytorchconferenceeu2026/33/M_Szymkowski_PyTorch_Conference.pdf)
- Trained-model baselines for reference: TrackNet was designed for high-speed tiny balls in tennis and badminton broadcast; a 2023 "widely applicable strong baseline" paper covers sports ball detection and tracking across sports. — [TrackNet](https://arxiv.org/pdf/1907.03698); [WASB](https://arxiv.org/pdf/2311.05237)
- Physics-guided fusion for 3D tracking of fast small objects (NEW, arXiv 2510.20126, October 2025) is a related direction for the tracker stage. — [arXiv 2510.20126](https://arxiv.org/html/2510.20126v1)
- A pickleball-capable open racquet-sports analyzer uses classical ball tracking with a Kalman filter rather than foundation models, and notes pickleball lacks labeled shot data. — [HF Bot-Derpy/racquet-sports-analyzer](https://huggingface.co/Bot-Derpy/racquet-sports-analyzer)

### Inferences
- Recommended division of labor given the evidence (my synthesis, not from a single source):
  1. Players, and probably paddles: SAM 3.1 on MLX with concept prompts, or YOLOE, then a standard tracker. No custom training is likely needed for players.
  2. Court landmarks: classical line detection and homography; grounding models add little here.
  3. Ball: do not rely on any zero-shot whole-frame model. Generate candidates with frame differencing/background subtraction inside the court region (broadcast camera is mostly static), then verify tight upscaled crops with a decision model (d1-3B at about 60-150 ms/crop locally) or a small VLM, then enforce trajectory consistency (ballistic segments, speed limits, Kalman/RANSAC fits) to reject false positives and bridge gaps.
  4. Use that verified, trajectory-filtered output (plus a hosted frontier VLM or SAM 3 on tiles for spot checks, and a modest amount of human correction) as labels to train a TrackNet/WASB-style multi-frame heatmap model at full resolution, which becomes the production ball detector.
- Why multi-frame heatmap students instead of RF-DETR/YOLO at 384 px: the KICKY result shows a single-frame low-resolution student inherits near-zero ball AP. TrackNet-style models take stacked consecutive frames at high resolution and exploit motion, which is precisely the signal single-image VLMs lack.
- Decision models are well matched to the verify step: binary question, calibrated-probability output, about 60-150 ms on this Mac. Their probability threshold should be set on a held-out random sample of human-labeled pickleball crops, since calibration on blurred 10 px balls is unproven.
- Because VLMs almost never reject absent objects (GroundingME rejection near 0%), a "box the ball" auto-labeler must always be paired with a presence question and a temporal filter; otherwise label noise will be dominated by hallucinated boxes on occluded frames.
- Budget check for a hosted labeler: at about 2 s/frame (Gemini 3 in the soccer test), labeling 2,000 sampled frames is about an hour of wall-clock time serially, which is acceptable; labeling every frame of a match is not.

### Gaps
- No 2025-2026 paper or blog found that (a) uses SAHI-style tiling with a VLM for sports balls, (b) uses motion-difference candidates with VLM/decision-model verification, or (c) reports accuracy of a zero-shot-labeled student versus a human-labeled TrackNet. These recipes are plausible engineering, not validated results.
- No pickleball-specific ball detection benchmark or dataset with reported numbers was found in this pass (the adjacent pickleball data is CourtSI-Ext, a spatial QA set).
- The KICKY numbers come from one hobby project on about 1.8k frames; they indicate direction, not a general accuracy level.

## 6. Video-native multimodal models for event understanding (rally start/end, shot type, who hit it, in/out)

### Takeaway
Measured results say video VLMs are not yet reliable referees or scorers for racket sports: the best commercial model on the TennisTV benchmark scored 45.1% vs 89.0% for humans, the best VLM on a racket-sport spatial benchmark (which includes a pickleball out-of-domain split) scored 64.6% vs 73.6% human, and models miscount rally hits. They are usable for coarse tasks (segmenting rallies, describing play) and as a second opinion, but in/out, hit attribution and shot timing should come from the tracked ball/player geometry.

### Cited Findings
- TennisTV (arXiv 2509.15602; v1 2025-09-19, v5 2026-04-16; NEW): 2,527 human-verified questions over 8 tasks from stroke level to rally level, 17 MLLMs evaluated; findings: frame-sampling density should be tailored per task and improved temporal grounding is essential. — [TennisTV abstract](https://arxiv.org/abs/2509.15602)
- TennisTV results as summarized: best commercial model (Gemini 2.5 Pro) 45.1% overall vs 89.0% human; in a rally-counting example with 4 true hits, GPT-4o answered 6 and Gemini 2.5 Pro answered 7. — [alphaXiv TennisTV benchmark page](https://alphaxiv.org/benchmarks/new-york-university/tennistv) (figures via search summary; not read in full)
- CourtSI-Bench (NEW, arXiv 2603.09896, 2026-03-10): 3,686 human-verified QA pairs on spatial counting, distance, localization and relational reasoning in badminton, tennis and table tennis broadcast views, with pickleball as the out-of-domain CourtSI-Ext set; 25 models evaluated; best VLM Gemini-Pro 64.6% overall (after post-processing), best open model Qwen2-VL-72B 47.2%, human 73.6%. — [alphaXiv overview of CourtSI](https://www.alphaxiv.org/overview/2603.09896)
- CourtSI failure modes: confusing objects that are close in 2D but far apart in 3D, monocular depth misinterpretation, fine-grained relations. Fine-tuning on CourtSI improved accuracy by 23.5 points in-domain and by 13.2 points on pickleball. — [alphaXiv overview of CourtSI](https://www.alphaxiv.org/overview/2603.09896)
- SPORTU (ICLR 2025): 1,701 slow-motion clips across 7 sports, 12,048 QA pairs; best overall Qwen2-VL-72B at 70.94% but only 44.12% on hard tasks; Claude-3.5-Sonnet best on hard tasks at 52.57%. — [SPORTU](https://arxiv.org/html/2410.08474v2)
- Molmo2 (NEW): video pointing F1 on Molmo2-VP: Molmo2-4B 39.9, Molmo2-8B 38.4, Gemini 3 Pro 20.0, Gemini 2.5 Pro 13.0, GPT-5 4.1, Claude Sonnet 4.5 3.5, Qwen3-VL-8B 1.5. Video tracking J&F: Molmo2-4B 56.7, Gemini 2.5 Pro 47.9, Gemini 3 Pro 44.6, SAM 3 36.3, Qwen3-VL-8B 28.7, GPT-5 23.5 (Ai2's numbers; no independent replication found). — [Molmo2 paper](https://arxiv.org/html/2601.10611v4)
- Molmo2 samples video at 2 fps with at most 128 frames (384 in long-context mode). — [Molmo2 paper](https://arxiv.org/html/2601.10611v4)
- Gemini 3 Pro is described by Google as tuned for fast-paced action when sampled above 1 fps, with a 10 fps golf-swing example; no sports accuracy figures given. — [Google blog](https://blog.google/technology/developers/gemini-3-pro-vision/)
- VideoNet domain-specific action recognition (NEW, 2026): Gemini 3.1 Pro reaches 70% while Qwen3-VL-8B drops to 45% (not tennis-specific; from a news summary). — [themodelwire](https://themodelwire.com/article/videonet-a-large-scale-dataset-for-domain-specific-action-recognition-01KQV2BJJCZDCJ3VQWFAJDPEDT)

### Inferences
- Default sampling rates (1-2 fps for Gemini and Molmo2) cannot see paddle-ball contact, which lasts a few milliseconds and falls between samples; even 10 fps misses most contacts in a fast hands battle. Event timing must come from the ball trajectory (direction/velocity discontinuities) and audio, with a VLM at most classifying a short high-fps crop around a detected contact.
- A workable hybrid: detect hits and bounces from trajectory kinks, attribute the hitter by nearest paddle/player at the contact frame, compute in/out from the homography, then ask a decision model or VLM narrow typed questions about a cropped clip or keyframe ("forehand or backhand?", "volley or groundstroke?") where calibrated probabilities are useful.
- Rally start/end is the event type most plausibly within current VLM ability (coarse, seconds-scale, strong visual cues such as serve stance and players relaxing), but no benchmark number for it was found; it is also easy to derive from ball-in-play tracking.
- The CourtSI pickleball split shows that racket-sport fine-tuning transfers only partially to pickleball (+13.2 vs +23.5 in-domain), so expect pickleball to be out of distribution for all current models.

### Gaps
- No benchmark found for VLM accuracy on pickleball events (shot type, in/out, who hit) or on line calling in any racket sport.
- No Gemini 3.x, GPT-6 or Claude 5.x results on TennisTV or SPORTU were found; the TennisTV figures above are for Gemini 2.5 Pro and came via a search summary rather than the full paper.
- Perception Language Model (PLM) and Qwen3-VL video results on sports benchmarks were not retrieved. "SportsQA" as named in the brief was not found in this pass.
- No measured data on running video-native VLMs (Molmo2, Qwen3-VL video) on Apple Silicon at useful frame rates.
