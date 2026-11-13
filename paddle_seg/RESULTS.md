# Finding the paddle in the image

## SAM 2, prompted from the body (2026-11-13)

SAM 3 (text prompts such as "pickleball paddle") is gated on Hugging Face (`facebook/sam3`, manual approval) and was not available, so this tried SAM 2 (`models/sam2.1_l.pt`, from Ultralytics, ungated) with prompts derived from the 3D body. Rally at 267 s, randomly sampled frames.

- **`try_sam2.py` (8 frames, near players):** one click where the paddle guess put the face, plus a "not this" click on the elbow. When the click landed on the paddle the outline was clean, including on a motion-blurred paddle (about 4 of 8 crops on the first sheet). When it landed on the court, SAM 2 returned the court.
- **`find_paddle.py` (12 frames, all four players):** 16 clicks on two rings around the wrist, then keep masks with a paddle-like area, length and position that do not cover the forearm. A candidate was returned for 12 of 24 near crops and 22 of 24 far crops, but on the two sheets inspected (24 crops) essentially none was the paddle: it picked shorts, banner panels and patches of court, and missed paddles that were plainly visible.

Conclusion: SAM 2 can outline the paddle but cannot tell which outline is the paddle; size-and-shape rules are not enough. Something that knows what a paddle looks like is needed: SAM 3 with a text prompt (needs `sam3.pt` from the gated repo; Ultralytics 8.4 has SAM 3 predictors), or a small detector trained on labelled paddle frames, whose box or point would then prompt SAM 2.

Also seen: far players' paddles are clearly visible in these crops, so image evidence for far players is not hopeless. The model's wrist is sometimes a few inches up the forearm.

Sheets: `out/sam2_sheet_{0,1}.jpg`, `out/find_{near,far}_{0,1}.jpg`.
