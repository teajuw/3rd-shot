"""Quick look: can SAM 2 outline the paddle when prompted from where the body says the hand is?

Usage: uv run python paddle_seg/try_sam2.py   (writes paddle_seg/out/sam2_sheet_*.jpg)
Random frames (seed 3), near players only. Two prompts per crop: A = a point where the current
paddle guess puts the face, with the elbow as a negative point; B = a box around the hand.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent.parent
g = np.load(ROOT / "bodies/out/rally_267_grounded.npz"); pd = np.load(ROOT / "bodies/out/rally_267_paddle.npz", allow_pickle=True)
e = json.loads((ROOT / "calib/out/t270_calibration.json").read_text())["no_distortion"]
e = eval(e) if isinstance(e, str) else e
R, _ = cv2.Rodrigues(np.array(e["rvec"], float)); t = np.array(e["tvec"], float); f = e["f_px"]
proj = lambda W: (lambda c: np.array([f * c[0] / c[2] + 960, f * c[1] / c[2] + 540]))(R @ W + t)
rng = np.random.default_rng(3)
idx = np.sort(rng.choice(len(g["frames"]), 8, replace=False))
model = SAM(str(ROOT / "models/sam2.1_l.pt"))
cap = cv2.VideoCapture(str(ROOT / "data/video/IpUxCoBHg_g.mp4"))
S, HALF = 4, 80                                    # upscale, half crop size in source px
rows = []
for j in idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(g["start_frame"]) + int(g["frames"][j])); ok, im = cap.read()
    for p in (0, 1):
        wrist, elbow = proj(g["kp_world"][j, p, 41]), proj(g["kp_world"][j, p, 8])
        face = proj(pd["centre"][j, p] + 0.2 * pd["axis"][j, p])
        c = wrist.astype(int)
        x0, y0 = c[0] - HALF, c[1] - HALF
        crop = cv2.resize(im[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
        loc = lambda q: ((q - np.array([x0, y0])) * S).tolist()
        tiles = [crop.copy()]
        for name, kw in (("A", dict(points=[[loc(face), loc(elbow)]], labels=[[1, 0]])), ("B", dict(bboxes=[loc(wrist - 55) + loc(wrist + 55)]))):
            r = model(crop, verbose=False, **kw)[0]
            vis = crop.copy()
            if r.masks is not None and len(r.masks.data):
                m = r.masks.data[0].cpu().numpy().astype(bool)
                m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0])).astype(bool)
                vis[m] = (0.45 * vis[m] + 0.55 * np.array([0, 255, 255])).astype(np.uint8)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)
            if name == "A":
                cv2.circle(vis, tuple(int(v) for v in loc(face)), 7, (0, 255, 0), -1); cv2.circle(vis, tuple(int(v) for v in loc(elbow)), 7, (0, 0, 255), -1)
            else:
                b = [int(v) for v in loc(wrist - 55) + loc(wrist + 55)]; cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(vis, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(vis)
        cv2.putText(tiles[0], f"t={g['frames'][j] / 60:.1f}s p{p}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
for k in range(0, len(rows), 4):
    cv2.imwrite(str(ROOT / f"paddle_seg/out/sam2_sheet_{k // 4}.jpg"), cv2.resize(np.vstack(rows[k:k + 4]), None, fx=0.5, fy=0.5))
print("frames", (g["frames"][idx] / 60).round(1).tolist())
