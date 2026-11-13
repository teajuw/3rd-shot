"""Find the paddle outline near a player's hand with SAM 2, with no text prompt and no training.

Usage: uv run python paddle_seg/find_paddle.py [n_frames=12] [seed=5]
Method per player-frame: crop around the paddle-hand wrist (from the 3D body), try 16 click points on two
rings around the wrist (0.45 and 0.8 ft), each with a "not this" click on the forearm, and keep the masks
that could be a paddle: right area for the player's distance, right length, close to the wrist, and not
covering the forearm. The highest-scoring one wins; none passing means "no paddle visible".
Writes contact sheets to paddle_seg/out/ and prints how many crops got a candidate.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent.parent
g = np.load(ROOT / "bodies/out/rally_267_grounded.npz")
e = json.loads((ROOT / "calib/out/t270_calibration.json").read_text())["no_distortion"]
e = eval(e) if isinstance(e, str) else e
R, _ = cv2.Rodrigues(np.array(e["rvec"], float)); t = np.array(e["tvec"], float); f = e["f_px"]
L_FT, W_FT = 1.31, 0.66


def proj(Wp):
    c = R @ Wp + t
    return np.array([f * c[0] / c[2] + 960, f * c[1] / c[2] + 540]), f / c[2]          # pixel, px per foot there


def find(model, im, wrist_w, elbow_w, S, half):
    wrist, ppf = proj(wrist_w); elbow, _ = proj(elbow_w)
    x0, y0 = int(wrist[0]) - half, int(wrist[1]) - half
    src = im[max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half]
    crop = cv2.resize(src, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    loc = lambda q: (np.asarray(q) - np.array([max(x0, 0), max(y0, 0)])) * S
    w, el = loc(wrist), loc(elbow)
    fore = w + 0.5 * (el - w)
    k = ppf * S                                                    # crop px per foot
    cands = [w + r * k * np.array([np.cos(a), np.sin(a)]) for r in (0.45, 0.8) for a in np.arange(8) * np.pi / 4]
    cands = [c for c in cands if 0 <= c[0] < crop.shape[1] and 0 <= c[1] < crop.shape[0]]
    res = model(crop, verbose=False, points=[[c.tolist(), fore.tolist()] for c in cands], labels=[[1, 0]] * len(cands))[0]
    best = None
    if res.masks is not None:
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None and res.boxes.conf is not None else np.ones(len(res.masks.data))
        for m, sc in zip(res.masks.data.cpu().numpy().astype(np.uint8), conf):
            m = cv2.resize(m, (crop.shape[1], crop.shape[0]))
            n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if n_lab < 2:
                continue
            m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
            area = m.sum() / (k * k)                               # square feet in the image plane
            if not 0.12 * L_FT * W_FT < area < 1.25 * L_FT * W_FT:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cs, key=cv2.contourArea))
            long_ft, short_ft = max(rw, rh) / k, min(rw, rh) / k
            if not 0.55 * L_FT < long_ft < 1.35 * L_FT or short_ft > 1.3 * W_FT:
                continue
            ys, xs = np.nonzero(m)
            near = np.min(np.hypot(xs - w[0], ys - w[1])) / k
            hit = lambda q: 0 <= int(q[1]) < m.shape[0] and 0 <= int(q[0]) < m.shape[1] and m[int(q[1]), int(q[0])]  # noqa: E731
            if near > 0.45 or hit(fore) or hit(el):
                continue
            fill = m.sum() / max(rw * rh, 1)                       # a paddle fills most of its bounding rectangle
            score = float(sc) + 0.5 * fill
            if best is None or score > best["score"]:
                best = dict(mask=m, score=score, rect=((cx, cy), (rw, rh), ang), long_ft=long_ft, short_ft=short_ft, conf=float(sc), fill=float(fill))
    return crop, best, w


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    idx = np.sort(rng.choice(len(g["frames"]), n_frames, replace=False))
    model = SAM(str(ROOT / "models/sam2.1_l.pt"))
    cap = cv2.VideoCapture(str(ROOT / "data/video/IpUxCoBHg_g.mp4"))
    tiles = {"near": [], "far": []}; found = {"near": 0, "far": 0}
    for j in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(g["start_frame"]) + int(g["frames"][j])); ok, im = cap.read()
        for p in range(4):
            grp = "near" if p < 2 else "far"
            S, half = (4, 80) if p < 2 else (8, 40)
            crop, best, w = find(model, im, g["kp_world"][j, p, 41], g["kp_world"][j, p, 8], S, half)
            vis = crop.copy()
            if best:
                found[grp] += 1
                m = best["mask"].astype(bool)
                vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
                cv2.drawContours(vis, [cv2.boxPoints(best["rect"]).astype(np.int32)], -1, (0, 255, 0), 2)
                cv2.putText(vis, f"{best['long_ft'] * 12:.0f}x{best['short_ft'] * 12:.0f} in", (8, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (int(w[0]), int(w[1])), 6, (0, 0, 255), 2)
            cv2.putText(crop, f"t={g['frames'][j] / 60:.1f} p{p}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            tiles[grp].append(np.hstack([crop, vis]))
    for grp, tl in tiles.items():
        for k in range(0, len(tl), 12):
            rows = [np.hstack(tl[i:i + 3]) for i in range(k, min(k + 12, len(tl)), 3) if len(tl[i:i + 3]) == 3]
            cv2.imwrite(str(ROOT / f"paddle_seg/out/find_{grp}_{k // 12}.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
        print(grp, "candidate found in", found[grp], "of", len(tl))
    print("times", (g["frames"][idx] / 60).round(1).tolist())


if __name__ == "__main__":
    main()
