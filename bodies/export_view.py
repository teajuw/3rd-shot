"""Pack one grounded run for the browser: a reduced mesh per player, key frames, roots and the ball.

Usage: bodies/.venv/bin/python bodies/export_view.py <grounded_prefix> <ball_path.npz|-> <out_dir> [--verts 1500] [--every 2]
Writes <out_dir>/near.bin and far.bin (int16 vertex positions, feet x 3000, relative to each frame's root)
and <out_dir>/meta.json (faces, roots, ball, frame rate).
"""
import argparse
import json
from pathlib import Path

import fast_simplification as fs
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("ball"); ap.add_argument("out")
ap.add_argument("--verts", type=int, default=1500); ap.add_argument("--every", type=int, default=2)
ap.add_argument("--paddle", default=None, help="paddle.py output prefix; supplies the merged ball and the paddles")
a = ap.parse_args()
g = np.load(a.run + ".npz"); V = np.load(a.run + "_verts.npy", mmap_mode="r")
faces = g["faces"].astype(np.int64)
n, P, NV, _ = V.shape
# reduce the mesh once, on one frame, and keep one original vertex per reduced vertex so every frame can reuse it
rest = np.asarray(V[0, 0], np.float32)
_, _, coll = fs.simplify(rest, faces, target_reduction=1 - a.verts / NV, return_collapses=True)
dp, df, imap = fs.replay_simplification(rest, faces, coll)
rep = np.zeros(len(dp), np.int64)
for k in range(len(dp)):
    members = np.flatnonzero(imap == k)
    rep[k] = members[np.argmin(np.linalg.norm(rest[members] - dp[k], axis=1))]
keys = np.arange(0, n, a.every)
Q = 3000.0
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for name, players in (("near", (0, 1)), ("far", (2, 3))):
    arr = np.stack([np.asarray(V[keys][:, p][:, rep], np.float32) for p in players])        # 2, K, v, 3
    assert np.abs(arr).max() * Q < 32767
    np.round(arr * Q).astype("<i2").tofile(out / f"{name}.bin")
ball = None
if a.ball != "-":
    b = np.load(a.ball)
    shot_f = g["frames"][keys]                                   # shot frame (60 fps) of each key frame
    path, keep = b["path"], b["keep"].astype(float)
    ball = []
    for f in shot_f:
        j = int(np.searchsorted(keep, f))
        if j <= 0 or j >= len(keep) or np.isnan(path[j - 1, 0]) or np.isnan(path[j, 0]):
            ball.append(None)
        else:
            w = (f - keep[j - 1]) / max(keep[j] - keep[j - 1], 1e-9)
            ball.append(np.round(path[j - 1] * (1 - w) + path[j] * w, 3).tolist())
paddle = hits = None
if a.paddle:
    pd = np.load(a.paddle + ".npz", allow_pickle=True)
    ball = [None if np.isnan(x[0]) else np.round(x, 3).tolist() for x in pd["ball"][keys]]
    paddle = np.round(np.concatenate([pd["centre"][keys], pd["axis"][keys], pd["normal"][keys]], -1), 3).transpose(1, 0, 2).tolist()
    hits = [{"t": round(h["t"], 2), "player": h["player"]} for h in json.loads(str(pd["hits"]))]
meta = {"fps": float(g["fps"]) / a.every, "frames": int(len(keys)), "verts": int(len(rep)), "q": Q, "faces": df.astype(int).ravel().tolist(),
        "roots": np.round(g["root_world"][keys], 3).transpose(1, 0, 2).tolist(), "ball": ball, "start_s": float(g["start_frame"]) / 60.0,
        "size": np.round(g["size"], 3).tolist(), "paddle": paddle, "hits": hits}
(out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
print(len(rep), "verts,", len(df), "faces,", len(keys), "key frames;", {f.name: f.stat().st_size for f in out.iterdir()}, "ball frames", sum(x is not None for x in (ball or [])))
