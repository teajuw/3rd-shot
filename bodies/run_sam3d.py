"""Run SAM 3D Body on the four tracked players of one shot and time it.

Usage: bodies/.venv/bin/python bodies/run_sam3d.py <video> <tracks.npz> <calibration.json> <out_prefix>
           [--step 2] [--frames 1800] [--mode body|full] [--device mps|cpu] [--first 0]

Boxes come from the existing YOLO keypoints (players/track_players.py), the focal length
from the court calibration, so SAM 3D Body's own detector and field-of-view estimator are
not used. Writes <out_prefix>.npz (per-frame parameters, joints, camera translation, timing)
and <out_prefix>_verts.npy (float16 vertices, camera frame, metres).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/sam-3d-body"))
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402

KEYS = ["pred_keypoints_3d", "pred_keypoints_2d", "pred_cam_t", "global_rot", "body_pose_params", "hand_pose_params",
        "scale_params", "shape_params", "pred_joint_coords", "pred_global_rots", "mhr_model_params", "focal_length"]


def boxes_from_kpts(k, w, h):
    """One box per player from the 17 pose keypoints, padded to take in head, hands and paddle."""
    out = np.full((len(k), 4), np.nan, np.float32)
    for p, kp in enumerate(k):
        ok = (kp[:, 2] > 0.3) & ~np.isnan(kp[:, 0])
        if ok.sum() < 6:
            continue
        x0, y0, x1, y1 = kp[ok, 0].min(), kp[ok, 1].min(), kp[ok, 0].max(), kp[ok, 1].max()
        ph, pw = y1 - y0, x1 - x0
        out[p] = [max(x0 - 0.25 * pw - 0.1 * ph, 0), max(y0 - 0.18 * ph, 0), min(x1 + 0.25 * pw + 0.1 * ph, w - 1), min(y1 + 0.10 * ph, h - 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("tracks"); ap.add_argument("calib"); ap.add_argument("out")
    ap.add_argument("--step", type=int, default=2); ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--mode", default="body"); ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    tr = np.load(a.tracks)
    kpts, start = tr["kpts"], int(tr["start_frame"])
    cal = json.loads(Path(a.calib).read_text())
    e = cal["no_distortion"]
    if isinstance(e, str):
        e = eval(e)  # noqa: S307  (our own calibration file)
    W, H = cal["image_size"]
    K = torch.tensor([[[e["f_px"], 0, W / 2], [0, e["f_px"], H / 2], [0, 0, 1]]], dtype=torch.float32)

    t0 = time.time()
    model, cfg = load_sam_3d_body(str(ROOT / "models/sam3d_body/model.ckpt"), device=a.device,
                                  mhr_path=str(ROOT / "models/sam3d_body/assets/mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f} s on {a.device}", flush=True)

    idx = list(range(a.first, min(a.first + a.frames, len(kpts)), a.step))
    n, P = len(idx), kpts.shape[1]
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start + a.first)
    res = {k: None for k in KEYS}
    verts = None
    valid = np.zeros((n, P), bool)
    boxes_all = np.full((n, P, 4), np.nan, np.float32)
    secs = np.zeros(n)
    cur = a.first
    for j, i in enumerate(idx):
        while cur <= i:
            ok, frame = cap.read(); cur += 1
        assert ok, f"could not read frame {i}"
        b = boxes_from_kpts(kpts[i], W, H)
        boxes_all[j] = b
        good = np.flatnonzero(~np.isnan(b[:, 0]))
        if not len(good):
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        outs = est.process_one_image(rgb, bboxes=b[good], cam_int=K, inference_type=a.mode)
        if a.device == "mps":
            torch.mps.synchronize()
        secs[j] = time.time() - t1
        for p, o in zip(good, outs):
            if verts is None:
                verts = np.lib.format.open_memmap(a.out + "_verts.npy", mode="w+", dtype=np.float16, shape=(n, P) + o["pred_vertices"].shape)
                for k in KEYS:
                    v = np.asarray(o[k])
                    res[k] = np.full((n, P) + v.shape, np.nan, np.float32)
            verts[j, p] = o["pred_vertices"]
            for k in KEYS:
                res[k][j, p] = o[k]
            valid[j, p] = True
        if j % 25 == 0:
            done = secs[: j + 1]
            print(f"{j + 1}/{n} frames, {done[done > 0][1:].mean() if j else done[0]:.2f} s/frame, {valid[: j + 1].sum()} bodies", flush=True)
    verts.flush()
    np.savez_compressed(a.out + ".npz", frames=np.array(idx), start_frame=start, valid=valid, boxes=boxes_all, secs=secs,
                        faces=est.faces, load_s=load_s, mode=a.mode, device=a.device, **res)
    s = secs[secs > 0][1:]
    per = s.sum() / valid[1:].sum()
    print(f"DONE {n} frames, {valid.sum()} bodies; {s.mean():.3f} s/frame median {np.median(s):.3f}; {per:.3f} s per person-frame")


if __name__ == "__main__":
    main()
