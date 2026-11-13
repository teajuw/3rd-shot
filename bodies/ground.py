"""Turn raw per-frame SAM 3D Body output into steady bodies standing on the court.

Usage: bodies/.venv/bin/python bodies/ground.py <run_prefix> <tracks.npz> <calibration.json> <out_prefix> [--fc 6] [--fc_root 3]

For each player:
  1. Shape: one body shape for the whole shot (median of the per-frame estimates).
  2. Motion: joint angles are filtered in a wrap-free representation (median of 3, then a
     zero-phase low-pass at --fc Hz), then the body is rebuilt with the fixed shape.
  3. Size and place: the model fixes where the body is in the image but not how far away it
     is; a body 4% smaller and 4% closer looks the same. The floor settles it: each frame the
     body is slid along its camera ray until its lowest point touches the court, and one size
     factor per player is chosen so that this agrees with the size the model saw. The distance
     along the ray is then low-passed at --fc_root Hz.
Writes <out_prefix>.npz (parameters, size factors, world keypoints and joints, metrics inputs)
and <out_prefix>_verts.npy (float16 world vertices in feet, relative to each frame's root).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import roma
import torch
from scipy.signal import butter, filtfilt, medfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mhr  # noqa: E402

M2FT = 3.28084


def lowpass(x, fc, fps, med=3):
    """Median of `med` then zero-phase Butterworth along axis 0."""
    x = np.asarray(x, np.float64)
    if med > 1:
        x = medfilt(x, [med] + [1] * (x.ndim - 1))
    if fc <= 0 or fc >= fps / 2:
        return x
    b, a = butter(2, fc / (fps / 2))
    return np.ascontiguousarray(filtfilt(b, a, x, axis=0, padlen=min(3 * max(len(a), len(b)), len(x) - 1)))


def fill(x, ok):
    """Linear interpolation over frames where ok is False."""
    x = np.array(x, np.float64)
    if ok.all():
        return x
    i = np.arange(len(x)); flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[~ok, c] = np.interp(i[~ok], i[ok], flat[ok, c])
    return flat.reshape(x.shape)


def smooth_rot_euler(euler, fc, fps, order="ZYX"):
    R = roma.euler_to_rotmat(order, torch.as_tensor(euler, dtype=torch.float64)).numpy()
    six = lowpass(R[:, :, :2].reshape(len(R), 6), fc, fps).reshape(len(R), 3, 2)
    a = six[:, :, 0] / np.linalg.norm(six[:, :, 0], axis=1, keepdims=True)
    b = six[:, :, 1] - (a * six[:, :, 1]).sum(1, keepdims=True) * a
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    Rs = np.stack([a, b, np.cross(a, b)], 2)
    return roma.rotmat_to_euler(order, torch.as_tensor(Rs)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("tracks"); ap.add_argument("calib"); ap.add_argument("out")
    ap.add_argument("--fc", type=float, default=6.0); ap.add_argument("--fc_root", type=float, default=3.0)
    ap.add_argument("--no_verts", action="store_true")
    a = ap.parse_args()
    d = np.load(a.run + ".npz")
    e = json.loads(Path(a.calib).read_text())["no_distortion"]
    e = eval(e) if isinstance(e, str) else e  # noqa: S307
    R, _ = cv2.Rodrigues(np.array(e["rvec"], float)); t = np.array(e["tvec"], float)
    up = R[:, 2]                                        # world z of a camera-frame vector x is up @ x
    frames, valid = d["frames"], d["valid"]
    n, P = valid.shape
    fps = 60 / int(frames[1] - frames[0])
    out = {k: [] for k in ("global_rot", "body_pose", "hand", "shape", "scale", "size", "lam", "lam_raw", "ray", "root_world", "kp_world", "joints_world", "kp_cam_raw_rel", "kp_cam_rel")}
    verts = None
    for p in range(P):
        ok = valid[:, p]
        shape = np.repeat(np.median(d["shape_params"][ok, p], 0)[None], n, 0)
        scale = np.repeat(np.median(d["scale_params"][ok, p], 0)[None], n, 0)
        bp = fill(d["body_pose_params"][:, p], ok)
        cont = mhr.compact_model_params_to_cont_body(torch.as_tensor(bp, dtype=torch.float32)).numpy()
        bp_s = mhr.compact_cont_to_model_params_body(torch.as_tensor(lowpass(cont, a.fc, fps), dtype=torch.float32)).numpy()
        bp_s[:, -3:] = 0
        g_s = smooth_rot_euler(fill(d["global_rot"][:, p], ok), a.fc, fps)
        hand = lowpass(fill(d["hand_pose_params"][:, p], ok), min(a.fc, 3.0), fps)
        V, K, J = mhr.forward(g_s, bp_s, hand, scale, shape)         # root-relative, camera axes, metres
        # where the model saw the body: direction of its camera translation, smoothed like the pose
        c = lowpass(fill(d["pred_cam_t"][:, p], ok), a.fc, fps)
        dist = np.linalg.norm(c, axis=1); ray = c / dist[:, None]
        low = (V @ up).min(1)                                         # lowest point of the body, metres along world up
        A, B = (up @ t) / M2FT, ray @ up
        # size factor s: floor contact gives distance lam(s) = (A - s*low)/B; pick s so lam agrees with s*dist on the median frame
        lo, hi = 0.7, 1.4
        for _ in range(40):
            s = (lo + hi) / 2
            if np.median((A - s * low) / B / (s * dist)) > 1:
                lo = s
            else:
                hi = s
        lam_raw = (A - s * low) / B
        lam = lowpass(lam_raw, a.fc_root, fps, med=5)
        to_world = lambda X: (M2FT * X - t) @ R                       # noqa: E731  camera metres -> court feet
        root_cam = lam[:, None] * ray
        root_world = to_world(root_cam)
        kp_world = (M2FT * (s * K + root_cam[:, None]) - t) @ R
        joints_world = (M2FT * (s * J + root_cam[:, None]) - t) @ R
        if not a.no_verts:
            if verts is None:
                verts = np.lib.format.open_memmap(a.out + "_verts.npy", mode="w+", dtype=np.float16, shape=(n, P, V.shape[1], 3))
            verts[:, p] = (M2FT * s * V) @ R                          # world axes, feet, relative to root_world
        Kraw = fill(d["pred_keypoints_3d"][:, p], ok)
        for k, v in (("global_rot", g_s), ("body_pose", bp_s), ("hand", hand), ("shape", shape[0]), ("scale", scale[0]), ("size", s), ("lam", lam), ("lam_raw", lam_raw),
                     ("ray", ray), ("root_world", root_world), ("kp_world", kp_world), ("joints_world", joints_world), ("kp_cam_raw_rel", Kraw), ("kp_cam_rel", K)):
            out[k].append(v)
        print(f"player {p}: size factor {s:.3f}, shape spread before lock {d['shape_params'][ok, p].std(0).mean():.3f}", flush=True)
    if verts is not None:
        verts.flush()
    np.savez_compressed(a.out + ".npz", frames=frames, start_frame=d["start_frame"], valid=valid, faces=d["faces"], fps=fps, fc=a.fc, fc_root=a.fc_root,
                        **{k: np.stack(v, 1) if np.ndim(v[0]) > 1 or k in ("lam", "lam_raw") else np.stack(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
