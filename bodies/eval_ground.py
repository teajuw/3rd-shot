"""Numbers for one grounded run: steadiness, agreement with the 2D keypoints, position against the court tracks, foot behaviour.

Usage: bodies/.venv/bin/python bodies/eval_ground.py <raw_run_prefix> <grounded_prefix> <tracks.npz> <calibration.json> [gvhmr_run_prefix]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

M2FT = 3.28084
# MHR70 slot -> COCO-17 index, for the joints both have
PAIRS = {5: 5, 6: 6, 7: 7, 8: 8, 62: 9, 41: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16}
WR, AN, HIP = [41, 62], [13, 14], [9, 10]


def steadiness(rel, fps):
    """rel: (n, P, J, 3) hip-relative positions in metres -> accel rms per player, p99 speed per player."""
    acc = np.diff(rel, 2, axis=0) * fps * fps
    vel = np.linalg.norm(np.diff(rel, 1, axis=0) * fps, axis=-1)
    return np.sqrt((acc ** 2).sum(-1).mean((0, 2))), np.percentile(vel, 99, axis=(0, 2))


def main():
    raw, g = np.load(sys.argv[1] + ".npz"), np.load(sys.argv[2] + ".npz")
    tr = np.load(sys.argv[3])
    e = json.loads(Path(sys.argv[4]).read_text())["no_distortion"]
    e = eval(e) if isinstance(e, str) else e  # noqa: S307
    f = e["f_px"]; R, _ = cv2.Rodrigues(np.array(e["rvec"], float)); t = np.array(e["tvec"], float)
    frames, fps = g["frames"], float(g["fps"])
    ok = raw["valid"].all(1)
    proj = lambda Pc: np.stack([f * Pc[..., 0] / Pc[..., 2] + 960, f * Pc[..., 1] / Pc[..., 2] + 540], -1)  # noqa: E731
    res = {"fc": float(g["fc"]), "fc_root": float(g["fc_root"]), "size_factor": np.round(g["size"], 3).tolist()}

    rel = lambda K: K[:, :, WR + AN] - K[:, :, HIP].mean(2, keepdims=True)  # noqa: E731
    a0, v0 = steadiness(rel(raw["pred_keypoints_3d"])[ok], fps)
    a1, v1 = steadiness(rel(g["kp_cam_rel"])[ok], fps)
    res["accel_rms_m_s2"] = {"raw": a0.round(1).tolist(), "smoothed": a1.round(1).tolist()}
    res["wrist_ankle_speed_p99_m_s"] = {"raw": v0.round(2).tolist(), "smoothed": v1.round(2).tolist()}
    if len(sys.argv) > 5:
        gv = np.load(sys.argv[5] + ".npz")
        a2, v2 = steadiness(rel(gv["pred_keypoints_3d"])[ok], fps)
        res["accel_rms_m_s2"]["gvhmr"] = a2.round(1).tolist(); res["wrist_ankle_speed_p99_m_s"]["gvhmr"] = v2.round(2).tolist()

    # agreement with the YOLO 2D keypoints (independent of SAM 3D Body), as % of box height
    k2 = tr["kpts"][frames]                                           # n, P, 17, 3
    box_h = (raw["boxes"][..., 3] - raw["boxes"][..., 1])[..., None]
    p_raw = proj(raw["pred_keypoints_3d"] + raw["pred_cam_t"][:, :, None])
    p_g = proj((g["kp_world"] @ R.T + t) / M2FT)
    out = {}
    for name, pp in (("raw", p_raw), ("smoothed", p_g)):
        err = np.stack([np.linalg.norm(pp[:, :, m] - k2[:, :, c, :2], axis=-1) for m, c in PAIRS.items()], -1)
        conf = np.stack([k2[:, :, c, 2] for c in PAIRS.values()], -1) > 0.5
        e_pct = 100 * err / box_h
        wr = [list(PAIRS).index(m) for m in WR]
        out[name] = {"all_joints_median_pct": [round(float(np.nanmedian(e_pct[ok][:, p][conf[ok][:, p]])), 2) for p in range(4)],
                     "wrists_median_pct": [round(float(np.nanmedian(e_pct[ok][:, p][:, wr][conf[ok][:, p][:, wr]])), 2) for p in range(4)],
                     "wrists_p90_pct": [round(float(np.nanpercentile(e_pct[ok][:, p][:, wr][conf[ok][:, p][:, wr]], 90)), 2) for p in range(4)]}
    res["gap_to_2d_keypoints"] = out

    # position against the court-plane foot tracks
    foot = tr["smooth"][frames]
    ank = g["kp_world"][:, :, AN].mean(2)
    dxy = ank[..., :2] - foot
    res["vs_court_tracks_ft"] = {"across_median": np.median(np.abs(dxy[..., 0]), 0).round(2).tolist(), "along_median": np.median(np.abs(dxy[..., 1]), 0).round(2).tolist(),
                                 "along_p90": np.percentile(np.abs(dxy[..., 1]), 90, axis=0).round(2).tolist(), "along_bias": np.median(dxy[..., 1], 0).round(2).tolist()}
    # feet: height of the lower ankle joint, and how far a planted foot slides
    az = g["kp_world"][:, :, AN, 2]
    low = az.min(-1)
    res["lower_ankle_height_ft"] = {"median": np.median(low, 0).round(2).tolist(), "p5": np.percentile(low, 5, axis=0).round(2).tolist(), "p95": np.percentile(low, 95, axis=0).round(2).tolist()}
    sp = np.linalg.norm(np.diff(g["kp_world"][:, :, AN, :2], axis=0), axis=-1) * fps          # ft/s horizontal
    planted = (az[1:] < np.median(low, 0)[None, :, None] + 0.1) & (np.abs(np.diff(az, axis=0)) * fps < 0.5)
    res["planted_foot_slide_ft_s"] = [round(float(np.median(sp[:, p][planted[:, p]])), 2) for p in range(4)]
    res["planted_share_pct"] = [round(100 * float(planted[:, p].any(-1).mean()), 1) for p in range(4)]
    root = g["root_world"]
    res["root_speed_p99_ft_s"] = np.percentile(np.linalg.norm(np.diff(root[..., :2], axis=0), axis=-1) * fps, 99, axis=0).round(1).tolist()
    lr = g["lam_raw"] - g["lam"]
    res["distance_correction_rms_ft"] = (np.sqrt((lr ** 2).mean(0)) * M2FT).round(2).tolist()
    print(json.dumps(res, indent=1))
    Path(sys.argv[2] + "_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
