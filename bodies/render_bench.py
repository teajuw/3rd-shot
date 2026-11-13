"""Videos and numbers for the SAM 3D Body benchmark.

Usage: bodies/.venv/bin/python bodies/render_bench.py <video> <run_prefix> <tracks.npz> <calibration.json> <out_prefix>
Writes <out_prefix>_overlay.mp4 (broadcast with the meshes drawn on it), <out_prefix>_players.mp4
(per player: video crop beside the bare mesh) and <out_prefix>_stats.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

M2FT = 3.28084
COL = np.array([[60, 200, 255], [255, 200, 60], [230, 90, 230], [90, 230, 120]], np.float32)   # BGR per player
NAMES = ["near-left", "near-right", "far-left", "far-right"]


def vertex_normals(v, faces):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        for c in range(3):
            n[:, c] += np.bincount(faces[:, k], weights=fn[:, c], minlength=len(v))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def splat(img, zbuf, P, n, f, colour, x0, y0, s, r=1):
    """Draw camera-frame points P (metres) into img with a depth buffer. (x0, y0) crop origin in full-frame px, s scale."""
    H, W = img.shape[:2]
    u = ((f * P[:, 0] / P[:, 2] + 960) - x0) * s
    w = ((f * P[:, 1] / P[:, 2] + 540) - y0) * s
    light = np.array([0.3, -0.5, -0.8]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    front = n[:, 2] < 0.2
    u, w, z, shade = u[front], w[front], P[front, 2], shade[front]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xi, yi = np.round(u + dx).astype(int), np.round(w + dy).astype(int)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            idx, zz, sh = yi[ok] * W + xi[ok], z[ok], shade[ok]
            o = np.argsort(-zz)
            idx, zz, sh = idx[o], zz[o], sh[o]
            near = zz < zbuf.flat[idx]
            idx, zz, sh = idx[near], zz[near], sh[near]
            zbuf.flat[idx] = zz
            img.reshape(-1, 3)[idx] = sh[:, None] * colour[None]


def main():
    video, run, tracks, calib, out = sys.argv[1:6]
    d = np.load(run + ".npz")
    V = np.load(run + "_verts.npy", mmap_mode="r")
    faces = d["faces"].astype(np.int64)
    tr = np.load(tracks)
    e = json.loads(Path(calib).read_text())["no_distortion"]
    e = eval(e) if isinstance(e, str) else e  # noqa: S307
    f = e["f_px"]
    R, _ = cv2.Rodrigues(np.array(e["rvec"], float)); t = np.array(e["tvec"], float)
    frames, start, valid = d["frames"], int(d["start_frame"]), d["valid"]
    n = len(frames)
    step = int(frames[1] - frames[0])
    fps = 60 / step

    def writer(path, w, h):
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow", "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    CW, CH = 300, 400                      # one crop cell
    wo, wp = writer(out + "_overlay.mp4", 1280, 720), writer(out + "_players.mp4", CW * 4, CH * 2)
    # fixed crop size per player (largest box over the run), centred on the box each frame, smoothed
    b = d["boxes"]
    cen = np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)
    for p in range(4):
        for c in range(2):
            x = cen[:, p, c]; ok = ~np.isnan(x)
            x = np.interp(np.arange(n), np.flatnonzero(ok), x[ok])
            k = max(int(fps * 0.6) | 1, 3)
            cen[:, p, c] = np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, "valid")
    hh = np.nanpercentile(b[..., 3] - b[..., 1], 98, axis=0) * 1.15          # crop height per player, px

    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, start + int(frames[0]))
    cur = int(frames[0])
    for j, i in enumerate(frames):
        while cur <= i:
            ok, im = cap.read(); cur += 1
        over = im.copy(); zb = np.full(im.shape[:2], np.inf, np.float32)
        cells = np.zeros((CH * 2, CW * 4, 3), np.uint8)
        for p in range(4):
            if not valid[j, p]:
                continue
            P = V[j, p].astype(np.float32) + d["pred_cam_t"][j, p]
            nrm = vertex_normals(P, faces)
            splat(over, zb, P, nrm, f, COL[p], 0, 0, 1.0, r=1 if P.shape[0] > 15000 else 2)
            s = CH / hh[p]
            x0, y0 = cen[j, p, 0] - CW / s / 2, cen[j, p, 1] - CH / s / 2
            Mx = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
            crop = cv2.warpAffine(im, Mx, (CW, CH), flags=cv2.INTER_LANCZOS4)
            mesh = np.full((CH, CW, 3), 34, np.uint8); zc = np.full((CH, CW), np.inf, np.float32)
            splat(mesh, zc, P, nrm, f, COL[p], x0, y0, s, r=max(2, int(np.ceil(s * 1.2 * (18439 / P.shape[0]) ** 0.5))))
            mesh = cv2.medianBlur(mesh, 3)
            col, row = (p % 2) * 2, p // 2
            cells[row * CH:(row + 1) * CH, col * CW:(col + 1) * CW] = crop
            cells[row * CH:(row + 1) * CH, (col + 1) * CW:(col + 2) * CW] = mesh
            cv2.putText(cells, NAMES[p], (col * CW + 8, row * CH + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        blend = cv2.addWeighted(over, 0.8, im, 0.2, 0)
        wo.stdin.write(cv2.resize(blend, (1280, 720), interpolation=cv2.INTER_AREA).tobytes())
        wp.stdin.write(cells.tobytes())
        if j % 100 == 0:
            print(f"rendered {j}/{n}", flush=True)
    for w in (wo, wp):
        w.stdin.close(); w.wait()

    # ---- numbers ----
    secs = d["secs"]; s = secs[secs > 0][1:]
    per = float(s.sum() / valid[1:].sum())
    # world position of the pelvis-ish root (mean of the two hips, MHR70 keypoints 9 and 10) versus court foot tracks
    kp = d["pred_keypoints_3d"] + d["pred_cam_t"][:, :, None]            # camera frame, metres
    ank = kp[:, :, [13, 14]].mean(2) * M2FT                               # ankles, feet units
    world = (ank - t) @ R                                                 # R^T (P - t)
    foot = tr["smooth"][frames]                                           # court-plane tracks, ft
    stats = {"frames": int(n), "fps_processed": fps, "person_frames": int(valid.sum()), "found_pct": [round(100 * float(valid[:, p].mean()), 1) for p in range(4)],
             "s_per_frame_mean": round(float(s.mean()), 3), "s_per_person_frame": round(per, 3),
             "minutes_per_video_minute_4_players_50fps": round(per * 4 * 50, 1), "minutes_per_video_minute_4_players_25fps": round(per * 4 * 25, 1),
             "players": []}
    for p in range(4):
        ok = valid[:, p]
        dxy = world[ok, p, :2] - foot[ok, p]
        root_rel = d["pred_keypoints_3d"][ok, p] - d["pred_keypoints_3d"][ok, p][:, [9, 10]].mean(1, keepdims=True)
        acc = np.diff(root_rel, 2, axis=0) * fps * fps                    # m/s^2
        stats["players"].append({
            "name": NAMES[p], "box_height_px": round(float(np.nanmedian(b[:, p, 3] - b[:, p, 1]))),
            "across_court_diff_ft_median": round(float(np.median(np.abs(dxy[:, 0]))), 2),
            "depth_diff_ft_median": round(float(np.median(np.abs(dxy[:, 1]))), 2), "depth_diff_ft_p90": round(float(np.percentile(np.abs(dxy[:, 1]), 90)), 2),
            "depth_bias_ft": round(float(np.median(dxy[:, 1])), 2),
            "depth_jitter_ft": round(float(np.std(np.diff(dxy[:, 1]))), 2),
            "ankle_height_ft_median": round(float(np.median(world[ok, p, 2])), 2),
            "joint_accel_rms_m_s2": round(float(np.sqrt((acc ** 2).sum(-1).mean())), 1),
            "body_height_m_median": round(float(np.median(np.ptp(np.asarray(V[ok, p][:, :, 1], np.float32), axis=1))), 2),
            "shape_std": round(float(d["shape_params"][ok, p].std(0).mean()), 3), "scale_std": round(float(d["scale_params"][ok, p].std(0).mean()), 3)})
    np.savez_compressed(out + "_world.npz", world_ankle=world, foot=foot, frames=frames, valid=valid)
    Path(out + "_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
