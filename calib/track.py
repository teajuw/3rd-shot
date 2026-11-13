"""Match-wide camera check: how well is focal length pinned down, and how much does the
main camera move between rallies?

1. Focal profile on the reference frame: fix f on a grid, refit pose to the court lines,
   and record line fit and held-out net-tape error. Shows how much the planar court alone
   constrains f, and how much the net adds.
2. For frames sampled across the match: coarse pan/tilt/zoom search about the reference
   camera centre (chamfer on a line mask), then line refinement, twice: with the camera
   centre free, and with it fixed to the reference. Frames that do not fit are other
   cameras, replays or graphics.

Usage: python track.py <video> <reference_calibration.json> <out_dir> [step_seconds]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate import (LINES, IN, NET_Y, W, camera_centre, find_ridges, line_residuals,
                       measure_net_tape, project, whiteness)

SCALE = np.r_[1000, 0.1, 1, 1, 1, 10, 10, 10]


def refine(img, p0, size, free, iters=5, centre=None):
    """Refine params against court lines. `free` = indices of params to optimise.
    If `centre` is given, tvec is tied to the rotation so the camera centre stays put."""
    white = whiteness(img)
    p = p0.copy()

    def expand(x, base):
        q = base.copy()
        q[free] = x
        if centre is not None:
            R, _ = cv2.Rodrigues(q[2:5])
            q[5:8] = -R @ centre
        return q

    obs = []
    for it in range(iters):
        obs = find_ridges(white, p, size, LINES, search=9 if it < 2 else 5)
        if len(obs) < 60:
            return p, obs, None
        base = p.copy()
        sol = least_squares(lambda x: line_residuals(expand(x, base), obs, LINES, size), base[free],
                            loss="soft_l1", f_scale=1.0, x_scale=SCALE[free])
        p = expand(sol.x, base)
    return p, obs, line_residuals(p, obs, LINES, size)


def tape_error_in(img, p, size):
    tape = measure_net_tape(img, p, size)
    if len(tape) < 20:
        return None, 0
    a = project(p, [(W / 2, NET_Y, 34 * IN), (W / 2, NET_Y, 35 * IN)], size)
    px_per_in = float(np.linalg.norm(a[1] - a[0]))
    return float(-(tape[:, 2] - tape[:, 1]).mean() / px_per_in), len(tape)


def focal_profile(img, pref, size):
    rows = []
    for f in (3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 10000):
        p0 = pref.copy()
        # scale distance with f so the starting point stays near the image
        C = camera_centre(pref)
        R, _ = cv2.Rodrigues(pref[2:5])
        target = np.array([W / 2, NET_Y, 0.0])
        C2 = target + (C - target) * (f / pref[0])
        p0[0] = f
        p0[5:8] = -R @ C2
        p, obs, r = refine(img, p0, size, free=np.arange(2, 8), iters=6)
        if r is None:
            continue
        err, n = tape_error_in(img, p, size)
        C = camera_centre(p)
        rows.append({"f": f, "line_rms_px": float(np.sqrt(np.mean(r ** 2))), "n": len(r),
                     "net_tape_err_in": err, "cam_dist_ft": float(np.linalg.norm(C - target)), "cam_z_ft": float(C[2])})
    return rows


def line_mask_dt(img):
    m = img.min(axis=2)
    top = cv2.morphologyEx(m, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    mask = (top > 45).astype(np.uint8)
    dt = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    return np.minimum(dt, 20.0)


def rot_xy(tilt, pan):
    cx, sx, cy, sy = np.cos(tilt), np.sin(tilt), np.cos(pan), np.sin(pan)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rx @ Ry


def coarse_search(dt, pref, size, C, samples):
    """Grid over pan/tilt/zoom about a fixed camera centre; score = mean clipped distance
    from projected court-line samples to the nearest line-mask pixel."""
    Rb, _ = cv2.Rodrigues(pref[2:5])
    h, w = dt.shape
    best = (1e9, None)

    def score(tilt, pan, f):
        R = rot_xy(tilt, pan) @ Rb
        cam = (R @ (samples - C).T).T
        uv = cam[:, :2] / cam[:, 2:3] * f + np.array([w / 2, h / 2])
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        if ok.sum() < 0.6 * len(uv):
            return 1e9, R
        ui = uv[ok].round().astype(int)
        return float(dt[ui[:, 1], ui[:, 0]].mean()) + 20.0 * (1 - ok.mean()), R

    grids = [(np.radians(np.arange(-3, 3.01, 0.3)), np.radians(np.arange(-4, 4.01, 0.3)), pref[0] * np.geomspace(0.6, 1.6, 15))]
    centre = (0.0, 0.0, pref[0])
    for level in range(2):
        if level == 0:
            tilts, pans, fs = grids[0]
        else:
            t0, p0, f0 = centre
            tilts = t0 + np.radians(np.arange(-0.3, 0.301, 0.05))
            pans = p0 + np.radians(np.arange(-0.3, 0.301, 0.05))
            fs = f0 * np.geomspace(0.94, 1.06, 9)
        for t in tilts:
            for pa in pans:
                for f in fs:
                    s, R = score(t, pa, f)
                    if s < best[0]:
                        best = (s, (t, pa, f, R))
        centre = best[1][:3]
    s, (t, pa, f, R) = best
    p = pref.copy()
    p[0] = f
    p[2:5] = cv2.Rodrigues(R)[0].ravel()
    p[5:8] = -R @ C
    return p, s


def pan_tilt(p):
    """Optical axis direction in world: pan about Z (deg, 0 = looking along +Y), tilt below horizontal."""
    R, _ = cv2.Rodrigues(p[2:5])
    d = R.T @ np.array([0, 0, 1.0])
    return float(np.degrees(np.arctan2(d[0], d[1]))), float(np.degrees(np.arcsin(-d[2])))


def main():
    video, ref_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = json.loads(ref_path.read_text())
    e = ref["no_distortion"]
    pref = np.r_[e["f_px"], 0.0, e["rvec"], e["tvec"]]
    size = tuple(ref["image_size"])
    Cref = camera_centre(pref)

    ref_img = cv2.imread(str(ref_path.resolve().parent.parent.parent / "data/frames/calib" / ref["frame"]))
    print("== focal profile on reference frame (pose refit at each fixed f) ==")
    prof = focal_profile(ref_img, pref, size)
    for r in prof:
        print(f"  f={r['f']:>6} line_rms={r['line_rms_px']:.3f}px n={r['n']} net_tape_err={r['net_tape_err_in']:+.2f}in "
              f"dist={r['cam_dist_ft']:.0f}ft z={r['cam_z_ft']:.1f}ft")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    if not any(frames_dir.glob("*.jpg")):
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps=1/{step}", "-q:v", "2",
                        str(frames_dir / "%05d.jpg")], check=True)
    files = sorted(frames_dir.glob("*.jpg"))
    print(f"\n== {len(files)} frames, one every {step:g}s ==")

    samples = []
    for a, b in LINES.values():
        a, b = np.array(a, float), np.array(b, float)
        n = int(np.linalg.norm(b - a) / 0.5)
        samples.append(a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None])
    samples = np.concatenate(samples)

    rows = []
    for i, fp in enumerate(files):
        img = cv2.imread(str(fp))
        t = i * step
        p0, s = coarse_search(line_mask_dt(img), pref, size, Cref, samples)
        row = {"t": t, "file": fp.name, "coarse_score": s, "main": False}
        if s < 4.0:
            pf, obs, r = refine(img, p0, size, free=np.r_[0, 2:8])
            pc, obs_c, rc = refine(img, p0, size, free=np.r_[0, 2:5], centre=Cref)
            if r is not None and rc is not None and len(rc) >= 250 and np.sqrt(np.mean(rc ** 2)) < 1.5:
                pan, tilt = pan_tilt(pc)
                err_c, n_c = tape_error_in(img, pc, size)
                err_f, _ = tape_error_in(img, pf, size)
                Cf = camera_centre(pf)
                row.update(main=True, n=len(rc), f_fixed=float(pc[0]), pan=pan, tilt=tilt,
                           rms_fixed=float(np.sqrt(np.mean(rc ** 2))), rms_free=float(np.sqrt(np.mean(r ** 2))),
                           f_free=float(pf[0]), centre_free=Cf.tolist(),
                           tape_in_fixed=err_c, tape_in_free=err_f, tape_cols=n_c)
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(files)} t={t:.0f}s main={row['main']} score={s:.2f}", flush=True)

    (out_dir / "track.json").write_text(json.dumps({"focal_profile": prof, "frames": rows}, indent=1))
    main_rows = [r for r in rows if r["main"]]
    print(f"\nmain-camera frames: {len(main_rows)}/{len(rows)}")

    def stat(name, vals, unit=""):
        v = np.array([x for x in vals if x is not None], float)
        print(f"  {name:<28} median={np.median(v):.3f}{unit} std={v.std():.3f} min={v.min():.3f} max={v.max():.3f} (n={len(v)})")

    stat("f (centre fixed)", [r["f_fixed"] for r in main_rows], "px")
    stat("pan", [r["pan"] for r in main_rows], "deg")
    stat("tilt", [r["tilt"] for r in main_rows], "deg")
    stat("line rms (centre fixed)", [r["rms_fixed"] for r in main_rows], "px")
    stat("line rms (centre free)", [r["rms_free"] for r in main_rows], "px")
    stat("net tape err, fixed centre", [r["tape_in_fixed"] for r in main_rows], "in")
    stat("net tape err, free centre", [r["tape_in_free"] for r in main_rows], "in")
    stat("f (centre free)", [r["f_free"] for r in main_rows], "px")
    C = np.array([r["centre_free"] for r in main_rows])
    for k, ax in enumerate("xyz"):
        stat(f"free centre {ax}", C[:, k], "ft")


if __name__ == "__main__":
    main()
