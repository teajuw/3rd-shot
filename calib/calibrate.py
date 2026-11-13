"""Single-frame full camera calibration from a pickleball court.

World frame (feet): X across the court (0 = left sideline outer edge as seen from the
main camera, 20 = right), Y along the court (0 = near baseline outer edge, 22 = net,
44 = far baseline), Z up.

Stage 1: PnP + focal/k1 fit on hand-marked rough keypoints (court intersections).
Stage 2: refine against the painted lines themselves: sample each model line, project,
         search along the image normal for the white ridge, minimise point-to-line distance.
Net tape and post tops are held out of the fit and used to check the height mapping.

Usage: python calibrate.py <frame.png> <points.json> <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

IN = 1.0 / 12.0
W, L = 20.0, 44.0
NET_Y = 22.0
NVZ = 7.0
HALF = 1.0 * IN  # half line width (lines are 2 in wide, court measured to outer edges)

# Painted line centres as 3D segments on Z=0.
LINES = {
    "near_baseline": ((0, HALF, 0), (W, HALF, 0)),
    "far_baseline": ((0, L - HALF, 0), (W, L - HALF, 0)),
    "left_sideline": ((HALF, 0, 0), (HALF, L, 0)),
    "right_sideline": ((W - HALF, 0, 0), (W - HALF, L, 0)),
    "near_nvz": ((0, NET_Y - NVZ + HALF, 0), (W, NET_Y - NVZ + HALF, 0)),
    "far_nvz": ((0, NET_Y + NVZ - HALF, 0), (W, NET_Y + NVZ - HALF, 0)),
    "near_centre": ((W / 2, 0, 0), (W / 2, NET_Y - NVZ, 0)),
    "far_centre": ((W / 2, NET_Y + NVZ, 0), (W / 2, L, 0)),
}

# Keypoints at line-centre intersections, matching the rough hand marks.
KP3D = {
    "near_L": (HALF, HALF, 0), "near_C": (W / 2, HALF, 0), "near_R": (W - HALF, HALF, 0),
    "nnvz_L": (HALF, 15 + HALF, 0), "nnvz_C": (W / 2, 15 + HALF, 0), "nnvz_R": (W - HALF, 15 + HALF, 0),
    "fnvz_L": (HALF, 29 - HALF, 0), "fnvz_C": (W / 2, 29 - HALF, 0), "fnvz_R": (W - HALF, 29 - HALF, 0),
    "far_L": (HALF, L - HALF, 0), "far_C": (W / 2, L - HALF, 0), "far_R": (W - HALF, L - HALF, 0),
}

# Net references (top of tape). Posts sit 1 ft outside each sideline at 36 in; centre is 34 in.
NET3D = {
    "net_post_L": (-1.0, NET_Y, 36 * IN),
    "net_C": (W / 2, NET_Y, 34 * IN),
    "net_post_R": (W + 1.0, NET_Y, 36 * IN),
}


def project(params, pts, size):
    """params = [f, k1, rvec(3), tvec(3)]; principal point at image centre, square pixels."""
    f, k1 = params[0], params[1]
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    dist = np.array([k1, 0, 0, 0, 0.0])
    out, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 1, 3), params[2:5], params[5:8], K, dist)
    return out.reshape(-1, 2)


def camera_centre(params):
    R, _ = cv2.Rodrigues(np.asarray(params[2:5], float))
    return -R.T @ np.asarray(params[5:8], float)


def stage1(kp2d, size, fix_k1=True):
    names = [n for n in KP3D if n in kp2d]
    obj = np.array([KP3D[n] for n in names], float)
    img = np.array([kp2d[n] for n in names], float)
    best = None
    for f0 in (1200, 2000, 3500, 6000):
        K = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1.0]])
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        x0 = np.r_[f0, 0.0, rvec.ravel(), tvec.ravel()]

        def res(x):
            p = x.copy()
            if fix_k1:
                p[1] = 0.0
            return (project(p, obj, size) - img).ravel()

        sol = least_squares(res, x0, x_scale=np.r_[1000, 0.1, 1, 1, 1, 10, 10, 10])
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x.copy()
    if fix_k1:
        p[1] = 0.0
    rms = np.sqrt(np.mean(res(best.x).reshape(-1, 2).__pow__(2).sum(1)))
    return p, rms, names


def whiteness(img_bgr):
    """High for white paint, low for the blue court: min over channels, lightly blurred."""
    m = img_bgr.min(axis=2).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def find_ridges(white, params, size, lines, step_ft=0.25, search=9, min_contrast=45, max_width=9.0):
    """For each sample along each model line, find the white ridge centre along the image normal.

    Returns list of (line_name, s in [0,1], observed_xy). Samples that look occluded
    (no clear ridge, or a white blob much wider than a line) are dropped.
    """
    h, w = white.shape
    obs = []
    offs = np.arange(-search, search + 1, 0.5)
    for name, (a, b) in lines.items():
        a, b = np.array(a, float), np.array(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step_ft))
        ss = (np.arange(n) + 0.5) / n
        P = a[None] + ss[:, None] * (b - a)[None]
        uv = project(params, P, size)
        uv2 = project(params, P + (b - a)[None] * 1e-3, size)
        t = uv2 - uv
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        for s, c, nv in zip(ss, uv, nrm):
            xs = c[0] + offs * nv[0]
            ys = c[1] + offs * nv[1]
            if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
                continue
            prof = cv2.remap(white, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            base = np.percentile(prof, 20)
            peak = prof.max()
            if peak - base < min_contrast:
                continue
            wgt = np.clip(prof - (base + 0.5 * (peak - base)), 0, None)
            if wgt.sum() <= 0:
                continue
            width = (wgt > 0).sum() * 0.5
            if width > max_width:
                continue
            # reject ridges touching the search window edge (probably something else)
            if wgt[0] > 0 or wgt[-1] > 0:
                continue
            d = (wgt * offs).sum() / wgt.sum()
            obs.append((name, float(s), (c[0] + d * nv[0], c[1] + d * nv[1])))
    return obs


def line_residuals(params, obs, lines, size):
    """Signed image-space distance from each observed ridge point to its projected model line."""
    res = []
    by = {}
    for name, s, xy in obs:
        by.setdefault(name, []).append((s, xy))
    for name, items in by.items():
        a, b = np.array(lines[name][0], float), np.array(lines[name][1], float)
        ss = np.array([i[0] for i in items])
        xy = np.array([i[1] for i in items])
        P = a[None] + ss[:, None] * (b - a)[None]
        uv = project(params, P, size)
        uv2 = project(params, P + (b - a)[None] * 1e-3, size)
        t = uv2 - uv
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        res.append(((xy - uv) * nrm).sum(1))
    return np.concatenate(res)


def stage2(img, params, size, lines, free_k1=True, iters=6):
    white = whiteness(img)
    p = params.copy()
    obs = []
    for it in range(iters):
        search = 9 if it < 2 else 5
        obs = find_ridges(white, p, size, lines, search=search)

        def res(x):
            q = x.copy()
            if not free_k1:
                q[1] = 0.0
            return line_residuals(q, obs, lines, size)

        sol = least_squares(res, p, loss="soft_l1", f_scale=1.0, x_scale=np.r_[1000, 0.1, 1, 1, 1, 10, 10, 10])
        p = sol.x.copy()
        if not free_k1:
            p[1] = 0.0
    r = line_residuals(p, obs, lines, size)
    return p, obs, r, sol


def net_tape_model(n=41):
    """Top of tape: straight from each post top (36 in) to the centre (34 in)."""
    xs = np.linspace(-1.0, W + 1.0, n)
    z = np.where(xs <= W / 2, 36 + (34 - 36) * (xs + 1) / (W / 2 + 1), 34 + (36 - 34) * (xs - W / 2) / (W / 2 + 1))
    return np.stack([xs, np.full(n, NET_Y), z * IN], 1)


def measure_net_tape(img, params, size):
    """Find the top edge of the white net tape near where the model predicts it.

    Returns (x_world, predicted_v, observed_v) for columns where a clean edge exists.
    """
    white = whiteness(img)
    pts = net_tape_model(85)
    uv = project(params, pts, size)
    out = []
    for P, (u, v) in zip(pts, uv):
        ui = int(round(u))
        if ui < 2 or ui >= size[0] - 2:
            continue
        v0, v1 = int(v) - 14, int(v) + 15
        col = white[v0:v1, ui - 1:ui + 2].mean(axis=1)
        g = np.diff(col)  # positive = dark above, bright below = top edge of tape
        k = int(np.argmax(g))
        if g[k] < 25:
            continue
        # sub-pixel edge: centroid of the positive gradient around the peak
        lo, hi = max(0, k - 2), min(len(g), k + 3)
        gw = np.clip(g[lo:hi], 0, None)
        edge = v0 + 0.5 + (gw * np.arange(lo, hi)).sum() / gw.sum()
        out.append((P[0], v, edge))
    return np.array(out)


def height_at(params, size, ground_xy, pixel):
    """Height (ft) of the point above ground_xy whose image is closest to `pixel`."""
    zs = np.linspace(0, 9, 1801)
    P = np.stack([np.full_like(zs, ground_xy[0]), np.full_like(zs, ground_xy[1]), zs], 1)
    uv = project(params, P, size)
    return float(zs[np.argmin(np.linalg.norm(uv - np.asarray(pixel, float), axis=1))])


def backproject_ground(params, size, pixel):
    """Intersect the (undistorted) ray through `pixel` with Z=0."""
    f, k1 = params[0], params[1]
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    und = cv2.undistortPoints(np.array([[pixel]], float), K, np.array([k1, 0, 0, 0, 0.0])).reshape(2)
    R, _ = cv2.Rodrigues(np.asarray(params[2:5], float))
    d = R.T @ np.array([und[0], und[1], 1.0])
    C = camera_centre(params)
    s = -C[2] / d[2]
    return (C + s * d)[:2]


def draw_overlay(img, params, size, obs, path):
    vis = img.copy()
    for name, (a, b) in LINES.items():
        a, b = np.array(a, float), np.array(b, float)
        P = a[None] + np.linspace(0, 1, 100)[:, None] * (b - a)[None]
        uv = project(params, P, size).round().astype(np.int32)
        cv2.polylines(vis, [uv.reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
    tape = project(params, net_tape_model(), size).round().astype(np.int32)
    cv2.polylines(vis, [tape.reshape(-1, 1, 2)], False, (0, 255, 0), 1, cv2.LINE_AA)
    for x in (-1.0, W + 1.0):
        post = project(params, [(x, NET_Y, 0), (x, NET_Y, 3.0)], size).round().astype(np.int32)
        cv2.line(vis, tuple(map(int, post[0])), tuple(map(int, post[1])), (0, 255, 0), 1, cv2.LINE_AA)
    # 6 ft poles at court corners and NVZ corners make the height mapping visible
    for gx, gy in [(0, 0), (W, 0), (0, 15), (W, 15), (0, 29), (W, 29), (0, L), (W, L)]:
        pole = project(params, [(gx, gy, 0), (gx, gy, 6.0)], size).round().astype(np.int32)
        cv2.line(vis, tuple(map(int, pole[0])), tuple(map(int, pole[1])), (0, 255, 255), 1, cv2.LINE_AA)
    for _, _, xy in obs:
        cv2.circle(vis, (int(round(xy[0])), int(round(xy[1]))), 1, (255, 0, 255), -1)
    cv2.imwrite(str(path), vis)


def main():
    frame_path, pts_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(frame_path))
    size = (img.shape[1], img.shape[0])
    marks = json.loads(pts_path.read_text())
    kp2d = marks["court"]

    p1, rms1, names = stage1(kp2d, size)
    print(f"stage 1 (rough marks, {len(names)} pts): f={p1[0]:.0f}px rms={rms1:.2f}px")

    report = {"frame": frame_path.name, "image_size": size, "stage1": {"f": p1[0], "rms_px": rms1}}
    fits = {}
    for label, free_k1 in (("no_distortion", False), ("with_k1", True)):
        p, obs, r, sol = stage2(img, p1, size, LINES, free_k1=free_k1)
        C = camera_centre(p)
        tape = measure_net_tape(img, p, size)
        tape_err = tape[:, 2] - tape[:, 1] if len(tape) else np.array([])
        fits[label] = (p, obs)
        # px per inch of height at the net centre, to convert tape error to inches
        a = project(p, [(W / 2, NET_Y, 34 * IN), (W / 2, NET_Y, 35 * IN)], size)
        px_per_in = float(np.linalg.norm(a[1] - a[0]))
        entry = {
            "f_px": float(p[0]), "k1": float(p[1]),
            "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / 2 / p[0]))),
            "camera_centre_ft": C.tolist(),
            "rvec": p[2:5].tolist(), "tvec": p[5:8].tolist(),
            "line_samples": int(len(r)),
            "line_rms_px": float(np.sqrt(np.mean(r ** 2))),
            "line_p95_px": float(np.percentile(np.abs(r), 95)),
            "net_tape_samples": int(len(tape)),
            "net_tape_mean_err_px": float(tape_err.mean()) if len(tape) else None,
            "net_tape_rms_err_px": float(np.sqrt(np.mean(tape_err ** 2))) if len(tape) else None,
            "px_per_inch_height_at_net": px_per_in,
            "net_tape_mean_err_in": float(-tape_err.mean() / px_per_in) if len(tape) else None,
        }
        report[label] = entry
        print(f"\n[{label}] f={p[0]:.1f}px k1={p[1]:+.4f} hfov={entry['hfov_deg']:.1f}deg")
        print(f"  camera centre (ft): x={C[0]:.2f} y={C[1]:.2f} z={C[2]:.2f}")
        print(f"  court lines: {len(r)} samples, rms={entry['line_rms_px']:.2f}px p95={entry['line_p95_px']:.2f}px")
        if len(tape):
            print(f"  net tape (held out): {len(tape)} cols, mean err={tape_err.mean():+.2f}px rms={entry['net_tape_rms_err_px']:.2f}px"
                  f" = observed tape {entry['net_tape_mean_err_in']:+.2f} in vs model ({px_per_in:.2f} px/in)")
            for x0, x1 in ((-1, 6), (6, 14), (14, 21)):
                m = (tape[:, 0] >= x0) & (tape[:, 0] < x1)
                if m.any():
                    print(f"    x in [{x0},{x1}) ft: mean err {tape_err[m].mean():+.2f}px ({-tape_err[m].mean() / px_per_in:+.2f} in)")
        draw_overlay(img, p, size, obs, out_dir / f"{frame_path.stem}_{label}.jpg")

    # Hand-marked height checks (net posts, players), using the k1 fit.
    p = fits["with_k1"][0]
    checks = {}
    for name, xyz in NET3D.items():
        if name in marks.get("net", {}):
            px = marks["net"][name]
            checks[name] = {"marked_px": px, "height_in": height_at(p, size, xyz[:2], px) / IN, "expected_in": xyz[2] / IN}
            print(f"  {name}: marked top -> {checks[name]['height_in']:.1f} in (expected {xyz[2] / IN:.0f})")
    for name, pl in marks.get("players", {}).items():
        g = backproject_ground(p, size, pl["feet"])
        hgt = height_at(p, size, g, pl["head"])
        checks[name] = {"ground_ft": g.tolist(), "height_ft": hgt}
        print(f"  {name}: stands at ({g[0]:.1f}, {g[1]:.1f}) ft, head top at {hgt:.2f} ft ({hgt * 12:.0f} in)")
    report["height_checks"] = checks
    (out_dir / f"{frame_path.stem}_calibration.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
