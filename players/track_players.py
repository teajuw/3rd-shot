"""Player dots: detect people with a pose model, put their feet on the calibrated court,
track the four players through one baseline-camera shot, smooth, and render.

Usage: python track_players.py <video> <calibration.json> <start_frame> <n_frames> <out_prefix>
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, filtfilt, medfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calib"))
from calibrate import L, LINES, NET_Y, W, camera_centre, project  # noqa: E402

FPS = 60
ANKLE_Z = 0.3            # ft: assumed ankle-joint height above the floor
CUTOFF_HZ = 2.5          # low-pass for the ground tracks
NAMES = ["near-left", "near-right", "far-left", "far-right"]
COLORS = [(60, 170, 255), (80, 220, 120), (255, 150, 60), (230, 110, 230)]  # BGR


def load_camera(path):
    ref = json.loads(Path(path).read_text())
    e = ref["no_distortion"]
    p = np.r_[e["f_px"], 0.0, e["rvec"], e["tvec"]]
    return p, tuple(ref["image_size"])


def to_ground(p, size, px, z=0.0):
    """Intersect pixel rays with the horizontal plane at height z. px: (N,2) -> (N,2) ft."""
    R, _ = cv2.Rodrigues(p[2:5])
    C = camera_centre(p)
    d = np.stack([(px[:, 0] - size[0] / 2) / p[0], (px[:, 1] - size[1] / 2) / p[0], np.ones(len(px))], 1) @ R
    s = (z - C[2]) / d[:, 2]
    return (C[None] + s[:, None] * d)[:, :2]


def feet_to_ground(p, size, ankles):
    """Ground point under a player from one or two ankle pixels.

    A lifted foot sits higher in the image, so its ray lands too far from the camera
    (several feet too far for the far-side players). Across-court position uses the mean
    of both ankles; depth uses the ankle nearer the camera, plus half the depth gap capped
    at 1.5 ft, so a planted stance is centred and a raised foot cannot drag the point away.
    """
    g = to_ground(p, size, ankles, ANKLE_Z)
    if len(g) == 1:
        return g[0]
    gap = abs(g[0, 1] - g[1, 1])
    return np.array([g[:, 0].mean(), g[:, 1].min() + min(gap, 1.5) / 2])


def frames(video, start_frame, n, size):
    t = max(0.0, (start_frame - 0.3) / FPS)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", video, "-frames:v", str(n),
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=1 << 24)
    nb = size[0] * size[1] * 3
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            return
        yield np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3)


def detect(video, start_frame, n, p, size, model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    dets = []  # per frame: list of dict(ground, conf, kpts, box, used_ankles)
    batch = []

    def flush():
        for r in model.predict(batch, imgsz=1280, device="mps", verbose=False, conf=0.25):
            out = []
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                kxy = r.keypoints.xy.cpu().numpy()
                kcf = r.keypoints.conf.cpu().numpy()
                for b, c, xy, kc in zip(boxes, confs, kxy, kcf):
                    ank = [xy[i] for i in (15, 16) if kc[i] > 0.5]
                    if ank:
                        g = feet_to_ground(p, size, np.array(ank))
                    else:
                        g = to_ground(p, size, np.array([[(b[0] + b[2]) / 2, b[3]]]), 0.0)[0]
                    out.append({"ground": g, "conf": float(c), "kpts": np.concatenate([xy, kc[:, None]], 1),
                                "box": b, "ankles": len(ank)})
            dets.append(out)
        batch.clear()

    for f in frames(video, start_frame, n, size):
        batch.append(f.copy())
        if len(batch) == 8:
            flush()
    if batch:
        flush()
    return dets


def track(dets):
    n = len(dets)
    pos = np.full((n, 4, 2), np.nan)
    src = np.zeros((n, 4), np.int8)  # 0 missing, 1 box-bottom, 2 one ankle, 3 both ankles
    kpts = np.full((n, 4, 17, 3), np.nan, np.float32)

    def on_court(d):
        g = d["ground"]
        return -3 <= g[0] <= W + 3 and -8 <= g[1] <= L + 8 and d["conf"] > 0.5

    init = None
    for i, ds in enumerate(dets):
        near = sorted([d for d in ds if on_court(d) and d["ground"][1] < NET_Y], key=lambda d: d["ground"][0])
        far = sorted([d for d in ds if on_court(d) and d["ground"][1] >= NET_Y], key=lambda d: d["ground"][0])
        if len(near) == 2 and len(far) == 2:
            init = i
            last = np.array([d["ground"] for d in near + far])
            break
    if init is None:
        raise RuntimeError("never saw two players on each side")
    gap = np.zeros(4, int)
    for i in list(range(init, n)):
        ds = dets[i]
        if ds:
            G = np.array([d["ground"] for d in ds])
            cost = np.linalg.norm(last[:, None] - G[None], axis=2)
            gate = np.minimum(4.0 + 0.3 * gap, 12.0)
            # players never cross the net: forbid assignments to the wrong side (2 ft tolerance)
            side = np.array([0, 0, 1, 1])
            wrong = (side[:, None] == 0) & (G[None, :, 1] > NET_Y + 2) | (side[:, None] == 1) & (G[None, :, 1] < NET_Y - 2)
            big = np.where((cost > gate[:, None]) | wrong, 1e6, cost)
            rows, cols = linear_sum_assignment(big)
            for r, c in zip(rows, cols):
                if big[r, c] < 1e5:
                    pos[i, r] = G[c]
                    src[i, r] = 1 + ds[c]["ankles"]
                    kpts[i, r] = ds[c]["kpts"]
                    last[r] = G[c]
                    gap[r] = -1
        gap += 1
    # frames before init: carry the first position backward (rare; shots start in view)
    return pos, src, kpts, init


def smooth(pos):
    out = np.empty_like(pos)
    b, a = butter(2, CUTOFF_HZ / (FPS / 2))
    t = np.arange(len(pos))
    for k in range(4):
        # outlier rejection: drop samples more than 1.5 ft from a quarter-second rolling median
        ok = ~np.isnan(pos[:, k, 0])
        filled = np.stack([np.interp(t, t[ok], pos[ok, k, ax]) for ax in range(2)], 1)
        med = np.stack([medfilt(filled[:, ax], 15) for ax in range(2)], 1)
        ok &= np.linalg.norm(filled - med, axis=1) < 1.5
        for ax in range(2):
            out[:, k, ax] = filtfilt(b, a, np.interp(t, t[ok], pos[ok, k, ax]))
    return out


def stats(pos, sm, src):
    rows = []
    for k in range(4):
        ok = ~np.isnan(pos[:, k, 0])
        resid = np.linalg.norm(pos[ok, k] - sm[ok, k], axis=1)
        v_raw = np.linalg.norm(np.diff(pos[:, k], axis=0), axis=1) * FPS
        v = np.linalg.norm(np.diff(sm[:, k], axis=0), axis=1) * FPS
        a_raw = np.linalg.norm(np.diff(pos[:, k], 2, axis=0), axis=1) * FPS ** 2
        a = np.linalg.norm(np.diff(sm[:, k], 2, axis=0), axis=1) * FPS ** 2
        # longest run of missing frames
        run = best = 0
        for m in ~ok:
            run = run + 1 if m else 0
            best = max(best, run)
        rows.append({"player": NAMES[k], "detected_pct": round(100 * ok.mean(), 1), "longest_gap_frames": best,
                     "both_ankles_pct": round(100 * (src[:, k] == 3).mean(), 1),
                     "raw_minus_smooth_rms_ft": round(float(np.sqrt(np.mean(resid ** 2))), 3),
                     "raw_speed_p50_ftps": round(float(np.nanmedian(v_raw)), 2),
                     "smooth_speed_p50_ftps": round(float(np.median(v)), 2),
                     "smooth_speed_max_ftps": round(float(v.max()), 1),
                     "raw_accel_rms_ftps2": round(float(np.sqrt(np.nanmean(a_raw ** 2))), 0),
                     "smooth_accel_rms_ftps2": round(float(np.sqrt(np.mean(a ** 2))), 1),
                     "distance_ft": round(float(v.sum() / FPS), 1),
                     "x_range_ft": [round(float(sm[:, k, 0].min()), 1), round(float(sm[:, k, 0].max()), 1)],
                     "y_range_ft": [round(float(sm[:, k, 1].min()), 1), round(float(sm[:, k, 1].max()), 1)]})
    return rows


# ---------- rendering ----------
PX = 15.0
MX, MY = 9.0, 13.0
PW, PH = 640, 1080


def court_xy(g):
    """Court feet -> panel pixels (near baseline at the bottom)."""
    x0 = (PW - (W + 2 * MX) * PX) / 2
    y0 = (PH - (L + 2 * MY) * PX) / 2
    g = np.asarray(g, float)
    return np.stack([x0 + (g[..., 0] + MX) * PX, y0 + (L + MY - g[..., 1]) * PX], -1)


def court_panel():
    im = np.full((PH, PW, 3), (92, 62, 38), np.uint8)
    a, b = court_xy([0, L]).astype(int), court_xy([W, 0]).astype(int)
    cv2.rectangle(im, tuple(a), tuple(b), (190, 110, 50), -1)
    k1, k2 = court_xy([0, NET_Y + 7]).astype(int), court_xy([W, NET_Y - 7]).astype(int)
    cv2.rectangle(im, tuple(k1), tuple(k2), (150, 88, 42), -1)
    for s, e in LINES.values():
        p1, p2 = court_xy(s[:2]), court_xy(e[:2])
        cv2.line(im, tuple(p1.round().astype(int)), tuple(p2.round().astype(int)), (255, 255, 255), 2, cv2.LINE_AA)
    n1, n2 = court_xy([-1, NET_Y]).round().astype(int), court_xy([W + 1, NET_Y]).round().astype(int)
    cv2.line(im, tuple(n1), tuple(n2), (30, 30, 30), 4, cv2.LINE_AA)
    return im


def render(video, start_frame, n, p, size, pos, sm, kpts, out):
    base = court_panel()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ang = np.linspace(0, 2 * np.pi, 40)
    speed = np.linalg.norm(np.gradient(sm, axis=0), axis=2) * FPS
    trail = int(1.5 * FPS)
    for i, f in enumerate(frames(video, start_frame, n, size)):
        f = f.copy()
        panel = base.copy()
        for k in range(4):
            g = sm[i, k]
            ring = np.stack([g[0] + 1.3 * np.cos(ang), g[1] + 1.3 * np.sin(ang), np.zeros(40)], 1)
            uv = project(p, ring, size)
            cv2.polylines(f, [np.round(uv * 8).astype(np.int32).reshape(-1, 1, 2)], True, COLORS[k], 3, cv2.LINE_AA, shift=3)
            if not np.isnan(pos[i, k, 0]):
                c = project(p, [(pos[i, k, 0], pos[i, k, 1], 0)], size)[0]
                cv2.circle(f, (int(c[0]), int(c[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
            tr = court_xy(sm[max(0, i - trail):i + 1, k])
            for j in range(1, len(tr)):
                w = j / len(tr)
                col = tuple(int(c * w + b * (1 - w)) for c, b in zip(COLORS[k], (150, 88, 42)))
                cv2.line(panel, tuple(tr[j - 1].round().astype(int)), tuple(tr[j].round().astype(int)), col, 2, cv2.LINE_AA)
            c = court_xy(g).round().astype(int)
            cv2.circle(panel, tuple(c), 11, COLORS[k], -1, cv2.LINE_AA)
            cv2.circle(panel, tuple(c), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"{speed[i, k]:.0f} ft/s", (c[0] + 16, c[1] + 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.zeros((1080, 1920, 3), np.uint8)
        canvas[180:900, :1280] = cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA)
        canvas[:, 1280:] = panel
        cv2.putText(canvas, "Players placed on the court from one camera", (24, 136), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "ring: smoothed ground position   white dot: raw per-frame estimate from the ankles",
                    (24, 164), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Right: true top-down court, 1.5 s trails, speed in ft/s. No ball yet.",
                    (24, 934), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()


def main():
    video, calib, start, n, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    p, size = load_camera(calib)
    cache = Path(prefix + "_tracks.npz")
    if cache.exists():
        z = np.load(cache)
        pos, src, kpts = z["pos"].copy(), z["src"], z["kpts"]
        for i, k in zip(*np.nonzero(src >= 2)):   # recompute ankle-based positions with the current rule
            kp = kpts[i, k]
            ank = kp[[15, 16]][kp[[15, 16], 2] > 0.5, :2].astype(float)
            pos[i, k] = feet_to_ground(p, size, ank)
    else:
        import time
        t0 = time.time()
        dets = detect(video, start, n, p, size, str(Path(__file__).resolve().parent.parent / "models/yolo11m-pose.pt"))
        print(f"detected {len(dets)} frames in {time.time() - t0:.0f}s; people per frame: "
              f"{np.bincount([len(d) for d in dets]).tolist()}")
        pos, src, kpts, init = track(dets)
        print("tracking initialised at frame", init)
    sm = smooth(pos)
    np.savez(cache, pos=pos, src=src, kpts=kpts, smooth=sm, start_frame=start, fps=FPS)
    rows = stats(pos, sm, src)
    Path(prefix + "_stats.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)
    import os
    if not os.environ.get("NO_RENDER"):
        render(video, start, len(pos), p, size, pos, sm, kpts, prefix + "_dots.mp4")
        print("wrote", prefix + "_dots.mp4")


if __name__ == "__main__":
    main()
