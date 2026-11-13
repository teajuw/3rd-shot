"""Render the fitted 3D ball path: broadcast with the path reprojected, a synthetic view
from an orbiting virtual camera, a side elevation and a top-down view.

Usage: python render.py <video> <calibration.json> <path.npz> <players_tracks.npz> <fit.json> <out.mp4>
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "players"))
from physics import Camera, R_BALL  # noqa: E402
from track_players import frames  # noqa: E402

W, L, NET_Y = 20.0, 44.0, 22.0
PCOL = [(60, 170, 255), (80, 220, 120), (255, 150, 60), (230, 110, 230)]
BALL = (60, 255, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class VCam:
    """Simple look-at pinhole camera for the synthetic view."""
    def __init__(self, eye, target, f, size):
        self.eye = np.array(eye, float)
        z = np.array(target, float) - self.eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.R = np.stack([x, y, z])
        self.f, self.size = f, size

    def project(self, P):
        c = (np.atleast_2d(P) - self.eye) @ self.R.T
        return np.stack([self.f * c[:, 0] / c[:, 2] + self.size[0] / 2, self.f * c[:, 1] / c[:, 2] + self.size[1] / 2], 1), c[:, 2]


def pts(a):
    return np.round(np.asarray(a) * 8).astype(np.int32).reshape(-1, 1, 2)


def court_lines():
    segs = [((0, 0), (W, 0)), ((0, L), (W, L)), ((0, 0), (0, L)), ((W, 0), (W, L)), ((0, 15), (W, 15)), ((0, 29), (W, 29)),
            ((W / 2, 0), (W / 2, 15)), ((W / 2, 29), (W / 2, L))]
    return [(np.array([*a, 0.0]), np.array([*b, 0.0])) for a, b in segs]


def draw_virtual(vc, ball, trail, players, size):
    im = np.full((size[1], size[0], 3), (70, 48, 30), np.uint8)

    def poly(P, col, fill=False, th=2):
        uv, z = vc.project(P)
        if (z < 1).any():
            return
        if fill:
            cv2.fillPoly(im, [pts(uv)], col, cv2.LINE_AA, shift=3)
        else:
            cv2.polylines(im, [pts(uv)], False, col, th, cv2.LINE_AA, shift=3)
    poly([(-12, -14, 0), (W + 12, -14, 0), (W + 12, L + 14, 0), (-12, L + 14, 0)], (96, 66, 40), True)
    poly([(0, 0, 0), (W, 0, 0), (W, L, 0), (0, L, 0)], (190, 110, 50), True)
    poly([(0, 15, 0), (W, 15, 0), (W, 29, 0), (0, 29, 0)], (150, 88, 42), True)
    for a, b in court_lines():
        poly([a, b], (255, 255, 255), th=2)
    # draw things back to front relative to the virtual camera
    items = []
    net = [(-1, NET_Y, 0), (-1, NET_Y, 3.0), (W / 2, NET_Y, 34 / 12), (W + 1, NET_Y, 3.0), (W + 1, NET_Y, 0)]
    items.append((np.linalg.norm(np.array([W / 2, NET_Y, 1.5]) - vc.eye), "net", net))
    for k, g in enumerate(players):
        items.append((np.linalg.norm(np.array([g[0], g[1], 3]) - vc.eye), "player", (k, g)))
    if ball is not None:
        items.append((np.linalg.norm(ball - vc.eye), "ball", ball))
    for _, kind, obj in sorted(items, key=lambda i: -i[0]):
        if kind == "net":
            ov = im.copy()
            uv, _ = vc.project(obj)
            cv2.fillPoly(ov, [pts(uv)], (40, 40, 40), cv2.LINE_AA, shift=3)
            cv2.addWeighted(ov, 0.55, im, 0.45, 0, im)
            poly(obj[1:4], (255, 255, 255), th=3)
            poly(obj[:2], (30, 30, 30), th=3)
            poly(obj[3:], (30, 30, 30), th=3)
        elif kind == "player":
            k, g = obj
            ring = [(g[0] + 1.0 * np.cos(a), g[1] + 1.0 * np.sin(a), 0) for a in np.linspace(0, 2 * np.pi, 24)]
            poly(ring, tuple(int(c * 0.6) for c in PCOL[k]), True)
            uv, z = vc.project([(g[0], g[1], 0.2), (g[0], g[1], 4.9), (g[0], g[1], 5.5)])
            w = max(2, int(vc.f * 1.1 / z[0]))
            cv2.line(im, tuple(pts(uv[0])[0, 0]), tuple(pts(uv[1])[0, 0]), PCOL[k], w, cv2.LINE_AA, shift=3)
            cv2.circle(im, tuple(pts(uv[2])[0, 0]), max(2, int(vc.f * 0.45 / z[0])) * 8, PCOL[k], -1, cv2.LINE_AA, shift=3)
        else:
            sh, zs = vc.project([(obj[0], obj[1], 0.02)])
            r = max(2, int(vc.f * 0.3 / zs[0]))
            cv2.ellipse(im, tuple(np.round(sh[0]).astype(int)), (r, max(1, r // 2)), 0, 0, 360, (40, 30, 20), -1, cv2.LINE_AA)
            if len(trail) > 1:
                poly(trail, BALL, th=2)
            uv, z = vc.project([obj])
            cv2.circle(im, tuple(pts(uv[0])[0, 0]), max(3, int(vc.f * 0.28 / z[0])) * 8, BALL, -1, cv2.LINE_AA, shift=3)
    return im


def draw_side(ball, trail, players, events, size):
    """Elevation looking across the court: Y along the page, height up. True scale."""
    im = np.full((size[1], size[0], 3), (40, 30, 22), np.uint8)
    s = size[0] / 34.0                                     # px per ft, showing Y in [5, 39]
    def m(y, z):
        return (int((y - 5) * s), int(size[1] - 40 - z * s))
    cv2.line(im, m(5, 0), m(39, 0), (190, 110, 50), 3)
    for y, lab in ((15, "kitchen line"), (29, "kitchen line"), (22, "net")):
        cv2.putText(im, lab, (m(y, 0)[0] - 44, size[1] - 12), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(im, m(y, 0), m(y, -0.2), (255, 255, 255), 2)
    cv2.line(im, m(22, 0), m(22, 34 / 12), (255, 255, 255), 3)
    for z in (2, 4, 6):
        cv2.line(im, m(5, z), m(39, z), (70, 60, 50), 1)
        cv2.putText(im, f"{z} ft", (6, m(5, z)[1] - 4), FONT, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    for e in events:
        if e["kind"] == "bounce":
            cv2.circle(im, m(e["pos_ft"][1], 0), 4, (120, 255, 120), -1, cv2.LINE_AA)
    for k, g in enumerate(players):
        cv2.line(im, m(g[1], 0), m(g[1], 5.6), PCOL[k], 5, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(im, [np.array([m(p[1], p[2]) for p in trail], np.int32)], False, BALL, 2, cv2.LINE_AA)
    if ball is not None:
        cv2.circle(im, m(ball[1], ball[2]), 7, BALL, -1, cv2.LINE_AA)
        cv2.putText(im, f"height {ball[2]:.1f} ft", (size[0] - 190, 28), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(im, "Side view, true scale (camera end at left)", (12, 28), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def draw_top(ball, trail, players, events, size):
    im = np.full((size[1], size[0], 3), (70, 48, 30), np.uint8)
    s = min(size[0] / 56.0, size[1] / 30.0)
    def m(x, y):
        return (int(size[0] / 2 + (y - 22) * s), int(size[1] / 2 + (x - 10) * s))
    cv2.rectangle(im, m(0, 0), m(W, L), (190, 110, 50), -1)
    cv2.rectangle(im, m(0, 15), m(W, 29), (150, 88, 42), -1)
    for a, b in court_lines():
        cv2.line(im, m(a[0], a[1]), m(b[0], b[1]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(im, m(-1, 22), m(W + 1, 22), (30, 30, 30), 4)
    for e in events:
        if e["kind"] == "bounce":
            cv2.circle(im, m(*e["pos_ft"][:2]), 4, (120, 255, 120), -1, cv2.LINE_AA)
    for k, g in enumerate(players):
        cv2.circle(im, m(g[0], g[1]), 9, PCOL[k], -1, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(im, [np.array([m(p[0], p[1]) for p in trail], np.int32)], False, BALL, 2, cv2.LINE_AA)
    if ball is not None:
        cv2.circle(im, m(ball[0], ball[1]), 6, BALL, -1, cv2.LINE_AA)
    cv2.putText(im, "Top view (green: bounces)", (12, 28), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return im


CAPTION = "Players are placeholders at their tracked floor positions. Hits and bounces were labelled by hand for this clip."


def main():
    global CAPTION
    video, calib, path_npz, players_npz, fit_json, out = sys.argv[1:7]
    if len(sys.argv) > 7:
        CAPTION = sys.argv[7]
    cam = Camera(json.loads(Path(calib).read_text()))
    z = np.load(path_npz)
    path, keep, i0, i1, start = z["path"], z["keep"], int(z["i0"]), int(z["i1"]), int(z["start_frame"])
    ground = np.load(players_npz)["smooth"]
    events = json.loads(Path(fit_json).read_text())["events"]
    f0, f1 = int(keep[i0]) - 30, int(keep[i1]) + 30          # half a second of lead-in and tail
    uniq_of = np.searchsorted(keep, np.arange(len(ground)), side="right") - 1   # frame -> unique index
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                            "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = f1 - f0 + 1
    for k, f in enumerate(frames(video, start + f0, n, (1920, 1080))):
        fr = f0 + k
        u = int(uniq_of[fr])
        ball = path[u] if not np.isnan(path[u, 0]) else None
        tr = path[max(i0, u - 30):u + 1]
        tr = tr[~np.isnan(tr[:, 0])]
        f = f.copy()
        if len(tr) > 1:
            cv2.polylines(f, [pts(cam.project(tr))], False, BALL, 3, cv2.LINE_AA, shift=3)
            sh = tr.copy()
            sh[:, 2] = 0
            cv2.polylines(f, [pts(cam.project(sh))], False, (30, 30, 30), 2, cv2.LINE_AA, shift=3)
        if ball is not None:
            uv = cam.project(ball[None])[0]
            cv2.circle(f, tuple(np.round(uv).astype(int)), 13, (0, 0, 255), 2, cv2.LINE_AA)
            g = cam.project(np.array([[ball[0], ball[1], 0.0]]))[0]
            cv2.line(f, tuple(np.round(uv).astype(int)), tuple(np.round(g).astype(int)), (30, 30, 30), 1, cv2.LINE_AA)
        ang = np.radians(215 + 110 * k / n)
        vc = VCam((10 + 58 * np.cos(ang), 22 + 58 * np.sin(ang), 15), (10, 22, 2), 1150, (960, 540))
        canvas = np.zeros((1080, 1920, 3), np.uint8)
        canvas[40:580, :960] = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA)
        canvas[40:580, 960:] = draw_virtual(vc, ball, tr, ground[fr], (960, 540))
        canvas[620:1040, :1180] = draw_side(ball, tr, ground[fr], events, (1180, 420))
        canvas[620:1040, 1200:] = draw_top(ball, tr, ground[fr], events, (720, 420))
        cv2.putText(canvas, "Broadcast: fitted 3D path reprojected (dark line = its shadow on the floor)", (12, 28), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Synthetic view from an orbiting virtual camera", (972, 28), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, CAPTION,
                    (12, 1068), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", out, n, "frames")


if __name__ == "__main__":
    main()
