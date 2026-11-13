"""Render a demo video from the solved camera: the broadcast frame with the 3D court model
drawn on it (left) and a top-down re-projection of the ground plane (right).

Usage: python render_overlay.py <video> <calibration.json> <out.mp4> <start_s> <duration_s>
"""
import json
import subprocess
import sys

import cv2
import numpy as np

from calibrate import IN, L, LINES, NET_Y, W, net_tape_model, project

FPS = 60
LEFT_W, LEFT_H = 1280, 720
OUT_W, OUT_H = 1920, 1080
PX_PER_FT = 16.0
MARGIN_X, MARGIN_Y = 10.0, 11.75  # ft of floor shown beyond the court


def poly(img, pts, color, thick=2, closed=False):
    cv2.polylines(img, [np.round(pts * 8).astype(np.int32).reshape(-1, 1, 2)], closed, color, thick, cv2.LINE_AA, shift=3)


def seg3d(a, b, n=40):
    a, b = np.array(a, float), np.array(b, float)
    return a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]


def draw_model(frame, p, size):
    red, green, yellow, cyan = (60, 60, 255), (80, 255, 80), (0, 230, 255), (255, 220, 0)
    for a, b in LINES.values():
        poly(frame, project(p, seg3d(a, b), size), red, 2)
    # net: tape, posts, bottom edge
    poly(frame, project(p, net_tape_model(), size), green, 2)
    for x in (-1.0, W + 1.0):
        poly(frame, project(p, seg3d((x, NET_Y, 0), (x, NET_Y, 3.0), 2), size), green, 2)
    # a 10 ft tall "glass wall" above the net plane with 1 ft height rungs
    for z in range(4, 11):
        c = cyan if z % 5 else (255, 255, 255)
        poly(frame, project(p, seg3d((0, NET_Y, z), (W, NET_Y, z)), size), c, 1)
        uv = project(p, [(W, NET_Y, z)], size)[0]
        if z in (5, 10):
            cv2.putText(frame, f"{z} ft", (int(uv[0]) + 8, int(uv[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for x in (0, W):
        poly(frame, project(p, seg3d((x, NET_Y, 3.0), (x, NET_Y, 10.0), 2), size), cyan, 1)
    # 6 ft poles with 1 ft ticks at court and kitchen corners
    for gx in (0, W):
        for gy in (0, 15, 29, L):
            poly(frame, project(p, seg3d((gx, gy, 0), (gx, gy, 6.0), 2), size), yellow, 2)
            for z in range(1, 7):
                d = 0.25 if gx == 0 else -0.25
                poly(frame, project(p, seg3d((gx, gy, z), (gx - d, gy, z), 2), size), yellow, 1)


def topdown_maps(p, size):
    """Pixel lookup maps that re-render the Z=0 plane as seen from directly above.
    Near baseline at the bottom, to match the broadcast view."""
    tw = int((W + 2 * MARGIN_X) * PX_PER_FT)
    th = int((L + 2 * MARGIN_Y) * PX_PER_FT)
    xs = (np.arange(tw) + 0.5) / PX_PER_FT - MARGIN_X
    ys = L + MARGIN_Y - (np.arange(th) + 0.5) / PX_PER_FT
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    uv = project(p, P, size).reshape(th, tw, 2).astype(np.float32)
    return uv[..., 0], uv[..., 1], tw, th


def main():
    video, calib, out = sys.argv[1], sys.argv[2], sys.argv[3]
    start, dur = float(sys.argv[4]), float(sys.argv[5])
    ref = json.loads(open(calib).read())
    e = ref["no_distortion"]
    p = np.r_[e["f_px"], 0.0, e["rvec"], e["tvec"]]
    size = tuple(ref["image_size"])
    mx, my, tw, th = topdown_maps(p, size)

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", video,
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    nbytes = size[0] * size[1] * 3
    n = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3).copy()
        top = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        draw_model(frame, p, size)

        canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
        left = cv2.resize(frame, (LEFT_W, LEFT_H), interpolation=cv2.INTER_AREA)
        y0 = (OUT_H - LEFT_H) // 2
        canvas[y0:y0 + LEFT_H, :LEFT_W] = left
        top = cv2.resize(top, (int(tw * OUT_H / th), OUT_H), interpolation=cv2.INTER_AREA)
        x0 = LEFT_W + (OUT_W - LEFT_W - top.shape[1]) // 2
        canvas[:, x0:x0 + top.shape[1]] = top

        cv2.putText(canvas, "Broadcast frame + calibrated 3D court model", (24, y0 - 44), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "red: court lines   green: net (34-36 in)   yellow: 6 ft poles   cyan: height rungs over the net, 1 ft apart",
                    (24, y0 - 14), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "One calibration from a single frame, held fixed for the whole clip. No tracking yet.",
                    (24, y0 + LEFT_H + 34), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Top-down view of the floor (right) is exact for the court surface only:",
                    (24, y0 + LEFT_H + 62), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "players and the net smear because they are not flat on the ground.",
                    (24, y0 + LEFT_H + 88), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        enc.stdin.write(canvas.tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    print(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
