"""Cut one short overlaid video per sampled shot, for checking the shot table by hand.

Usage: uv run python shots/make_check.py     (reads shots/out/samples.json; writes shots/out/check/c<NN>.mp4 and items.json)
Each video runs from 0.8 s before the hit to 0.6 s after the ball arrives (at most 4.5 s). Drawn on it:
  yellow ring   the player the table says hit the ball (label HIT appears at the hit frame)
  green cross   where the table says the ball bounced; green ring = the opponent it says volleyed it
The two samples are shuffled together so the checker cannot tell which a shot came from.
"""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fit3d"))
from physics import Camera  # noqa: E402
CAM = Camera(json.loads((ROOT / "calib/out/t270_calibration.json").read_text()))
VID = str(ROOT / "data/video/IpUxCoBHg_g.mp4")


def floor_px(xy):
    return CAM.project(np.array([[xy[0], xy[1], 0.0]]))[0]


def ring(img, xy, col, r_ft=1.6):
    pts = np.array([[xy[0] + r_ft * np.cos(a), xy[1] + r_ft * np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 40)])
    cv2.polylines(img, [CAM.project(pts).astype(np.int32)], True, col, 4, cv2.LINE_AA)


def main():
    rows = json.loads((ROOT / "shots/out/shots.json").read_text()); S = json.loads((ROOT / "shots/out/samples.json").read_text())
    shots = {f"b_{int(s['start_s'])}": s for s in json.loads((ROOT / "cuts/out/IpUxCoBHg_g_shots.json").read_text())["shots"] if s["label"] == "baseline"}
    ids = [(i, "random") for i in S["random"]] + [(i, "geometric") for i in S["geometric"]]
    order = np.random.default_rng(S["seed"]).permutation(len(ids))
    out = ROOT / "shots/out/check"; out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(VID); items = []
    for n, j in enumerate(order):
        i, sample = ids[j]; r = rows[i]
        g = np.load(ROOT / f"batch/out/{r['clip']}_tracks.npz")["smooth"]
        start = shots[r["clip"]]["start_frame"] + 1
        f_hit = min(len(g) - 1, int(round(r["t"] * 60))); dt = r["dt"] or 1.2
        f0 = max(0, f_hit - 48); f1 = min(len(g) - 1, f_hit + int(min(dt, 3.1) * 60) + 36)
        far = r["side"] == "far"
        tgt = None if r["target"] is None else ([20 - r["target"][0], 44 - r["target"][1]] if far else r["target"])
        name = f"c{n:02d}"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", "60", "-i", "-",
                               "-vf", "scale=1280:720", "-c:v", "libx264", "-crf", "27", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / f"{name}.mp4")], stdin=subprocess.PIPE)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + f0)
        for f in range(f0, f1 + 1):
            ok, img = cap.read()
            if not ok:
                break
            if not np.isnan(g[f, r["hitter"]]).any():
                ring(img, g[f, r["hitter"]], (0, 220, 255))
            if tgt is not None:
                if r["target_kind"] == "bounce":
                    p = floor_px(tgt).astype(int)
                    cv2.drawMarker(img, tuple(p), (80, 230, 90), cv2.MARKER_TILTED_CROSS, 34, 5, cv2.LINE_AA)
                else:
                    ring(img, tgt, (80, 230, 90), 1.2)
            if abs(f - f_hit) <= 4:
                p = floor_px(g[f_hit, r["hitter"]]).astype(int)
                cv2.putText(img, "HIT", (p[0] - 40, p[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 220, 255), 5, cv2.LINE_AA)
            ff.stdin.write(img.tobytes())
        ff.stdin.close(); ff.wait()
        items.append({"id": name, "row": i, "sample": sample, "clip": r["clip"], "t": r["t"], "target_kind": r["target_kind"], "hit_at_s": round((f_hit - f0) / 60, 2), "frames": f1 - f0 + 1})
        print(name, r["clip"], r["t"], r["target_kind"], flush=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))


if __name__ == "__main__":
    main()
