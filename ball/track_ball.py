"""Link per-frame ball candidates into one 2D track, build a random check sheet, render.

Linking is a dynamic programme over candidates: each candidate carries a weight (how
ball-coloured it is, plus a bonus when TrackNet agrees), consecutive picks must be
reachable at a plausible ball speed, skipped frames cost a little, and restarting the
track costs more. The best-scoring path is the track.

Usage: python track_ball.py <video> <cands.npz> <out_prefix> [mode]
  mode: fused (default) | colour | tracknet
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "players"))
from track_players import frames  # noqa: E402

SIZE = (1920, 1080)
VMAX = 90.0       # px per frame at 60 fps
MAXGAP = 12       # frames a track may skip
GAP_COST = 0.15
RESTART_COST = 4.0
AGREE_R = 15.0


def build_nodes(z, mode):
    C, T1, T2 = z["colour"], z["tracknet_s1"], z["tracknet_s2"]
    nodes = []
    for i in range(len(C)):
        tn = np.concatenate([T1[i], T2[i]])
        tn = tn[~np.isnan(tn[:, 0])]
        out = []
        if mode in ("fused", "colour"):
            for c in C[i][~np.isnan(C[i][:, 0])]:
                w = float(np.clip((c[2] - 15) / 40, 0, 1))
                agree = mode == "fused" and len(tn) and np.min(np.linalg.norm(tn[:, :2] - c[:2], axis=1)) < AGREE_R
                if agree:
                    w += 1.0
                if w > 0.05:
                    out.append((c[0], c[1], w, 1 + 2 * bool(agree)))
        if mode in ("fused", "tracknet"):
            for t in tn:
                if not any(np.hypot(o[0] - t[0], o[1] - t[1]) < AGREE_R for o in out):
                    out.append((t[0], t[1], 0.6 if mode == "fused" else 1.0, 2))
        nodes.append(np.array(out, np.float32).reshape(-1, 4))
    return nodes  # columns: x, y, weight, source (1 colour, 2 tracknet, 3 both)


def link(nodes):
    """One global best path. A node may continue from a reachable earlier node, or restart
    from the best path that ended on any earlier frame (paying RESTART_COST)."""
    n = len(nodes)
    best = [np.zeros(len(x)) for x in nodes]
    prev = [np.full((len(x), 3), -1, int) for x in nodes]   # (frame, index, restarted)
    run_best, run_arg = 0.0, (-1, -1)                       # best path ending strictly before frame t
    for t in range(n):
        for b in range(len(nodes[t])):
            xb, yb, wb = nodes[t][b, :3]
            s, p = run_best + wb - RESTART_COST, (run_arg[0], run_arg[1], 1)
            for g in range(1, MAXGAP + 1):
                if t - g < 0:
                    break
                A = nodes[t - g]
                if len(A) == 0:
                    continue
                d = np.hypot(A[:, 0] - xb, A[:, 1] - yb)
                cand = best[t - g] + wb - GAP_COST * (g - 1) - 0.3 * d / (VMAX * g)
                cand[d > VMAX * g] = -np.inf
                a = int(np.argmax(cand))
                if cand[a] > s:
                    s, p = cand[a], (t - g, a, 0)
            best[t][b], prev[t][b] = s, p
        if len(nodes[t]) and best[t].max() > run_best:
            run_best, run_arg = float(best[t].max()), (t, int(best[t].argmax()))
    track = np.full((n, 4), np.nan, np.float32)
    t, b = run_arg
    chain = []
    while t >= 0:
        chain.append((t, b))
        pt, pb, restarted = prev[t][b]
        if restarted or pt < 0:
            if len(chain) >= 5:               # drop fragments shorter than 5 detections
                for tt, bb in chain:
                    track[tt] = nodes[tt][bb]
            chain = []
        t, b = pt, pb
    return track


def fill(track, max_gap=8):
    """Linear interpolation across short gaps (display only; flagged as interpolated)."""
    out = track[:, :2].copy()
    ok = ~np.isnan(out[:, 0])
    idx = np.flatnonzero(ok)
    interp = np.zeros(len(out), bool)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap:
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                out[k] = out[a] * (1 - w) + out[b] * w
                interp[k] = True
    return out, interp


def check_sheet(video, start, track, path, k=30, seed=3):
    random.seed(seed)
    pick = sorted(random.sample(range(len(track)), k))
    tiles = []
    for i, f in enumerate(frames(video, start, pick[-1] + 1, SIZE)):
        if i not in pick:
            continue
        x, y = track[i, :2]
        if np.isnan(x):
            tile = cv2.resize(f, (480, 270))
            cv2.putText(tile, f"#{i} NO DETECTION", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            tile = cv2.copyMakeBorder(tile, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
            tile = cv2.resize(tile, (270, 270))
        else:
            x0, y0 = int(np.clip(x - 45, 0, SIZE[0] - 90)), int(np.clip(y - 45, 0, SIZE[1] - 90))
            tile = cv2.resize(f[y0:y0 + 90, x0:x0 + 90], (270, 270), interpolation=cv2.INTER_CUBIC)
            cx, cy = int((x - x0) * 3), int((y - y0) * 3)
            cv2.circle(tile, (cx, cy), 30, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, f"#{i} src{int(track[i, 3])}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]
    cv2.imwrite(path, np.vstack(rows))
    return pick


def render(video, start, track, out):
    import subprocess
    xy, interp = fill(track)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                            "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for i, f in enumerate(frames(video, start, len(track), SIZE)):
        f = f.copy()
        for j in range(max(1, i - 45), i + 1):
            if np.isnan(xy[j, 0]) or np.isnan(xy[j - 1, 0]):
                continue
            a = (j - (i - 45)) / 45
            cv2.line(f, tuple(xy[j - 1].round().astype(int)), tuple(xy[j].round().astype(int)),
                     (0, int(255 * a), 255), 2, cv2.LINE_AA)
        if not np.isnan(xy[i, 0]):
            col = (160, 160, 160) if interp[i] else (0, 0, 255)
            cv2.circle(f, tuple(xy[i].round().astype(int)), 14, col, 2, cv2.LINE_AA)
        cv2.putText(f, "Ball track: red ring = detected, grey ring = filled across a short gap, trail = last 0.75 s",
                    (24, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    enc.wait()


def main():
    video, cands, prefix = sys.argv[1:4]
    mode = sys.argv[4] if len(sys.argv) > 4 else "fused"
    z = np.load(cands)
    start = int(z["start_frame"])
    track = link(build_nodes(z, mode))
    det = ~np.isnan(track[:, 0])
    xy, interp = fill(track)
    covered = ~np.isnan(xy[:, 0])
    gaps, run = [], 0
    for c in covered:
        if not c:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    if run:
        gaps.append(run)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    stats = {"mode": mode, "frames": len(track), "detected_pct": round(100 * det.mean(), 1),
             "with_short_gaps_filled_pct": round(100 * covered.mean(), 1),
             "source_colour_only_pct": round(100 * (track[:, 3] == 1).mean(), 1),
             "source_tracknet_only_pct": round(100 * (track[:, 3] == 2).mean(), 1),
             "source_both_pct": round(100 * (track[:, 3] == 3).mean(), 1),
             "remaining_gaps": len(gaps), "longest_gap_frames": max(gaps) if gaps else 0,
             "step_px_p50": round(float(np.nanmedian(step)), 1), "step_px_p99": round(float(np.nanpercentile(step, 99)), 1)}
    print(json.dumps(stats))
    np.savez(f"{prefix}_{mode}_track.npz", track=track, filled=xy, interpolated=interp, start_frame=start)
    Path(f"{prefix}_{mode}_stats.json").write_text(json.dumps(stats, indent=1))
    import os
    if mode == "fused" and not os.environ.get("NO_RENDER"):
        pick = check_sheet(video, start, track, f"{prefix}_check.jpg", seed=int(sys.argv[5]) if len(sys.argv) > 5 else 3)
        Path(f"{prefix}_check_frames.json").write_text(json.dumps(pick))
        render(video, start, track, f"{prefix}_ball.mp4")


if __name__ == "__main__":
    main()
