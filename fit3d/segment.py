"""Prepare the 2D ball track for 3D fitting.

1. Remove duplicated frames (the 60 fps file is 50 fps footage with every 6th frame
   repeated) and put the track on a true 50 Hz clock.
2. Split it into arcs with an optimal piecewise-quadratic segmentation: within one flight
   the image path is close to quadratic in time; a hit or bounce breaks that.
3. Label each break as a hit (a player's wrist is close) or a bounce (none is).

Usage: python segment.py <ball_track.npz> <players_tracks.npz> <signal.npz> <out_prefix>
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC_FPS = 50.0
CAP = 4.0          # px: residuals are capped here so one bad detection cannot split an arc
LAMBDA = 150.0     # cost of starting a new arc, in capped squared pixels
MINLEN, MAXLEN = 5, 110
PLAY_GAP = 25      # unique frames: a longer hole separates stretches of play


def unique_clock(n, start_frame, signal_path):
    """Indices of non-duplicate frames and their time in seconds on the 50 Hz source clock."""
    d = np.load(signal_path)["diff"][start_frame:start_frame + n]
    dup_idx = np.flatnonzero(d < 0.04)
    phase = int(np.bincount(dup_idx % 6, minlength=6).argmax())
    keep = np.flatnonzero(np.arange(n) % 6 != phase)
    return keep, np.arange(len(keep)) / SRC_FPS, phase


def seg_cost(t, xy):
    """Capped squared residual of a quadratic-in-time fit to x(t), y(t); NaNs ignored."""
    ok = ~np.isnan(xy[:, 0])
    if ok.sum() < MINLEN:
        return np.inf, None
    tt = t[ok] - t[ok][0]
    A = np.stack([np.ones_like(tt), tt, tt * tt], 1)
    coef, *_ = np.linalg.lstsq(A, xy[ok], rcond=None)
    r = np.linalg.norm(A @ coef - xy[ok], axis=1)
    return float(np.minimum(r, CAP).__pow__(2).sum()), coef


def segment(t, xy):
    """DP over break positions. Arcs share their end sample with the next arc's start."""
    n = len(t)
    best = np.full(n, np.inf)
    back = np.full(n, -1)
    best[0] = 0
    for j in range(MINLEN - 1, n):
        if np.isnan(xy[j, 0]):
            continue
        for i in range(max(0, j - MAXLEN), j - MINLEN + 2):
            if not np.isfinite(best[i]) or np.isnan(xy[i, 0]):
                continue
            c, _ = seg_cost(t[i:j + 1], xy[i:j + 1])
            if best[i] + c + LAMBDA < best[j]:
                best[j], back[j] = best[i] + c + LAMBDA, i
    j = int(np.flatnonzero(np.isfinite(best))[-1])
    cuts = [j]
    while back[j] >= 0:
        j = int(back[j])
        cuts.append(j)
    return cuts[::-1]


def main():
    ball_path, players_path, signal_path, prefix = sys.argv[1:5]
    b = np.load(ball_path)
    track = b["track"].copy()
    start_frame = int(b["start_frame"])
    track[track[:, 0] < 8] = np.nan                      # detections stuck on the image border
    keep, t, phase = unique_clock(len(track), start_frame, signal_path)
    xy = track[keep, :2].astype(float)
    kp = np.load(players_path)["kpts"][keep]             # (n, 4, 17, 3)
    print(f"{len(track)} frames -> {len(keep)} unique (duplicate phase {phase}); {int((~np.isnan(xy[:, 0])).sum())} with a ball")

    # stretches of play: split where the ball is missing for more than PLAY_GAP unique frames
    ok = ~np.isnan(xy[:, 0])
    idx = np.flatnonzero(ok)
    stretches, s = [], idx[0]
    for a, c in zip(idx[:-1], idx[1:]):
        if c - a > PLAY_GAP:
            stretches.append((int(s), int(a)))
            s = c
    stretches.append((int(s), int(idx[-1])))

    arcs, events = [], []
    for si, (a, c) in enumerate(stretches):
        if c - a < 20:
            continue
        cuts = [a + k for k in segment(t[a:c + 1], xy[a:c + 1])]
        for k, (i, j) in enumerate(zip(cuts[:-1], cuts[1:])):
            cost, coef = seg_cost(t[i:j + 1], xy[i:j + 1])
            n_obs = int((~np.isnan(xy[i:j + 1, 0])).sum())
            arcs.append({"stretch": si, "i": int(i), "j": int(j), "n_obs": n_obs,
                         "rms_px": round(float(np.sqrt(cost / n_obs)), 2)})
        for k, e in enumerate(cuts):
            # nearest wrist, in units of that player's shoulder-to-hip length
            best = (np.inf, -1)
            for pl in range(4):
                K = kp[e, pl]
                if np.isnan(K[0, 0]):
                    continue
                torso = np.linalg.norm(K[[5, 6], :2].mean(0) - K[[11, 12], :2].mean(0))
                dw = min(np.linalg.norm(K[w, :2] - xy[e]) for w in (9, 10))
                if dw / max(torso, 1) < best[0]:
                    best = (dw / max(torso, 1), pl)
            kind = "hit" if best[0] < 1.6 else "bounce"
            events.append({"stretch": si, "index": int(e), "t": round(float(t[e]), 3), "px": xy[e].round(1).tolist(),
                           "kind": kind, "player": int(best[1]), "wrist_dist_torsos": round(float(best[0]), 2),
                           "terminal": k in (0, len(cuts) - 1)})
    out = {"start_frame": start_frame, "dup_phase": phase, "keep": keep.tolist(), "t": t.tolist(),
           "xy": [[None if np.isnan(v) else float(v) for v in p] for p in xy],
           "stretches": stretches, "arcs": arcs, "events": events}
    Path(prefix + "_segments.json").write_text(json.dumps(out))
    print("stretches (unique-frame index):", stretches)
    print(f"{len(arcs)} arcs, {len(events)} events: {sum(e['kind'] == 'hit' for e in events)} hits, {sum(e['kind'] == 'bounce' for e in events)} bounces")
    print("arc 2D fit rms px:", np.round(np.percentile([a["rms_px"] for a in arcs], [50, 90, 100]), 2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(24, 9), sharex=True)
    ax[0].plot(t, xy[:, 1], ".", ms=3)
    ax[0].invert_yaxis()
    ax[0].set_ylabel("image y (px)")
    ax[1].plot(t, xy[:, 0], ".", ms=3)
    ax[1].set_ylabel("image x (px)")
    ax[1].set_xlabel("seconds (50 Hz source clock)")
    for e in events:
        for x in ax:
            x.axvline(e["t"], color="red" if e["kind"] == "hit" else "green", lw=0.8, alpha=0.8)
        ax[0].text(e["t"], 180, f"{'H' if e['kind'] == 'hit' else 'B'}{e['player'] if e['kind'] == 'hit' else ''}", fontsize=7, ha="center")
    for x in ax:
        x.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(prefix + "_segments.png", dpi=70)


if __name__ == "__main__":
    main()
