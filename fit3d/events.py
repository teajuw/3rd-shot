"""Turn the over-split 2D arcs into labelled events: hit (by which player) or bounce.

For each break between two arcs, compare the ball's image velocity just before and after,
and look at how close the nearest wrist is:
  bounce  the ball was moving down the screen and now moves up, sideways motion carries on,
          and no wrist is close
  hit     a wrist is close
  neither the break is not a real event; the two arcs are merged

Usage: python events.py <segments.json> <players_tracks.npz> <out_prefix>
"""
import json
import sys
from pathlib import Path

import numpy as np

HIT_DIST = 1.3       # nearest wrist within this many torso lengths
BOUNCE_CLEAR = 0.9   # a bounce needs every wrist farther than this


def quad(t, xy):
    ok = ~np.isnan(xy[:, 0])
    tt = t[ok] - t[0]
    A = np.stack([np.ones_like(tt), tt, tt * tt], 1)
    c, *_ = np.linalg.lstsq(A, xy[ok], rcond=None)
    T = t[-1] - t[0]
    return c[1], c[1] + 2 * c[2] * T          # velocity at start, at end (px/s)


def wrist_info(kp, xy, e):
    best = (np.inf, -1)
    for f in range(max(0, e - 2), min(len(xy), e + 3)):
        if np.isnan(xy[f, 0]):
            continue
        for pl in range(4):
            K = kp[f, pl]
            if np.isnan(K[0, 0]):
                continue
            torso = max(np.linalg.norm(K[[5, 6], :2].mean(0) - K[[11, 12], :2].mean(0)), 1.0)
            d = min(np.linalg.norm(K[w, :2] - xy[f]) for w in (9, 10)) / torso
            if d < best[0]:
                best = (float(d), pl)
    return best


def main():
    seg_path, players_path, prefix = sys.argv[1:4]
    seg = json.loads(Path(seg_path).read_text())
    t = np.array(seg["t"])
    xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
    kp = np.load(players_path)["kpts"][np.array(seg["keep"])]

    events = []
    for si in sorted({a["stretch"] for a in seg["arcs"]}):
        arcs = [a for a in seg["arcs"] if a["stretch"] == si]
        cuts = [arcs[0]["i"]] + [a["j"] for a in arcs]
        vel = [quad(t[a["i"]:a["j"] + 1], xy[a["i"]:a["j"] + 1]) for a in arcs]
        for k, e in enumerate(cuts):
            d, pl = wrist_info(kp, xy, e)
            if k in (0, len(cuts) - 1):
                kind = "hit" if d < HIT_DIST else "edge"
                events.append({"stretch": si, "index": int(e), "t": float(t[e]), "px": xy[e].tolist(), "kind": kind,
                               "player": pl if kind == "hit" else -1, "wrist": round(d, 2), "terminal": True})
                continue
            vA, vB = vel[k - 1][1], vel[k][0]
            down_then_up = vA[1] > 60 and vB[1] < vA[1] - 150
            sideways_kept = vA[0] * vB[0] > 0 or min(abs(vA[0]), abs(vB[0])) < 120
            if down_then_up and sideways_kept and d > BOUNCE_CLEAR:
                kind = "bounce"
            elif d < HIT_DIST:
                kind = "hit"
            else:
                continue   # not a real event: arcs merge
            events.append({"stretch": si, "index": int(e), "t": float(t[e]), "px": xy[e].tolist(), "kind": kind,
                           "player": pl if kind == "hit" else -1, "wrist": round(d, 2), "terminal": False,
                           "v_before": np.round(vA).tolist(), "v_after": np.round(vB).tolist()})

    # merge events closer than 5 samples: keep a hit over a bounce, else the first
    merged = []
    for ev in events:
        if merged and ev["stretch"] == merged[-1]["stretch"] and ev["index"] - merged[-1]["index"] < 5:
            if merged[-1]["kind"] != "hit" and ev["kind"] == "hit":
                merged[-1] = ev
            continue
        merged.append(ev)
    events = merged

    Path(prefix + "_events.json").write_text(json.dumps(events, indent=1))
    kinds = [e["kind"] for e in events]
    print(f"{len(events)} events: {kinds.count('hit')} hits, {kinds.count('bounce')} bounces, {kinds.count('edge')} unanchored edges")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(26, 9), sharex=True)
    ax[0].plot(t, xy[:, 1], ".", ms=3)
    ax[0].invert_yaxis()
    ax[0].set_ylabel("image y (px)")
    ax[1].plot(t, xy[:, 0], ".", ms=3)
    ax[1].set_ylabel("image x (px)")
    col = {"hit": "red", "bounce": "green", "edge": "gray"}
    for e in events:
        for x in ax:
            x.axvline(e["t"], color=col[e["kind"]], lw=0.9)
        ax[0].text(e["t"], 170, {"hit": f"H{e['player']}", "bounce": "B", "edge": "?"}[e["kind"]], fontsize=8, ha="center")
    for x in ax:
        x.grid(alpha=0.3)
        x.set_xticks(np.arange(0, 34, 1))
    plt.tight_layout()
    plt.savefig(prefix + "_events.png", dpi=70)


if __name__ == "__main__":
    main()
