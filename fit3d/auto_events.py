"""Automatic hit and bounce labelling from sound plus the 2D ball track.

Hits    A paddle hit is a sharp, loud onset in the 1.5-9 kHz band. Each strong onset is
        kept as a hit if the ball's image velocity changes across it (or the ball is
        hidden there). The hitter is the player whose wrist is nearest in the image.
Bounces Between two consecutive hits, at most one bounce: the visual break where the ball
        goes from moving down the screen to moving up while keeping its sideways motion,
        landing on the far side of the net from the last hitter. A weaker onset nearby
        refines its time.

Audio lags the picture by a fixed delay (sound travel plus broadcast offset), measured
once per match; it is a parameter here.

Usage: python auto_events.py <video> <segments.json> <players_tracks.npz> <calibration.json> <t_start> <t_end> <out.json>
  t_start/t_end are on the 50 Hz unique-frame clock of the shot (seconds).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import NET_Y, R_BALL, Camera  # noqa: E402

AV_DELAY = 0.087        # s: audio onset arrives this long after the event in the picture
HIT_STRENGTH = 1.6      # log-energy rise over 10 ms
BOUNCE_STRENGTH = 0.6
SR = 48000


def onsets(video, t0, t1):
    """Onset times (video seconds) and strengths in [t0, t1]."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0 - 0.5:.3f}", "-t", f"{t1 - t0 + 1.5:.3f}", "-i", video,
                          "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    y = sosfiltfilt(butter(4, [1500, 9000], btype="band", fs=SR, output="sos"), x)
    hop = 96
    env = np.sqrt(np.convolve(y ** 2, np.ones(240) / 240, "same"))[::hop]
    le = np.log(env + 1e-5)
    on = np.r_[np.zeros(5), np.maximum(le[5:] - le[:-5], 0)]
    pk, _ = find_peaks(on, height=BOUNCE_STRENGTH, distance=30)
    return t0 - 0.5 + pk * hop / SR, on[pk]


def velocity(t, xy, lo, hi):
    """Mean image velocity (px/s) from a line fit to observed samples with lo <= t <= hi."""
    m = (t >= lo) & (t <= hi) & ~np.isnan(xy[:, 0])
    if m.sum() < 3:
        return None
    A = np.stack([np.ones(m.sum()), t[m] - t[m][0]], 1)
    c, *_ = np.linalg.lstsq(A, xy[m], rcond=None)
    return c[1]


def nearest_wrist(kp, xy, i):
    seen = np.flatnonzero(~np.isnan(xy[:, 0]))
    j = seen[np.argmin(np.abs(seen - i))]
    best = (np.inf, -1)
    for pl in range(4):
        K = kp[j, pl]
        if np.isnan(K[0, 0]):
            continue
        torso = max(np.linalg.norm(K[[5, 6], :2].mean(0) - K[[11, 12], :2].mean(0)), 1.0)
        d = min(np.linalg.norm(K[w, :2] - xy[j]) for w in (9, 10)) / torso
        if d < best[0]:
            best = (float(d), pl)
    return best


def main():
    video, seg_path, players_path, calib_path = sys.argv[1:5]
    ts, te, out = float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    seg = json.loads(Path(seg_path).read_text())
    cam = Camera(json.loads(Path(calib_path).read_text()))
    t = np.array(seg["t"])
    xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
    keep = np.array(seg["keep"])
    kp = np.load(players_path)["kpts"][keep]
    vid_t = (seg["start_frame"] + keep) / 60.0                 # video time of each unique sample

    on_t, on_s = onsets(video, vid_t[0], vid_t[-1])
    on_u = np.interp(on_t - AV_DELAY, vid_t, t)                # onset time on the unique clock, delay removed

    # ---- visual breaks with curvature-aware velocity change (from the piecewise-quadratic arcs)
    def arc_vel(a):
        m = slice(a["i"], a["j"] + 1)
        ok = ~np.isnan(xy[m, 0])
        tt = t[m][ok] - t[a["i"]]
        A = np.stack([np.ones_like(tt), tt, tt * tt], 1)
        c, *_ = np.linalg.lstsq(A, xy[m][ok], rcond=None)
        T = t[a["j"]] - t[a["i"]]
        return c[1], c[1] + 2 * c[2] * T
    vbreaks = {}                                      # sample index -> (v_before, v_after)
    arcs = seg["arcs"]
    for a0, a1 in zip(arcs[:-1], arcs[1:]):
        if a0["stretch"] == a1["stretch"] and a0["j"] == a1["i"]:
            vbreaks[a0["j"]] = (arc_vel(a0)[1], arc_vel(a1)[0])
    bidx = np.array(sorted(vbreaks))

    def hidden(tu, w=0.06):
        m = (t >= tu - w) & (t <= tu + w)
        return m.any() and np.isnan(xy[m, 0]).mean() > 0.6

    # ---- hits: an onset supported by a visual break (or a hidden ball) and a wrist nearby.
    # score = loudness + size of the velocity change - distance to the nearest wrist
    hits = []
    for tu, s in zip(on_u, on_s):
        if s < 1.2 or not (ts - 0.05 <= tu <= te + 0.05):
            continue
        near = bidx[np.abs(t[bidx] - tu) <= 0.10] if len(bidx) else []
        i = int(np.argmin(np.abs(t - tu)))
        d, pl = nearest_wrist(kp, xy, i)
        if len(near):
            b = int(near[np.argmin(np.abs(t[near] - tu))])
            vb, va = vbreaks[b]
            dv = float(np.linalg.norm(va - vb))
            bounce_like = vb[1] > 60 and va[1] < vb[1] - 150 and (vb[0] * va[0] > 0 or min(abs(vb[0]), abs(va[0])) < 120)
            if bounce_like and d > 0.9:
                continue                               # the sound of a bounce
        elif hidden(tu):
            dv = 600.0                                 # contact hidden by the hitter: neutral evidence
        else:
            continue                                   # loud, but the ball carried straight on
        if d > 1.6:
            continue
        score = s + dv / 500.0 - d
        if score < 1.6:
            continue
        hits.append({"kind": "hit", "t": float(tu), "index": i, "player": pl, "strength": float(s), "wrist": round(d, 2),
                     "dv": round(dv), "score": round(score, 2)})
    # one hit per 0.3 s: keep the best-scoring
    hits.sort(key=lambda h: h["t"])
    merged = []
    for h in hits:
        if merged and h["t"] - merged[-1]["t"] < 0.3:
            if h["score"] > merged[-1]["score"]:
                merged[-1] = h
        else:
            merged.append(h)
    hits = merged

    # ---- rules of play: consecutive hits come from opposite sides. Where two in a row are
    # on the same side, one is a false alarm (a loud bounce, a shoe); drop the quieter.
    changed = True
    while changed:
        changed = False
        for a, b in zip(hits[:-1], hits[1:]):
            if (a["player"] >= 2) == (b["player"] >= 2):
                hits.remove(a if a["score"] < b["score"] else b)
                changed = True
                break

    def vel2(lo, mid, hi):
        for w in (0.14, 0.22, 0.30):
            vb, va = velocity(t, xy, mid - w, mid), velocity(t, xy, mid, mid + w)
            if vb is not None and va is not None:
                return vb, va
        return None, None

    # ---- bounces: at most one between consecutive hits
    breaks = sorted({a["i"] for a in seg["arcs"]} | {a["j"] for a in seg["arcs"]})
    events = []
    for k, h in enumerate(hits):
        events.append(h)
        t_next = hits[k + 1]["t"] if k + 1 < len(hits) else min(te, h["t"] + 2.5)
        best = None
        for b in breaks:
            tb = t[b]
            if not (h["t"] + 0.12 < tb < t_next - 0.075) or np.isnan(xy[b, 0]):   # closer than 75 ms is the hit itself
                continue
            vb, va = vel2(None, tb, None)
            if vb is None or va is None:
                continue
            down_up = vb[1] > 60 and va[1] < vb[1] - 150
            side_kept = vb[0] * va[0] > 0 or min(abs(vb[0]), abs(va[0])) < 120
            floor = cam.on_plane(xy[b], R_BALL)
            hitter_far = h["player"] >= 2
            landed_far = floor[1] > NET_Y
            if not (down_up and side_kept) or landed_far == hitter_far:
                continue

            if not (-4 <= floor[0] <= 24 and -6 <= floor[1] <= 50):
                continue
            score = vb[1] - va[1]
            if best is None or score > best[0]:
                best = (score, b, floor)
        if best is not None:
            _, b, floor = best
            tb = float(t[b])
            near = np.flatnonzero((np.abs(on_u - tb) < 0.05) & (on_u > h["t"] + 0.1) & (on_u < t_next - 0.04))
            heard = len(near) > 0
            if heard:
                tb = float(on_u[near[np.argmax(on_s[near])]])
            events.append({"kind": "bounce", "t": tb, "index": int(b), "heard": bool(heard),
                           "floor_ft": np.round(floor[:2], 1).tolist()})
    events = [e for e in events if ts - 0.05 <= e["t"] <= te + 0.05]
    Path(out).write_text(json.dumps({"_note": "automatic: auto_events.py", "av_delay_s": AV_DELAY, "events": events}, indent=1))
    kinds = [e["kind"] for e in events]
    print(f"{len(events)} events: {kinds.count('hit')} hits, {kinds.count('bounce')} bounces")
    for e in events:
        if e["kind"] == "hit":
            print(f"  hit    t={e['t']:6.3f} p{e['player']} strength {e['strength']:.2f} wrist {e['wrist']} dv {e['dv']} score {e['score']}")
        else:
            print(f"  bounce t={e['t']:6.3f} floor {e['floor_ft']} heard={e['heard']}")


if __name__ == "__main__":
    main()
