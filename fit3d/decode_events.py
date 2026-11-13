"""Hits and bounces for a whole shot, by decoding one best sequence of events.

Candidates are every place the ball's image track breaks (segment.py arcs) plus every
clear audio onset. Each candidate can be a near hit, a far hit, a bounce or nothing. A
dynamic programme picks the labelling with the best total score, where the score has
  - per-candidate evidence: loudness, change of ball velocity, wrist distance, the
    down-then-up shape of a bounce;
  - between consecutive events, the order of play: a near hit is followed by a bounce
    on the far side or a far hit, the ball travels up the screen after a near hit and
    down after a far hit, and so on. Breaking the order costs score but is allowed
    (dead-ball bouncing before a serve or after the point does not follow it).

Unlike auto_events.py this covers every part of the shot, needs no hit before a bounce,
and decides the hitter's side jointly with the rest of the rally.

Usage: python decode_events.py <video> <segments.json> <players_tracks.npz> <calibration.json> <out.json>
Output events carry "f": frame within the shot at 60 fps, and "t": 50 Hz unique clock.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_events import AV_DELAY, onsets, velocity  # noqa: E402
from physics import NET_Y, R_BALL, Camera  # noqa: E402

P = {                      # weights; override any with env DECODE_<NAME>
    "ONSET_MIN": 1.2,      # audio onsets weaker than this are not candidates on their own
    "HIT_BIAS": 2.6,       # evidence a hit needs before it beats "nothing"
    "HIT_AUDIO": 1.0,
    "HIT_DV": 1.0,         # per 500 px/s of velocity change, capped at 1500
    "HIT_WRIST": 1.0,      # per torso length from the nearest wrist on that side, capped at 3
    "BOUNCE_BIAS": 0.3,
    "BOUNCE_AUDIO": 0.3,
    "BOUNCE_VEE": 1.0,     # per 1000 px/s of vertical reversal, capped at 1500
    "ORDER": 2.5,          # cost of breaking the order of play
    "DIRECTION": 2.5,      # cost of the ball travelling the wrong way after a hit
    "BB": 1.0,             # cost of a second bounce on the same side (dead ball)
    "FAULT": 1.0,          # cost of a hit followed by a bounce on the hitter's own side (into the net)
    "MIN_HIT_GAP": 10,     # frames between two hits
}
for k in P:
    if f"DECODE_{k}" in os.environ:
        P[k] = float(os.environ[f"DECODE_{k}"])


def vel(t, xy, lo, hi):
    """Image velocity (px/s) from a line through the seen samples in [lo, hi]; two are enough."""
    m = (t >= lo) & (t <= hi) & ~np.isnan(xy[:, 0])
    if m.sum() < 2:
        return None
    if m.sum() == 2:
        i, j = np.flatnonzero(m)
        return (xy[j] - xy[i]) / (t[j] - t[i])
    A = np.stack([np.ones(m.sum()), t[m] - t[m][0]], 1)
    return np.linalg.lstsq(A, xy[m], rcond=None)[0][1]


def candidates(seg, kp, cam, video):
    t = np.array(seg["t"])
    keep = np.array(seg["keep"], float)
    xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
    seen = np.flatnonzero(~np.isnan(xy[:, 0]))
    vid_t = (seg["start_frame"] + keep) / 60.0
    on_t, on_s = onsets(video, vid_t[0], vid_t[-1])
    on_f = (on_t - AV_DELAY) * 60.0 - seg["start_frame"]
    ok = (on_f >= -2) & (on_f <= keep[-1] + 2)
    on_f, on_s = on_f[ok], on_s[ok]

    brk = sorted({a["i"] for a in seg["arcs"]} | {a["j"] for a in seg["arcs"]})
    frames = [(float(keep[b]), True) for b in brk]
    for f, s in zip(on_f, on_s):
        if s >= P["ONSET_MIN"] and all(abs(f - g) > 2 for g, _ in frames):
            frames.append((float(f), False))
    frames.sort()

    out = []
    for f, is_break in frames:
        tu = float(np.interp(f, keep, t))
        near_on = np.flatnonzero(np.abs(on_f - f) <= 2.5)
        a = float(on_s[near_on].max()) if len(near_on) else 0.0
        f_out = float(on_f[near_on[np.argmax(on_s[near_on])]]) if len(near_on) else f
        # velocities either side, each measured inside one flight: never across a neighbouring candidate
        prev = max([np.interp(g, keep, t) for g, _ in frames if g < f - 2.5], default=-1e9)
        nxt = min([np.interp(g, keep, t) for g, _ in frames if g > f + 2.5], default=1e9)
        vb = va = None
        for w in (0.14, 0.22, 0.30):
            vb = vel(t, xy, max(tu - w, prev - 1e-6), tu + 1e-6) if vb is None else vb
            va = vel(t, xy, tu - 1e-6, min(tu + w, nxt + 1e-6)) if va is None else va
        i = int(np.argmin(np.abs(t - tu)))
        pos, d_side = None, [1.5, 1.5]
        if len(seen):
            j = int(seen[np.argmin(np.abs(seen - i))])
            if abs(j - i) <= 10:
                pos = xy[j]
                for side in (0, 1):
                    best = 3.0
                    for pl in (2 * side, 2 * side + 1):
                        K = kp[j, pl]
                        if np.isnan(K[0, 0]):
                            continue
                        torso = max(np.linalg.norm(K[[5, 6], :2].mean(0) - K[[11, 12], :2].mean(0)), 1.0)
                        best = min(best, min(np.linalg.norm(K[w, :2] - pos) for w in (9, 10)) / torso)
                    d_side[side] = float(best)
        c = {"f": f_out, "f_cand": f, "t": tu, "break": is_break, "audio": a, "pos": None if pos is None else pos.tolist(),
             "d_near": round(d_side[0], 2), "d_far": round(d_side[1], 2),
             "vb": None if vb is None else vb.tolist(), "va": None if va is None else va.tolist()}
        # contact hidden by a body: neutral evidence. No ball anywhere near in time: none.
        dv = (600.0 if pos is not None else 0.0) if vb is None or va is None else float(np.linalg.norm(np.array(va) - np.array(vb)))
        c["dv"] = round(dv)
        base = P["HIT_AUDIO"] * min(a, 4.0) + P["HIT_DV"] * min(dv, 1500.0) / 500.0 - P["HIT_BIAS"]
        c["s_near"] = base - P["HIT_WRIST"] * min(d_side[0], 3.0)
        c["s_far"] = base - P["HIT_WRIST"] * min(d_side[1], 3.0)
        c["s_bounce"] = None
        visible = not np.isnan(xy[i, 0]) or (pos is not None and abs(j - i) <= 2)
        if vb is not None and va is not None and visible:
            vee = vb[1] > 40 and va[1] < vb[1] - 120
            side_kept = vb[0] * va[0] > 0 or min(abs(vb[0]), abs(va[0])) < 120
            floor = cam.on_plane(np.array(pos), R_BALL)
            if vee and side_kept and -4 <= floor[0] <= 24 and -6 <= floor[1] <= 50:
                c["s_bounce"] = P["BOUNCE_BIAS"] + P["BOUNCE_AUDIO"] * min(a, 3.0) + P["BOUNCE_VEE"] * min(vb[1] - va[1], 1500.0) / 1000.0
                c["bounce_far"] = bool(floor[1] > NET_Y)
                c["floor_ft"] = np.round(floor[:2], 1).tolist()
        out.append(c)
    return out


def options(c):
    o = [("near", c["s_near"]), ("far", c["s_far"])]
    if c["s_bounce"] is not None:
        o.append(("bounce_far" if c["bounce_far"] else "bounce_near", c["s_bounce"]))
    return o


def pair_cost(a, la, b, lb):
    """Cost of event (b, lb) directly following event (a, la)."""
    gap = b["f"] - a["f"]
    if gap < 2:
        return np.inf
    hit_a, hit_b = la in ("near", "far"), lb in ("near", "far")
    if hit_a and hit_b and gap < P["MIN_HIT_GAP"]:
        return np.inf
    legal = {"near": ("far", "bounce_far"), "far": ("near", "bounce_near"), "bounce_far": ("far",), "bounce_near": ("near",)}
    cost = 0.0
    if lb not in legal[la]:
        if not hit_a and la == lb:
            cost += P["BB"]
        elif hit_a and lb == "bounce_" + la:
            cost += P["FAULT"]
        else:
            cost += P["ORDER"]
    if hit_a and a["pos"] is not None and b["pos"] is not None:
        dy = b["pos"][1] - a["pos"][1]
        if (la == "near" and dy > 30) or (la == "far" and dy < -30):
            cost += P["DIRECTION"]
    return cost


def decode(cands):
    n = len(cands)
    opts = [options(c) for c in cands]
    best = [[s for _, s in o] for o in opts]           # best total ending at (i, k)
    back = [[None] * len(o) for o in opts]
    for j in range(n):
        for kj, (lj, sj) in enumerate(opts[j]):
            for i in range(j):
                for ki, (li, _) in enumerate(opts[i]):
                    v = best[i][ki] - pair_cost(cands[i], li, cands[j], lj) + sj
                    if v > best[j][kj]:
                        best[j][kj], back[j][kj] = v, (i, ki)
    end, top = None, 0.0
    for j in range(n):
        for kj in range(len(opts[j])):
            if best[j][kj] > top:
                top, end = best[j][kj], (j, kj)
    events = []
    while end is not None:
        j, kj = end
        c, (lab, s) = cands[j], opts[j][kj]
        e = {"kind": "bounce" if lab.startswith("bounce") else "hit", "f": round(c["f"], 1), "t": c["t"], "score": round(s, 2),
             "audio": round(c["audio"], 2), "dv": c["dv"], "d_near": c["d_near"], "d_far": c["d_far"]}
        if e["kind"] == "hit":
            e["side"] = lab
        else:
            e["side"] = "far" if c["bounce_far"] else "near"
            e["floor_ft"] = c["floor_ft"]
        events.append(e)
        end = back[j][kj]
    return events[::-1]


def run(video, seg_path, players_path, calib_path):
    seg = json.loads(Path(seg_path).read_text())
    cam = Camera(json.loads(Path(calib_path).read_text()))
    kp = np.load(players_path)["kpts"][np.array(seg["keep"])]
    cands = candidates(seg, kp, cam, video)
    return decode(cands), cands


if __name__ == "__main__":
    events, cands = run(*sys.argv[1:5])
    Path(sys.argv[5]).write_text(json.dumps({"_note": "automatic: decode_events.py", "params": P, "events": events, "candidates": cands}, indent=1))
    hits = sum(e["kind"] == "hit" for e in events)
    print(f"{len(events)} events: {hits} hits, {len(events) - hits} bounces from {len(cands)} candidates")
