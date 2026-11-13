"""Why do some flights fit badly? Refit each flight on its own under competing hypotheses.

  A  anchored   ends held near the global fit's event positions (0.3 ft); gravity + drag
  B  free ends  same physics, no anchors                 -> are the anchors (events, hitter positions) wrong?
  C  + accel    anchored, plus a constant extra acceleration -> is unmodelled force (spin) the issue?
  D  + bounce   anchored, with one floor contact inside  -> is a bounce missing?
Also reports how concentrated the error is (share of squared error in the worst 10% of
samples, and which third of the flight it sits in) -> are a few bad detections the issue?

Usage: python diagnose.py <segments.json> <fit.json> <calibration.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import G, K_DRAG, R_BALL, Camera  # noqa: E402

H = 0.005


def sim(p0, v0, T, extra=(0, 0, 0)):
    n = max(2, int(np.ceil(T / H)) + 2)
    P = np.empty((n, 3))
    p, v = np.array(p0, float), np.array(v0, float)
    ex = np.array(extra, float)

    def acc(v):
        a = -K_DRAG * np.sqrt(v @ v) * v + ex
        a[2] -= G
        return a
    V = np.empty((n, 3))
    for i in range(n):
        P[i], V[i] = p, v
        k1 = acc(v); k2 = acc(v + 0.5 * H * k1); k3 = acc(v + 0.5 * H * k2); k4 = acc(v + H * k3)
        p = p + H * (v + H / 6 * (k1 + k2 + k3))
        v = v + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.arange(n) * H, P, V


def at(ts, P, t):
    return np.stack([np.interp(t, ts, P[:, k]) for k in range(3)], -1)


def main():
    seg = json.loads(Path(sys.argv[1]).read_text())
    fit = json.loads(Path(sys.argv[2]).read_text())
    cam = Camera(json.loads(Path(sys.argv[3]).read_text()))
    t = np.array(seg["t"])
    xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
    print(" t0    kind  dur  n | rms: A-anch B-free C-accel D-bounce | extra accel (ft/s2) x,y,z | top10% share  worst third | B end shift ft")
    rows = []
    for a in fit["arcs"]:
        t0, T = a["t0"], a["duration_s"]
        idx = np.flatnonzero((t >= t0 - 1e-6) & (t <= t0 + T + 1e-6) & ~np.isnan(xy[:, 0]))
        if len(idx) < 8:
            continue
        tt = np.clip(t[idx] - t0, 0, T)
        obs = xy[idx]
        s, e = np.array(a["start_ft"]), np.array(a["end_ft"])
        v_init = (e - s) / T
        v_init[2] += 0.5 * G * T

        def res(x, anchored, with_acc):
            ex = x[6:9] if with_acc else (0, 0, 0)
            ts, P, _ = sim(x[:3], x[3:6], T, ex)
            r = [((cam.project(at(ts, P, tt)) - obs) / 2.0).ravel()]
            if anchored:
                r.append((x[:3] - s) / 0.3)
                r.append((at(ts, P, np.array([T]))[0] - e) / 0.3)
            if with_acc:
                r.append(x[6:9] / 60.0)                # mild shrinkage so it is not used for nothing
            return np.concatenate(r)

        def rms(x, with_acc=False):
            ts, P, _ = sim(x[:3], x[3:6], T, x[6:9] if with_acc else (0, 0, 0))
            d = np.linalg.norm(cam.project(at(ts, P, tt)) - obs, axis=1)
            return float(np.sqrt(np.mean(d ** 2))), d, at(ts, P, np.array([T]))[0]

        x0 = np.r_[s, v_init]
        A = least_squares(res, x0, args=(True, False)).x
        B = least_squares(res, A, args=(False, False)).x
        C = least_squares(res, np.r_[A, 0, 0, 0], args=(True, True)).x
        rA, dA, _ = rms(A)
        rB, _, endB = rms(B)
        rC, _, _ = rms(C, True)

        # D: one bounce inside (only meaningful when the flight is long enough)
        rD = np.nan
        if T > 0.5:
            def resD(x):
                tb = np.clip(x[6], 0.15, 0.85) * T
                ts1, P1, V1 = sim(x[:3], x[3:6], tb)
                pb = at(ts1, P1, np.array([tb]))[0]
                ts2, P2, _ = sim(pb, x[7:10], T - tb)
                first = tt <= tb
                pr = np.where(first[:, None], at(ts1, P1, np.minimum(tt, tb)), at(ts2, P2, np.maximum(tt - tb, 0)))
                return np.concatenate([((cam.project(pr) - obs) / 2.0).ravel(), (x[:3] - s) / 0.3,
                                       (at(ts2, P2, np.array([T - tb]))[0] - e) / 0.3, [(pb[2] - R_BALL) / 0.05]])
            best = None
            for f in (0.3, 0.5, 0.7):
                mid = s + (e - s) * f
                mid[2] = R_BALL
                v1 = (mid - s) / (f * T); v1[2] += 0.5 * G * f * T
                v2 = (e - mid) / ((1 - f) * T); v2[2] += 0.5 * G * (1 - f) * T
                sol = least_squares(resD, np.r_[s, v1, f, v2])
                if best is None or sol.cost < best.cost:
                    best = sol
            rr = best.fun[:2 * len(idx)].reshape(-1, 2) * 2.0
            rD = float(np.sqrt(np.mean((rr ** 2).sum(1))))

        sq = np.sort(dA ** 2)[::-1]
        top = float(sq[:max(1, len(sq) // 10)].sum() / sq.sum())
        thirds = [float((dA[(tt >= T * k / 3) & (tt <= T * (k + 1) / 3)] ** 2).sum()) for k in range(3)]
        worst = ["start", "middle", "end"][int(np.argmax(thirds))]
        shift = float(np.linalg.norm(endB - e))
        kind = a["from"][0] + ">" + a["to"][0]
        flag = "*" if (rA > 6) else " "
        print(f"{t0:6.2f} {kind} {T:5.2f} {len(idx):3d} | {rA:6.2f}{flag} {rB:6.2f} {rC:6.2f} {rD:8.2f} | {C[6]:6.1f} {C[7]:6.1f} {C[8]:6.1f} | {top:5.2f}  {worst:6s} | {shift:6.1f}")
        rows.append((rA, rB, rC, rD, top))
    R = np.array(rows)
    bad = R[:, 0] > 6
    print(f"\nflights {len(R)}; anchored rms > 6 px: {bad.sum()}")
    for name, k in (("B free ends", 1), ("C extra accel", 2), ("D one bounce", 3)):
        fixed = np.nansum((R[bad, k] < 4))
        print(f"  {name:<14} brings {int(fixed)} of {bad.sum()} bad flights under 4 px; median on bad {np.nanmedian(R[bad, k]):.2f}, on good {np.nanmedian(R[~bad, k]):.2f}")
    print(f"  top-10% error share: bad {np.median(R[bad, 4]):.2f}, good {np.median(R[~bad, 4]):.2f}")


if __name__ == "__main__":
    main()
