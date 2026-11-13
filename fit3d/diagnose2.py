"""Two more hypotheses per flight:
  E  time shift   anchored fit, but the flight's clock may shift (hit time wrong)
  F  trim ends    anchored fit ignoring the 3 samples nearest each end (ball merged with paddle/hand at contact)
  G  both
Usage: python diagnose2.py <segments.json> <fit.json> <calibration.json>
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import G, Camera
from diagnose import sim, at

seg = json.loads(Path(sys.argv[1]).read_text()); fit = json.loads(Path(sys.argv[2]).read_text())
cam = Camera(json.loads(Path(sys.argv[3]).read_text()))
t = np.array(seg["t"]); xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
print(" t0    kind  dur  n | A-anch  E-shift (dt ms)  F-trim  G-both (dt ms)")
rows = []
for a in fit["arcs"]:
    t0, T = a["t0"], a["duration_s"]
    idx = np.flatnonzero((t >= t0 - 1e-6) & (t <= t0 + T + 1e-6) & ~np.isnan(xy[:, 0]))
    if len(idx) < 12: continue
    tt = t[idx] - t0; obs = xy[idx]
    s, e = np.array(a["start_ft"]), np.array(a["end_ft"])
    v = (e - s) / T; v[2] += 0.5 * G * T
    def run(shift, trim):
        m = np.ones(len(idx), bool)
        if trim: m[:3] = False; m[-3:] = False
        def res(x):
            dt = x[6] if shift else 0.0
            ts, P, _ = sim(x[:3], x[3:6], T + 0.3)
            tq = np.clip(tt[m] - dt, 0, T + 0.3)
            r = [((cam.project(at(ts, P, tq)) - obs[m]) / 2.0).ravel(), (x[:3] - s) / 0.3,
                 (at(ts, P, np.array([max(T - dt, 0.05)]))[0] - e) / 0.3]
            if shift: r.append([dt / 0.1])
            return np.concatenate(r)
        sol = least_squares(res, np.r_[s, v, 0.0] if shift else np.r_[s, v])
        d = sol.fun[:2 * m.sum()].reshape(-1, 2) * 2
        return float(np.sqrt(np.mean((d ** 2).sum(1)))), (sol.x[6] * 1000 if shift else 0.0)
    A, _ = run(False, False); E, dE = run(True, False); F, _ = run(False, True); Gb, dG = run(True, True)
    print(f"{t0:6.2f} {a['from'][0]}>{a['to'][0]} {T:5.2f} {len(idx):3d} | {A:6.2f}{'*' if A > 6 else ' '} {E:6.2f} ({dE:+5.0f})   {F:6.2f}  {Gb:6.2f} ({dG:+5.0f})")
    rows.append((A, E, F, Gb, dG))
R = np.array(rows); bad = R[:, 0] > 6
for n, k in (("E time shift", 1), ("F trim ends", 2), ("G both", 3)):
    print(f"  {n:<13} brings {int((R[bad, k] < 4).sum())} of {bad.sum()} bad flights under 4 px; median bad {np.median(R[bad, k]):.2f}, good {np.median(R[~bad, k]):.2f}")
print("  shifts on bad flights (ms):", np.round(R[bad, 4]).tolist(), " on good:", np.round(R[~bad, 4]).tolist())
