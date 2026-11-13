"""Diagnostic: fit each 2D arc on its own with a free 3D flight (6 unknowns) and see
whether depth is determined: print endpoint heights, depths and speed."""
import json, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, 'fit3d')
from physics import Camera, simulate, G

seg = json.load(open('fit3d/out/rally_267_segments.json'))
cam = Camera(json.load(open('calib/out/t270_calibration.json')))
t = np.array(seg['t']); xy = np.array([[np.nan if v is None else v for v in p] for p in seg['xy']])
DT = 1 / 50

def free_fit(i, j):
    obs = xy[i:j + 1]; ok = ~np.isnan(obs[:, 0]); n = j - i + 1; T = (n - 1) * DT
    def res(x):
        P, _ = simulate(x[:3], x[3:], n, DT)
        return ((cam.project(P) - obs)[ok]).ravel()
    best = None
    first, last = obs[ok][0], obs[ok][-1]
    k0, k1 = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
    for y0 in np.arange(-6, 51, 4.0):
        for y1 in np.arange(-6, 51, 4.0):
            p0, p1 = cam.at_depth(first, y0), cam.at_depth(last, y1)
            if p0[2] < -0.5 or p1[2] < -0.5 or p0[2] > 25 or p1[2] > 25: continue
            Tk = max((k1 - k0) * DT, DT)
            v0 = (p1 - p0) / Tk; v0[2] += 0.5 * G * Tk
            pstart = p0 - v0 * k0 * DT; pstart[2] -= 0.5 * G * (k0 * DT) ** 2 * 0  # k0 is 0 by construction
            c = np.sum(res(np.r_[p0, v0]) ** 2)
            if best is None or c < best[0]: best = (c, np.r_[p0, v0])
    sol = least_squares(res, best[1], loss='soft_l1', f_scale=2.0, x_scale=np.r_[1, 5, 1, 10, 20, 10])
    P, V = simulate(sol.x[:3], sol.x[3:], n, DT)
    r = np.linalg.norm((cam.project(P) - obs)[ok], axis=1)
    # conditioning: std of start depth from the Jacobian
    J = sol.jac; cov = np.linalg.pinv(J.T @ J) * (r ** 2).sum() * 2 / max(1, 2 * ok.sum() - 6)
    return P, V, float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(abs(cov[1, 1])))

print(' arc  i    j   n  rms  | start (x,y,z)        end (x,y,z)          | speed mph  sd(y0) ft')
for a in seg['arcs']:
    if a['n_obs'] < 8: continue
    P, V, rms, sd = free_fit(a['i'], a['j'])
    print(f"{a['i']:5d} {a['j']:5d} {a['n_obs']:3d} {rms:5.2f} | {P[0,0]:6.1f} {P[0,1]:6.1f} {P[0,2]:5.1f}   {P[-1,0]:6.1f} {P[-1,1]:6.1f} {P[-1,2]:5.1f} | {np.linalg.norm(V[0])*0.6818:5.1f}  {sd:6.2f}")
