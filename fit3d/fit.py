"""Anchored 3D fit of a ball track: one drag-model flight per arc between labelled events.

Unknowns
  each event   3D position (a bounce sits on the floor, so only x and y) and a small time offset
  each arc     launch velocity
Residuals
  reprojection of every observed sample (2 px noise scale, robust)
  each flight must arrive at the next event's position
  a hit must be within reach of the hitter's ground position, at a hittable height
  the ball stays above the floor
  weak bounce physics: vertical speed reverses with restitution about 0.66

Not used in the fit, so available as checks: net clearance, which side each bounce lands
on, hit heights, speeds.

Usage: python fit.py <segments.json> <events.json> <players_tracks.npz> <calibration.json> <out_prefix>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics import G, K_DRAG, NET_Y, R_BALL, Camera  # noqa: E402

DT = 1 / 50
H = 0.005                 # integration step (s)
SIG_PX = 2.0
SIG_JOIN = 0.08           # ft
import os
TRIM = float(os.environ.get("TRIM", 0.06))   # s
REACH = float(os.environ.get("REACH", 4.5))               # ft, horizontal distance hitter's feet -> ball at contact
E_REST = 0.66
PADDLE_REACH = float(os.environ.get("PADDLE_REACH", 1.1))   # ft, palm to the far end of the paddle face
PADDLE_SIG = float(os.environ.get("PADDLE_SIG", 0.15))      # ft, how firmly contact is held within that reach
K_TAN = float(os.environ.get("K_TAN", 0.57))   # share of horizontal speed kept through a bounce (weak prior); sim/RESULTS.md measures about 0.57
MPH = 0.6818


def flight(p0, v0, T):
    """Integrate from p0, v0 for T seconds on a fixed grid. Returns times, positions, velocities."""
    n = max(2, int(np.ceil(T / H)) + 2)
    P = np.empty((n, 3))
    V = np.empty((n, 3))
    p, v = np.array(p0, float), np.array(v0, float)

    def acc(v):
        a = -K_DRAG * np.sqrt(v @ v) * v
        a[2] -= G
        return a
    for i in range(n):
        P[i], V[i] = p, v
        k1 = acc(v)
        k2 = acc(v + 0.5 * H * k1)
        k3 = acc(v + 0.5 * H * k2)
        k4 = acc(v + H * k3)
        p = p + H * (v + H / 6 * (k1 + k2 + k3))
        v = v + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.arange(n) * H, P, V


def interp(ts, P, t):
    return np.stack([np.interp(t, ts, P[:, k]) for k in range(3)], -1)


class Problem:
    def __init__(self, seg, events, players, cam):
        self.cam = cam
        self.t = np.array(seg["t"])
        self.xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
        keep = np.array(seg["keep"])
        self.ground = players["smooth"][keep]            # (n, 4, 2) player ground positions
        # optional: the paddle hand from the 3D bodies (bodies/ground.py output), a much tighter anchor for hits
        self.grip = None
        if os.environ.get("HANDS"):
            g = np.load(os.environ["HANDS"])
            kp = g["kp_world"]                              # frames, players, 70, 3 (court feet)
            self.grip = (kp[:, :, 41] + kp[:, :, 28] + kp[:, :, 40]) / 3      # right wrist and the index and pinky knuckles
            self.grip_frames = g["frames"].astype(float)   # shot frames at 60 fps
            self.keep_f = keep.astype(float)
        self.ev = events
        self.nE = len(events)
        # parameter layout
        self.pos_slice, k = [], 0
        for e in events:
            d = 2 if e["kind"] == "bounce" else 3
            self.pos_slice.append(slice(k, k + d))
            k += d
        self.tau0 = k
        k += self.nE
        self.vel0 = k
        self.n = k + 3 * (self.nE - 1)
        # observations per arc
        # event times: an explicit "t" (e.g. from audio) wins over the sample index
        self.et = np.array([e.get("t", self.t[e["index"]]) for e in events])
        self.obs = []
        for a in range(self.nE - 1):
            idx = np.flatnonzero((self.t >= self.et[a] - 1e-6) & (self.t <= self.et[a + 1] + 1e-6) & ~np.isnan(self.xy[:, 0]))
            # detections right at contact are unreliable (ball merged with paddle or hand, launch blur):
            # ignore samples within TRIM of either event, unless that leaves too few
            core = idx[(self.t[idx] >= self.et[a] + TRIM) & (self.t[idx] <= self.et[a + 1] - TRIM)]
            self.obs.append(core if len(core) >= 4 else idx)

    def grip_at(self, t_unique, player):
        """Palm position of a player at a time on the 50 Hz clock, or None outside the bodies' span."""
        if self.grip is None:
            return None
        f = float(np.interp(t_unique, self.t, self.keep_f))
        if f < self.grip_frames[0] or f > self.grip_frames[-1]:
            return None
        return np.array([np.interp(f, self.grip_frames, self.grip[:, player, c]) for c in range(3)])

    def node(self, x, k):
        s = self.pos_slice[k]
        return np.r_[x[s], R_BALL] if self.ev[k]["kind"] == "bounce" else x[s]

    def node_time(self, x, k):
        return self.et[k] + x[self.tau0 + k]

    def arc(self, x, a):
        p0 = self.node(x, a)
        v0 = x[self.vel0 + 3 * a: self.vel0 + 3 * a + 3]
        t0, t1 = self.node_time(x, a), self.node_time(x, a + 1)
        ts, P, V = flight(p0, v0, max(t1 - t0, 2 * H))
        return t0, t1, ts, P, V

    def residuals(self, x, parts=False):
        out = {"reproj": [], "join": [], "reach": [], "floor": [], "bounce": [], "tau": []}
        ends = []
        for a in range(self.nE - 1):
            t0, t1, ts, P, V = self.arc(x, a)
            idx = self.obs[a]
            tt = np.clip(self.t[idx] - t0, 0, ts[-1])
            out["reproj"].append(((self.cam.project(interp(ts, P, tt)) - self.xy[idx]) / SIG_PX).ravel())
            pe = interp(ts, P, np.array([t1 - t0]))[0]
            ve = interp(ts, V, np.array([t1 - t0]))[0]
            out["join"].append((pe - self.node(x, a + 1)) / SIG_JOIN)
            zs = np.interp(np.linspace(0, t1 - t0, 12), ts, P[:, 2])       # fixed count: length must not vary
            out["floor"].append(np.minimum(zs - R_BALL, 0) / 0.03)
            ends.append((V[0], ve))
        for k, e in enumerate(self.ev):
            p = self.node(x, k)
            if e["kind"] == "hit":
                hand = self.grip_at(self.node_time(x, k), e["player"])
                if hand is not None:
                    # contact is on the paddle face, which spans roughly 0.3 to 1.1 ft from the palm
                    d = np.linalg.norm(p - hand)
                    out["reach"].append(np.array([max(0.0, d - PADDLE_REACH) / PADDLE_SIG, (d - 0.7) / 1.5,
                                                  min(0.0, p[2] - 0.2) / 0.2, 0.0]))
                else:
                    g = self.ground[e["index"], e["player"]]
                    d = np.linalg.norm(p[:2] - g)
                    out["reach"].append(np.array([max(0.0, d - REACH) / 0.4, (p[1] - g[1]) / 4.0,
                                                  min(0.0, p[2] - 0.4) / 0.2, max(0.0, p[2] - 8.5) / 0.3]))
            elif 0 < k < self.nE - 1:
                v_in, v_out = ends[k - 1][1], ends[k][0]
                out["bounce"].append(np.array([(v_out[2] + E_REST * v_in[2]) / (0.25 * abs(v_in[2]) + 1.0),
                                               *((v_out[:2] - K_TAN * v_in[:2]) / (0.4 * np.linalg.norm(v_in[:2]) + 2.0))]))
            out["tau"].append(np.array([x[self.tau0 + k] / (0.04 if e.get("hidden") else 0.02)]))
        if parts:
            return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
        return np.concatenate([np.concatenate(v) for v in out.values() if v])

    def initial(self):
        x = np.zeros(self.n)
        nodes = []
        seen = np.flatnonzero(~np.isnan(self.xy[:, 0]))
        for k, e in enumerate(self.ev):
            px = self.xy[seen[np.argmin(np.abs(seen - e["index"]))]]   # nearest observed sample
            if e["kind"] == "bounce":
                if np.isnan(px[0]) or e.get("hidden"):
                    p = None
                else:
                    p = self.cam.on_plane(px, R_BALL)
            else:
                hand = self.grip_at(self.et[k], e["player"])
                g = self.ground[e["index"], e["player"]]
                p = self.cam.at_depth(px, hand[1] if hand is not None else g[1])
                p[2] = np.clip(p[2], 0.3 if hand is not None else 0.6, 7.0)
            nodes.append(p)
        for k, p in enumerate(nodes):
            if p is None:   # hidden bounce: halfway between neighbours, on the floor
                p = 0.5 * (nodes[k - 1] + nodes[k + 1])
                p[2] = R_BALL
                nodes[k] = p
            x[self.pos_slice[k]] = p[:2] if self.ev[k]["kind"] == "bounce" else p
        for a in range(self.nE - 1):
            T = max(self.et[a + 1] - self.et[a], 0.04)
            v = (nodes[a + 1] - nodes[a]) / T
            v[2] += 0.5 * G * T
            x[self.vel0 + 3 * a: self.vel0 + 3 * a + 3] = v
        return x

    def sparsity(self, x0):
        m = len(self.residuals(x0))
        S = lil_matrix((m, self.n), dtype=int)
        base = self.residuals(x0)
        # numeric probe: perturb each parameter once (cheap enough at this size, done once)
        for j in range(self.n):
            xp = x0.copy()
            xp[j] += 1e-3 if j < self.tau0 or j >= self.vel0 else 1e-4
            S[np.flatnonzero(np.abs(self.residuals(xp) - base) > 1e-12), j] = 1
        return S


def main():
    seg_path, ev_path, players_path, calib_path, prefix = sys.argv[1:6]
    seg = json.loads(Path(seg_path).read_text())
    events = json.loads(Path(ev_path).read_text())["events"]
    players = np.load(players_path)
    cam = Camera(json.loads(Path(calib_path).read_text()))
    pr = Problem(seg, events, players, cam)
    x0 = pr.initial()
    r0 = pr.residuals(x0, parts=True)
    print("initial reprojection rms px:", round(float(np.sqrt(np.mean(r0["reproj"] ** 2)) * SIG_PX * np.sqrt(2)), 2))
    lo = np.full(pr.n, -np.inf)
    hi = np.full(pr.n, np.inf)
    lo[pr.tau0:pr.vel0], hi[pr.tau0:pr.vel0] = -0.06, 0.06
    sol = least_squares(pr.residuals, x0, jac_sparsity=pr.sparsity(x0), loss="soft_l1", f_scale=1.5,
                        bounds=(lo, hi), max_nfev=200, verbose=0)
    x = sol.x
    parts = pr.residuals(x, parts=True)
    rp = parts["reproj"].reshape(-1, 2) * SIG_PX
    err = np.linalg.norm(rp, axis=1)
    print(f"fit: {sol.nfev} evaluations, status {sol.status}")
    print(f"reprojection: rms {np.sqrt(np.mean(err ** 2)):.2f} px, median {np.median(err):.2f}, p95 {np.percentile(err, 95):.2f}, n={len(err)}")
    print(f"join mismatch max {np.abs(parts['join']).max() * SIG_JOIN:.3f} ft")

    # sample the fitted path on every unique frame and collect per-arc facts
    i0, i1 = int(np.searchsorted(pr.t, pr.et[0])), int(np.searchsorted(pr.t, pr.et[-1]))
    path = np.full((len(pr.t), 3), np.nan)
    arcs = []
    k = 0
    for a in range(pr.nE - 1):
        t0, t1, ts, P, V = pr.arc(x, a)
        idx = np.flatnonzero((pr.t >= t0 - 1e-6) & (pr.t <= t1 + 1e-6))
        path[idx] = interp(ts, P, np.clip(pr.t[idx] - t0, 0, t1 - t0))
        n = len(pr.obs[a])
        e = err[k:k + n]
        k += n
        m = ts <= (t1 - t0)
        Pm = P[m]
        cross = None
        s = np.flatnonzero((Pm[:-1, 1] - NET_Y) * (Pm[1:, 1] - NET_Y) < 0)
        if len(s):
            w = (NET_Y - Pm[s[0], 1]) / (Pm[s[0] + 1, 1] - Pm[s[0], 1])
            pc = Pm[s[0]] * (1 - w) + Pm[s[0] + 1] * w
            net_h = (34 + 2 * abs(pc[0] - 10) / 11) / 12     # 34 in at centre, 36 in at posts
            cross = {"x_ft": round(float(pc[0]), 2), "height_ft": round(float(pc[2]), 2),
                     "clearance_in": round(float((pc[2] - R_BALL - net_h) * 12), 1)}
        arcs.append({"from": events[a]["kind"], "to": events[a + 1]["kind"], "t0": round(float(t0), 3),
                     "duration_s": round(float(t1 - t0), 3), "n_obs": int(n),
                     "reproj_rms_px": round(float(np.sqrt(np.mean(e ** 2))), 2) if n else None,
                     "launch_speed_mph": round(float(np.linalg.norm(V[0]) * MPH), 1),
                     "apex_ft": round(float(Pm[:, 2].max()), 2), "net": cross,
                     "start_ft": np.round(P[0], 2).tolist(), "end_ft": np.round(interp(ts, P, np.array([t1 - t0]))[0], 2).tolist()})
    nodes = []
    for kk, e in enumerate(events):
        p = pr.node(x, kk)
        d = {"index": e["index"], "kind": e["kind"], "t": round(float(pr.node_time(x, kk)), 3), "pos_ft": np.round(p, 2).tolist(),
             "time_offset_ms": round(float(x[pr.tau0 + kk] * 1000), 1)}
        if e["kind"] == "hit":
            g = pr.ground[e["index"], e["player"]]
            d.update(player=e["player"], height_ft=round(float(p[2]), 2), dist_to_feet_ft=round(float(np.linalg.norm(p[:2] - g)), 2))
        else:
            d.update(side="near" if p[1] < NET_Y else "far",
                     in_court=bool(0 <= p[0] <= 20 and 0 <= p[1] <= 44),
                     in_kitchen=bool(0 <= p[0] <= 20 and 15 <= p[1] <= 29))
        nodes.append(d)
    out = {"reproj_rms_px": float(np.sqrt(np.mean(err ** 2))), "reproj_median_px": float(np.median(err)),
           "reproj_p95_px": float(np.percentile(err, 95)), "n_obs": int(len(err)), "arcs": arcs, "events": nodes}
    Path(prefix + "_fit.json").write_text(json.dumps(out, indent=1))
    np.savez(prefix + "_path.npz", path=path, t=pr.t, keep=np.array(seg["keep"]), i0=i0, i1=i1, start_frame=seg["start_frame"])

    cl = [a["net"]["clearance_in"] for a in arcs if a["net"]]
    bn = [d for d in nodes if d["kind"] == "bounce"]
    bad = [a for a in arcs if (a["reproj_rms_px"] or 0) > 6 or (a["net"] and a["net"]["clearance_in"] < -4)]
    print(f"SUMMARY arcs {len(arcs)}  suspect {len(bad)}  net clearance in [{min(cl):+.1f}, {max(cl):+.1f}]  bounces in court {sum(d['in_court'] for d in bn)}/{len(bn)}")
    print("\narc  from->to        dur   obs  rms   mph  apex  net clearance")
    for a in arcs:
        nc = f"{a['net']['clearance_in']:+6.1f} in" if a["net"] else "   (same side)"
        print(f"  {a['from']:>6}->{a['to']:<6} {a['duration_s']:5.2f} {a['n_obs']:4d} {a['reproj_rms_px'] or 0:5.2f} {a['launch_speed_mph']:5.1f} {a['apex_ft']:5.2f}  {nc}")
    print("\nevents")
    for d in nodes:
        if d["kind"] == "hit":
            print(f"  hit    t={d['t']:6.2f} p{d['player']} at ({d['pos_ft'][0]:5.1f},{d['pos_ft'][1]:5.1f}) height {d['height_ft']:4.2f} ft, {d['dist_to_feet_ft']:.1f} ft from feet, dt {d['time_offset_ms']:+.0f} ms")
        else:
            print(f"  bounce t={d['t']:6.2f}    at ({d['pos_ft'][0]:5.1f},{d['pos_ft'][1]:5.1f}) {d['side']} side, kitchen={d['in_kitchen']}, in={d['in_court']}, dt {d['time_offset_ms']:+.0f} ms")


if __name__ == "__main__":
    main()
