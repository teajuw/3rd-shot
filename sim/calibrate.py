"""Fit the ball simulator's shared physics to footage and test it on held-out flights and bounces.

Usage: uv run python sim/calibrate.py [seed=11]

Data: every flight between two labelled events in the clips that have trustworthy events
(rally 267: hand-assigned events; b_158, b_404, b_1095: Trevor's labels, live play only).
Flights and bounces are split at random, half to fit shared parameters and half to test.

Test 1, after the bounce: fit the flight coming into a bounce, apply the bounce model, fly on, and
compare with where the ball was actually seen after the bounce. Nothing after the bounce is fitted.
Test 2, landing: fit a flight on its first 60% only, then predict when and where it lands.

Models compared
  old     drag only; bounce keeps 66% of vertical and 80% of horizontal speed (the current pipeline's numbers)
  old+    same form, the two bounce numbers fitted on the training half
  new     drag only in flight; bounce with speed-dependent restitution and slide-or-grip friction, fitted
  new+spin  as new, plus per-flight spin (lift in flight, carried into the bounce)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fit3d")); sys.path.insert(0, str(ROOT / "sim"))
import ballsim as B  # noqa: E402
from physics import Camera  # noqa: E402

CAM = Camera(json.loads((ROOT / "calib/out/t270_calibration.json").read_text()))
SIG_PX, TRIM, REACH = 2.0, 0.04, 4.5
LIVE = {"b_158": (0, 96), "b_404": (365, 1239), "b_1095": (0, 10 ** 9)}


def load_clips():
    clips = []
    seg = json.loads((ROOT / "fit3d/out/rally_267_segments.json").read_text())
    ev = json.loads((ROOT / "fit3d/events_hand_rally_267.json").read_text())["events"]
    clips.append(("rally_267", seg, np.load(ROOT / "players/out/rally_267_tracks.npz"), [dict(e) for e in ev]))
    for cid, (lo, hi) in LIVE.items():
        seg = json.loads((ROOT / f"batch/out/{cid}_segments.json").read_text())
        lab = json.loads((ROOT / f"eval/labels/labels/{cid}.json").read_text()); lab = lab.get("data", lab)
        keep = np.array(seg["keep"]); xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
        tr = np.load(ROOT / f"batch/out/{cid}_tracks.npz"); ground = tr["smooth"][keep]
        breaks = sorted({a["i"] for a in seg["arcs"]} | {a["j"] for a in seg["arcs"]})
        ev = []
        for e in sorted(lab["events"], key=lambda e: e["f"]):
            if e["kind"] in ("start", "end") or not lo <= e["f"] <= hi:
                continue
            i = int(np.argmin(np.abs(keep - e["f"])))
            near = [b for b in breaks if abs(b - i) <= 4]                 # snap to the track's own break when one is close
            if near:
                i = min(near, key=lambda b: abs(b - i))
            if e["kind"] == "bounce":
                ev.append({"index": i, "kind": "bounce"})
            else:
                seen = np.flatnonzero(~np.isnan(xy[:, 0]))
                if not len(seen):
                    continue
                px = xy[seen[np.argmin(np.abs(seen - i))]]
                cand = (0, 1) if e["kind"] == "near" else (2, 3)
                d = [np.linalg.norm(CAM.project(np.array([[*ground[i, p], 3.0]]))[0] - px) if not np.isnan(ground[i, p, 0]) else 1e9 for p in cand]
                ev.append({"index": i, "kind": "hit", "player": cand[int(np.argmin(d))]})
        clips.append((cid, seg, tr, ev))
    return clips


def build(clips):
    """One record per flight: observations, how it starts and ends, and what was seen after it if it ends in a bounce."""
    flights = []
    for cid, seg, tr, ev in clips:
        t = np.array(seg["t"]); xy = np.array([[np.nan if v is None else v for v in p] for p in seg["xy"]])
        ground = tr["smooth"][np.array(seg["keep"])]
        ev = sorted(ev, key=lambda e: e["index"])
        for a in range(len(ev) - 1):
            e0, e1 = ev[a], ev[a + 1]
            t0, t1 = t[e0["index"]], t[e1["index"]]
            if not 0.12 < t1 - t0 < 2.5:
                continue
            m = (t >= t0 + TRIM) & (t <= t1 - TRIM) & ~np.isnan(xy[:, 0])
            if m.sum() < 5:
                continue
            f = {"clip": cid, "t0": t0, "T": t1 - t0, "tt": t[m] - t0, "xy": xy[m], "start": e0["kind"], "end": e1["kind"]}
            seen = np.flatnonzero(~np.isnan(xy[:, 0]))
            for key, e in (("px0", e0), ("px1", e1)):
                f[key] = xy[seen[np.argmin(np.abs(seen - e["index"]))]]
            for key, e in (("g0", e0), ("g1", e1)):
                f[key] = ground[e["index"], e["player"]] if e["kind"] == "hit" else None
            if e1["kind"] == "bounce" and a + 2 < len(ev):
                t2 = min(t[ev[a + 2]["index"]], t1 + 0.35)
                mp = (t >= t1 + 0.02) & (t <= t2 - 0.02) & ~np.isnan(xy[:, 0])
                if mp.sum() >= 3:
                    f["post_t"], f["post_xy"] = t[mp] - t1, xy[mp]
            flights.append(f)
    return flights


def fit_flight(f, p, spin=False, frac=1.0, anchor_end=True):
    """Fit launch position and velocity (and spin) of one flight. frac < 1 uses only the first part of the observations."""
    use = f["tt"] <= f["tt"][0] + frac * (f["tt"][-1] - f["tt"][0]) + 1e-9
    tt, xy = f["tt"][use], f["xy"][use]
    start_b = f["start"] == "bounce"
    if start_b:
        p_init = CAM.on_plane(f["px0"], B.R_BALL)
    else:
        p_init = CAM.at_depth(f["px0"], f["g0"][1]); p_init[2] = np.clip(p_init[2], 0.6, 7.0)
    if anchor_end:
        p_end = CAM.on_plane(f["px1"], B.R_BALL) if f["end"] == "bounce" else CAM.at_depth(f["px1"], f["g1"][1])
        v_init = (p_end - p_init) / f["T"]; v_init[2] += 0.5 * B.G * f["T"]
    else:
        # prediction mode: the end of the flight must not leak in, so start from the last observation used,
        # assuming the ball travels away from the side it started on at a typical 30 ft/s along the court
        vy = 30.0 if p_init[1] < B.NET_Y else -30.0
        p_last = CAM.at_depth(xy[-1], p_init[1] + vy * tt[-1])
        v_init = (p_last - p_init) / tt[-1]; v_init[2] += 0.5 * B.G * tt[-1]
    x0 = np.r_[p_init[:2] if start_b else p_init, v_init, [0.0, 0.0] if spin else []]
    nP = 2 if start_b else 3

    def unpack(x):
        p0 = np.r_[x[:2], B.R_BALL] if start_b else x[:3]
        v0 = x[nP:nP + 3]
        w = None
        if spin:
            hv = np.array([v0[0], v0[1], 0.0]); hv /= max(np.linalg.norm(hv), 1e-9)
            side = np.cross(hv, [0, 0, 1.0])                       # horizontal axis square to travel: topspin/backspin
            w = x[nP + 3] * 100 * side + x[nP + 4] * 100 * np.array([0, 0, 1.0])
        return p0, v0, w

    def res(x):
        p0, v0, w = unpack(x)
        ts, P, V = B.flight(p0, v0, f["T"] + 0.05, p, w)
        r = [((CAM.project(B.at(ts, P, tt)) - xy) / SIG_PX).ravel()]
        zs = np.interp(np.linspace(0, f["T"] if anchor_end else tt[-1], 10), ts, P[:, 2])   # in prediction mode only up to the last observation used
        r.append(np.minimum(zs - B.R_BALL, 0) / 0.03)
        pe = B.at(ts, P, np.array([f["T"]]))[0]
        if anchor_end and f["end"] == "bounce":
            r.append(np.array([(pe[2] - B.R_BALL) / 0.05]))
        if anchor_end and f["end"] == "hit":
            d = np.linalg.norm(pe[:2] - f["g1"]); r.append(np.array([max(0.0, d - REACH) / 0.4, min(0.0, pe[2] - 0.4) / 0.2, max(0.0, pe[2] - 8.5) / 0.3]))
        if not start_b:
            d = np.linalg.norm(p0[:2] - f["g0"]); r.append(np.array([max(0.0, d - REACH) / 0.4, min(0.0, p0[2] - 0.4) / 0.2, max(0.0, p0[2] - 8.5) / 0.3]))
        if spin:
            r.append(x[nP + 3:] / 2.5)                              # weak pull to no spin (scale 250 rad/s, about 2,400 rpm)
        return np.concatenate(r)

    sol = least_squares(res, x0, loss="soft_l1", f_scale=1.5, max_nfev=60)
    p0, v0, w = unpack(sol.x)
    ts, P, V = B.flight(p0, v0, f["T"] + 0.6, p, w)
    err = np.linalg.norm(CAM.project(B.at(ts, P, tt)) - xy, axis=1)
    return {"p0": p0, "v0": v0, "w": w, "ts": ts, "P": P, "V": V, "rms": float(np.sqrt(np.mean(err ** 2)))}


def post_error(f, fit, p):
    """Predict the hop after the bounce from the fitted incoming flight; pixel error against what was seen."""
    pe, ve = B.at(fit["ts"], fit["P"], np.array([f["T"]]))[0], B.at(fit["ts"], fit["V"], np.array([f["T"]]))[0]
    v2, w2 = B.bounce(ve, fit["w"], p)
    ts, P, _ = B.flight(np.array([pe[0], pe[1], B.R_BALL]), v2, f["post_t"][-1] + 0.05, p, w2)
    return np.linalg.norm(CAM.project(B.at(ts, P, f["post_t"])) - f["post_xy"], axis=1)


def landing(f, fit):
    """Time and pixel where the fitted flight first comes down to the floor."""
    ts, P = fit["ts"], fit["P"]
    k = np.flatnonzero((P[1:, 2] < B.R_BALL) & (np.diff(P[:, 2]) < 0))
    if not len(k):
        return None
    k = k[0] + 1
    a = (P[k - 1, 2] - B.R_BALL) / (P[k - 1, 2] - P[k, 2])
    q = P[k - 1] + a * (P[k] - P[k - 1])
    return ts[k - 1] + a * (ts[k] - ts[k - 1]), CAM.project(q[None])[0]


def summarise(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": round(float(np.median(x)), 2), "mean": round(float(np.mean(x)), 2), "p90": round(float(np.percentile(x, 90)), 2)}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    rng = np.random.default_rng(seed)
    flights = build(load_clips())
    bounces = [f for f in flights if "post_t" in f and len(f["tt"]) >= 6]
    lands = [f for f in flights if f["end"] == "bounce" and len(f["tt"]) >= 10]
    print(f"{len(flights)} flights; {len(bounces)} bounces with a visible hop after; {len(lands)} flights long enough for the landing test")
    print("by clip:", {c: sum(f["clip"] == c for f in flights) for c in sorted({f["clip"] for f in flights})})
    base = B.Params()
    out = {"seed": seed, "flights": len(flights)}

    # ---------- test 1: after the bounce ----------
    perm = rng.permutation(len(bounces)); tr_i, te_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
    fits = {"plain": [fit_flight(f, base) for f in bounces], "spin": [fit_flight(f, base, spin=True) for f in bounces]}

    def loss(p, which, idx):
        return float(np.mean([np.sqrt(np.mean(np.minimum(post_error(bounces[i], fits[which][i], p), 60.0) ** 2)) for i in idx]))

    models = {"old": (base.with_(kt=0.8, e0=0.66, e1=0.0), "plain")}
    r = minimize(lambda x: loss(base.with_(kt=float(np.clip(x[1], 0, 1.2)), e0=float(np.clip(x[0], 0.3, 0.95))), "plain", tr_i), [0.66, 0.8], method="Nelder-Mead", options={"xatol": 0.005, "fatol": 0.01, "maxiter": 120})
    models["old+"] = (base.with_(kt=float(np.clip(r.x[1], 0, 1.2)), e0=float(np.clip(r.x[0], 0.3, 0.95))), "plain")
    for name, which in (("new", "plain"), ("new+spin", "spin")):
        mk = lambda x: base.with_(e0=float(np.clip(x[0], 0.3, 0.95)), e1=float(np.clip(x[1], 0, 0.02)), mu=float(np.clip(x[2], 0.02, 1.5)))  # noqa: E731
        best = None
        for x0 in ([0.66, 0.002, 0.4], [0.75, 0.005, 0.2], [0.6, 0.0, 0.8]):
            r = minimize(lambda x: loss(mk(x), which, tr_i), x0, method="Nelder-Mead", options={"xatol": 0.003, "fatol": 0.01, "maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        models[name] = (mk(best.x), which)
    out["bounce_test"] = {"train_n": int(len(tr_i)), "test_n": int(len(te_i)), "models": {}}
    per = {}
    for name, (p, which) in models.items():
        per[name] = np.array([np.sqrt(np.mean(post_error(bounces[i], fits[which][i], p) ** 2)) for i in te_i])
        tr = np.array([np.sqrt(np.mean(post_error(bounces[i], fits[which][i], p) ** 2)) for i in tr_i])
        out["bounce_test"]["models"][name] = {"params": {k: round(getattr(p, k), 4) for k in ("e0", "e1", "mu", "kt")}, "train_px": summarise(tr), "test_px": summarise(per[name])}
    for name in ("old+", "new", "new+spin"):
        out["bounce_test"]["models"][name]["test_bounces_better_than_old"] = f"{int((per[name] < per['old']).sum())} of {len(te_i)}"
    out["bounce_test"]["incoming_fit_rms_px"] = {k: summarise([x["rms"] for x in v]) for k, v in fits.items()}
    out["bounce_test"]["impact_speed_ft_s"] = summarise([abs(B.at(x["ts"], x["V"], np.array([f["T"]]))[0][2]) for f, x in zip(bounces, fits["plain"])])

    # ---------- test 2: landing from the first 60% ----------
    perm = rng.permutation(len(lands)); tr_l, te_l = perm[: len(perm) // 2], perm[len(perm) // 2:]

    def land_err(f, p, spin):
        fit = fit_flight(f, p, spin=spin, frac=0.6, anchor_end=False)
        L = landing(f, fit)
        if L is None:
            return 0.5, 200.0
        return abs(L[0] - f["T"]), float(np.linalg.norm(L[1] - f["px1"]))

    def cd_loss(cd):
        return float(np.mean([min(land_err(lands[i], base.with_(cd=cd), False)[1], 150.0) for i in tr_l]))
    grid = [0.20, 0.27, 0.33, 0.40, 0.47, 0.55]
    cd_best = grid[int(np.argmin([cd_loss(c) for c in grid]))]
    out["landing_test"] = {"train_n": int(len(tr_l)), "test_n": int(len(te_l)), "cd_fitted_on_train": cd_best, "models": {}}
    for name, p, spin in (("old (Cd 0.33, no spin)", base, False), ("drag fitted", base.with_(cd=cd_best), False), ("with spin", base, True)):
        e = np.array([land_err(lands[i], p, spin) for i in te_l])
        out["landing_test"]["models"][name] = {"time_error_ms": summarise(e[:, 0] * 1000), "pixel_error": summarise(e[:, 1])}
    (ROOT / "sim/out").mkdir(exist_ok=True)
    (ROOT / f"sim/out/calibration_seed{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
