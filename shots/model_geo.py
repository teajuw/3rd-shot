"""Does describing the target by court geometry help? Same situations and folds as shots/model.py, different target coordinates.

Usage: uv run python shots/model_geo.py [seed=7]
  xy     the "+ opponents" model from model.py: smooths past targets in court feet
  ray    the same weights, but each past target is stored as (direction from the hitter, share of the room used on that
         line) and replayed from the new hitter's position, so "two feet inside the sideline" stays two feet inside
Both are scored as a density over court feet (the ray model is converted with its change-of-variables factor).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M

NET = 22.0


def to_ray(h, t):
    d = t - h; dist = np.hypot(d[:, 0], d[:, 1]); a = np.arctan2(d[:, 0], d[:, 1])
    dx, dy = np.sin(a), np.cos(a)
    s_net = (NET - h[:, 1]) / dy
    with np.errstate(divide="ignore", invalid="ignore"):
        s_side = np.where(dx > 0, (20 - h[:, 0]) / dx, np.where(dx < 0, -h[:, 0] / dx, np.inf))
    room = np.maximum(np.minimum((44 - h[:, 1]) / dy, s_side) - s_net, 2.0)
    return np.degrees(a), (dist - s_net) / room, dist, room


def ray_density(T, H, F, tr, te, ha, hu, ls, loo=False):
    a_tr, u_tr, _, _ = to_ray(H[tr], T[tr]); a, u, dist, room = to_ray(H[te], T[te])
    w = M.weights(F, M.FEATS["+ opponents"], tr, te, ls, loo)
    k = np.exp(-0.5 * ((a[:, None] - a_tr[None]) / ha) ** 2 - 0.5 * ((u[:, None] - u_tr[None]) / hu) ** 2) / (2 * np.pi * ha * hu)
    p_au = (w * k).sum(1)                                    # per degree per unit share
    return p_au * (180 / np.pi) / (dist * room)              # per square foot


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rows = json.loads((M.ROOT / "shots/out/shots.json").read_text()); r, T, F = M.table(rows)
    H = np.stack([F["hx"], F["hy"]], 1)
    clips = sorted({x["clip"] for x in r}); rng = np.random.default_rng(seed)
    fo = {c: k % 5 for k, c in enumerate(rng.permutation(clips))}; fold = np.array([fo[x["clip"]] for x in r])
    ll = {m: np.zeros(len(r)) for m in ("frequency", "xy", "ray", "mix")}
    for k in range(5):
        te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        hf, _ = M.choose(T, F, tr, "frequency"); h, ls = M.choose(T, F, tr, "+ opponents")
        freq = lambda q, loo=False: M.density(T, F, tr, q, "frequency", hf, 1.0, loo=loo)  # noqa: E731
        best = (-np.inf,)
        for ha in (3, 6, 10, 15):
            for hu in (0.08, 0.15, 0.25):
                for l2 in (0.5, 1.0, 2.0):
                    p = 0.97 * (0.85 * ray_density(T, H, F, tr, tr, ha, hu, l2, loo=True) + 0.15 * freq(tr, True) / 0.97 - 0.15 * 0.03 / M.AREA / 0.97) + 0.03 / M.AREA
                    v = np.log(np.maximum(p, 1e-300)).mean()
                    if v > best[0]:
                        best = (v, ha, hu, l2)
        _, ha, hu, l2 = best
        base = (freq(te) - 0.03 / M.AREA) / 0.97
        p_ray = 0.97 * (0.85 * ray_density(T, H, F, tr, te, ha, hu, l2) + 0.15 * base) + 0.03 / M.AREA
        p_xy = M.density(T, F, tr, te, "+ opponents", h, ls)
        ll["frequency"][te] = np.log(freq(te)); ll["xy"][te] = np.log(p_xy); ll["ray"][te] = np.log(p_ray); ll["mix"][te] = np.log(0.5 * p_xy + 0.5 * p_ray)
        print(f"  fold {k}: ray widths {ha} deg, {hu} share, situation {l2}", flush=True)
    boot = np.random.default_rng(0); idx = [np.flatnonzero([x["clip"] == c for x in r]) for c in clips]
    bs = [np.concatenate([idx[i] for i in boot.integers(0, len(clips), len(clips))]) for _ in range(400)]
    kind = np.array([x["target_kind"] for x in r]); res = {}
    for m in ("xy", "ray", "mix"):
        d = ll[m] - ll["xy"]; g = [d[b].mean() for b in bs]
        res[m] = {"log_likelihood_per_shot": round(float(ll[m].mean()), 3), "gain_over_xy": round(float(d.mean()), 3),
                  "gain_90pct_interval": [round(float(np.percentile(g, 5)), 3), round(float(np.percentile(g, 95)), 3)],
                  "by_kind": {k2: round(float(d[kind == k2].mean()), 3) for k2 in ("bounce", "volley")}}
        print(m, res[m])
    (M.ROOT / f"shots/out/eval_geo_seed{seed}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
