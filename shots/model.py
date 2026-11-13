"""Shot-target distribution, version 0: where on the opponents' side will this shot go?

Usage: uv run python shots/model.py [seed=7]      (reads shots/out/shots.json; writes shots/out/eval_seed<seed>.json)

Every model returns a probability density over the target point (feet, hitter always at the near end).
  uniform       the same everywhere (reference)
  frequency     where shots went overall, ignoring the situation (a smoothed map of training targets)
  position      conditioned on where the hitter stood
  + stroke      also on contact height and whether the ball had bounced
  + opponents   also on where the two opponents stood
Conditional models are kernel-weighted maps: training shots from similar situations count more. They need no
training beyond choosing a few widths, which is done inside the training folds only.

Evaluation: clips are split at random into 5 folds; each fold is predicted by a model that never saw its clips.
Reported: mean log-likelihood per shot (higher is better), how often the true 5.5 x 5 ft zone is the model's top
zone or in its top three (16 zones), and calibration of the zone probabilities.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
X0, X1, Y0, Y1 = -3.0, 23.0, 20.0, 48.0            # where a target may fall (court is x 0..20, y 22..44, plus a margin for out balls)
AREA = (X1 - X0) * (Y1 - Y0)
FEATS = {"position": ["hx", "hy"], "+ stroke": ["hx", "hy", "height", "bounce"], "+ opponents": ["hx", "hy", "height", "bounce", "o1x", "o1y", "o2x", "o2y"]}
SCALE = {"hx": 4.0, "hy": 5.0, "height": 2.0, "bounce": 0.7, "o1x": 5.0, "o1y": 6.0, "o2x": 5.0, "o2y": 6.0}      # "one unit of difference", in feet


def table(rows):
    r = [x for x in rows if x["target"] is not None and X0 <= x["target"][0] <= X1 and Y0 <= x["target"][1] <= Y1]
    T = np.array([x["target"] for x in r], float)
    F = {"hx": [x["hitter_xy"][0] for x in r], "hy": [x["hitter_xy"][1] for x in r], "height": [x["contact_height_ft"] for x in r],
         "bounce": [float(x["after_bounce"]) for x in r], "o1x": [x["opponents_xy"][0][0] for x in r], "o1y": [x["opponents_xy"][0][1] for x in r],
         "o2x": [x["opponents_xy"][1][0] for x in r], "o2y": [x["opponents_xy"][1][1] for x in r]}
    return r, T, {k: np.array(v, float) for k, v in F.items()}


def gauss(T_train, w, q, h):
    """Density at points q (m, 2) of a weighted sum of round Gaussians (width h) at T_train. w: (m, n) rows sum to 1."""
    d2 = ((q[:, None, :] - T_train[None]) ** 2).sum(-1)
    return (w * np.exp(-0.5 * d2 / h ** 2)).sum(1) / (2 * np.pi * h ** 2)


def weights(F, feats, tr, te, ls, loo=False):
    d2 = np.zeros((len(te), len(tr)))
    for k in feats:
        d2 += ((F[k][te][:, None] - F[k][tr][None]) / (SCALE[k] * ls)) ** 2
    w = np.exp(-0.5 * d2)
    if loo:
        np.fill_diagonal(w, 0)
    return w / np.maximum(w.sum(1, keepdims=True), 1e-300)


def density(T, F, tr, te, model, h, ls, q=None, loo=False, shrink=0.15):
    """Density of each test shot's situation at q (default: its own true target). Mixed with uniform so it is never zero."""
    q = T[te] if q is None else q
    if model == "uniform":
        return np.full(len(q), 1 / AREA)
    w_all = np.ones((len(te), len(tr)))
    if loo:
        np.fill_diagonal(w_all, 0)
    w_all /= w_all.sum(1, keepdims=True)
    p = gauss(T[tr], w_all, q, h)
    if model != "frequency":
        p = (1 - shrink) * gauss(T[tr], weights(F, FEATS[model], tr, te, ls, loo), q, h) + shrink * p
    return 0.97 * p + 0.03 / AREA


def choose(T, F, tr, model):
    """Pick the map width h (and the situation width for conditional models) by leave-one-out on the training shots."""
    best = (-np.inf, 2.0, 1.0)
    for h in (1.0, 1.5, 2.0, 3.0, 4.0):
        for ls in ((1.0,) if model == "frequency" else (0.5, 1.0, 2.0, 4.0)):
            ll = np.log(density(T, F, tr, tr, model, h, ls, loo=True)).mean()
            if ll > best[0]:
                best = (ll, h, ls)
    return best[1], best[2]


ZX, ZY = np.linspace(0, 20, 5), np.linspace(22, 44, 5)


def zone_probs(T, F, tr, te, model, h, ls):
    """Probability of each of 16 zones for each test shot (integrated on a 1 ft grid), renormalised over the court."""
    gx, gy = np.meshgrid(np.arange(0.5, 20, 1.0), np.arange(22.5, 44, 1.0))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    zi = (np.clip(np.searchsorted(ZX, grid[:, 0], "right") - 1, 0, 3) * 4 + np.clip(np.searchsorted(ZY, grid[:, 1], "right") - 1, 0, 3))
    out = np.zeros((len(te), 16))
    for a, i in enumerate(te):
        p = density(T, F, tr, np.array([i]), model, h, ls, q=grid) if model == "uniform" else None
        if p is None:
            w_all = np.full((len(grid), len(tr)), 1 / len(tr))
            p = gauss(T[tr], w_all, grid, h)
            if model != "frequency":
                w = weights(F, FEATS[model], tr, np.array([i]), ls)
                p = 0.85 * gauss(T[tr], np.repeat(w, len(grid), 0), grid, h) + 0.15 * p
            p = 0.97 * p + 0.03 / AREA
        out[a] = np.bincount(zi, weights=p, minlength=16)
    return out / out.sum(1, keepdims=True)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rows = json.loads((ROOT / "shots/out/shots.json").read_text())
    r, T, F = table(rows)
    clips = sorted({x["clip"] for x in r})
    rng = np.random.default_rng(seed)
    fold_of = {c: k % 5 for k, c in enumerate(rng.permutation(clips))}
    fold = np.array([fold_of[x["clip"]] for x in r])
    true_zone = np.clip(np.searchsorted(ZX, T[:, 0], "right") - 1, 0, 3) * 4 + np.clip(np.searchsorted(ZY, T[:, 1], "right") - 1, 0, 3)
    in_court = (T[:, 0] >= 0) & (T[:, 0] <= 20) & (T[:, 1] >= 22) & (T[:, 1] <= 44)
    models = ["uniform", "frequency"] + list(FEATS)
    res = {"seed": seed, "shots_total": len(rows), "shots_with_target": len(r), "clips": len(clips),
           "by_target_kind": {k: int(sum(x["target_kind"] == k for x in r)) for k in ("bounce", "volley")}, "models": {}}
    ll = {m: np.zeros(len(r)) for m in models}; zp = {m: np.zeros((len(r), 16)) for m in models}; picked = {m: [] for m in models}
    for k in range(5):
        te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        if not len(te) or len(tr) < 10:
            continue
        for m in models:
            h, ls = (2.0, 1.0) if m == "uniform" else choose(T, F, tr, m)
            picked[m].append((h, ls))
            ll[m][te] = np.log(density(T, F, tr, te, m, h, ls))
            zp[m][te] = zone_probs(T, F, tr, te, m, h, ls)
    boot = np.random.default_rng(0)
    kind = np.array([x["target_kind"] for x in r])
    for m in models:
        top = np.argsort(-zp[m], 1)
        t1 = (top[:, 0] == true_zone)[in_court].mean(); t3 = (top[:, :3] == true_zone[:, None]).any(1)[in_court].mean()
        conf = zp[m].max(1)[in_court]; hit = (top[:, 0] == true_zone)[in_court]
        ece = sum(abs(hit[(conf >= a) & (conf < a + 0.1)].mean() - conf[(conf >= a) & (conf < a + 0.1)].mean()) * ((conf >= a) & (conf < a + 0.1)).mean()
                  for a in np.arange(0, 1, 0.1) if ((conf >= a) & (conf < a + 0.1)).any())
        d = ll[m] - ll["frequency"]
        # uncertainty of the gain over the frequency map: resample whole clips
        idx_by_clip = [np.flatnonzero([x["clip"] == c for x in r]) for c in clips]
        bs = [np.concatenate([idx_by_clip[i] for i in boot.integers(0, len(clips), len(clips))]) for _ in range(400)]
        gains = [d[b].mean() for b in bs]
        res["models"][m] = {"log_likelihood_per_shot": round(float(ll[m].mean()), 3), "gain_over_frequency": round(float(d.mean()), 3),
                            "gain_90pct_interval": [round(float(np.percentile(gains, 5)), 3), round(float(np.percentile(gains, 95)), 3)],
                            "top1_zone_pct": round(100 * float(t1), 1), "top3_zone_pct": round(100 * float(t3), 1), "calibration_error": round(float(ece), 3),
                            "gain_by_target_kind": {k: round(float(d[kind == k].mean()), 3) for k in ("bounce", "volley")},
                            "widths_chosen": picked[m]}
    (ROOT / f"shots/out/eval_seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(f"{res['shots_with_target']} shots with a target from {res['clips']} clips ({res['by_target_kind']})")
    for m in models:
        v = res["models"][m]
        print(f"  {m:12s} log-lik {v['log_likelihood_per_shot']:7.3f}  gain {v['gain_over_frequency']:+.3f} {v['gain_90pct_interval']}  top-1 {v['top1_zone_pct']:5.1f}%  top-3 {v['top3_zone_pct']:5.1f}%  cal err {v['calibration_error']:.3f}  gain by kind {v['gain_by_target_kind']}")


if __name__ == "__main__":
    main()
