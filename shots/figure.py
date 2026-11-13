"""Draw the article figure: one typical held-out shot, predicted without and with the opponents' positions.

Usage: uv run python shots/figure.py     (writes article/shot-maps.png)
The shot is the one whose gain is the median among bounced, in-court test shots (seed 7), so it is not a best case.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M

rows = json.loads((M.ROOT / "shots/out/shots.json").read_text()); r, T, F = M.table(rows)
clips = sorted({x["clip"] for x in r}); rng = np.random.default_rng(7)
fo = {c: k % 5 for k, c in enumerate(rng.permutation(clips))}; fold = np.array([fo[x["clip"]] for x in r])
gx, gy = np.meshgrid(np.arange(0.25, 20, 0.5), np.arange(22.25, 44, 0.5)); grid = np.stack([gx.ravel(), gy.ravel()], 1)
gain = np.zeros(len(r)); cfg = {}
for k in range(5):
    te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
    hf, _ = M.choose(T, F, tr, "frequency"); ho, lo = M.choose(T, F, tr, "+ opponents")
    gain[te] = np.log(M.density(T, F, tr, te, "+ opponents", ho, lo)) - np.log(M.density(T, F, tr, te, "frequency", hf, 1.0))
    for i in te:
        cfg[i] = (tr, hf, ho, lo)
ok = np.array([x["target_kind"] == "bounce" and 0 <= t[0] <= 20 and 22 <= t[1] <= 44 for x, t in zip(r, T)])
i = np.flatnonzero(ok)[np.argmin(abs(gain[ok] - np.median(gain[ok])))]
tr, hf, ho, lo = cfg[i]


def dens(m, h, ls):
    p = M.gauss(T[tr], np.full((len(grid), len(tr)), 1 / len(tr)), grid, h)
    if m != "frequency":
        w = M.weights(F, M.FEATS[m], tr, np.array([i]), ls); p = 0.85 * M.gauss(T[tr], np.repeat(w, len(grid), 0), grid, h) + 0.15 * p
    return (p / p.sum()).reshape(gx.shape)


a, b = dens("frequency", hf, 1.0), dens("+ opponents", ho, lo); vmax = max(a.max(), b.max()); x = r[i]
fig, axes = plt.subplots(1, 2, figsize=(8, 8.6), dpi=200)
for ax, d, title in ((axes[0], a, "Ignoring the situation"), (axes[1], b, "Knowing where all four stand")):
    ax.imshow(d, extent=(0, 20, 22, 44), origin="lower", cmap="Blues", vmin=0, vmax=vmax, alpha=0.95)
    for y in (0, 15, 29, 44):
        ax.plot([0, 20], [y, y], color="#444", lw=1)
    for xx in (0, 20):
        ax.plot([xx, xx], [0, 44], color="#444", lw=1)
    ax.plot([10, 10], [0, 15], color="#444", lw=1); ax.plot([10, 10], [29, 44], color="#444", lw=1)
    ax.plot([-1, 21], [22, 22], color="#111", lw=3)
    ax.plot([x["hitter_xy"][0], x["target"][0]], [x["hitter_xy"][1], x["target"][1]], color="#888", lw=1, ls=":")
    ax.scatter(*x["hitter_xy"], s=150, color="#111", zorder=5, label="Hitter")
    ax.scatter(*x["partner_xy"], s=120, facecolor="white", edgecolor="#111", lw=1.5, zorder=5, label="Partner")
    o = np.array(x["opponents_xy"]); ax.scatter(o[:, 0], o[:, 1], s=130, marker="D", color="#eb6834", edgecolor="white", zorder=5, label="Opponents")
    ax.scatter(*x["target"], s=170, marker="x", color="#111", lw=3, zorder=6, label="Where it bounced")
    ax.set_title(title, fontsize=15, pad=10); ax.set_xlim(-2, 22); ax.set_ylim(-1, 45); ax.set_aspect("equal"); ax.axis("off")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=12, handletextpad=0.3, columnspacing=1.2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07, wspace=0.02)
fig.savefig(M.ROOT / "article/shot-maps.png", facecolor="white")
print("shot", x["clip"], x["t"], "gain", round(float(gain[i]), 3))
