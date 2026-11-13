"""Court geometry seen from each hit, how far opponents move while the ball flies, and two samples to check by hand.

Usage: uv run python shots/geometry.py [seed=11] [n_random=40] [n_geometric=20]
Reads shots/out/shots.json and batch/out/*_tracks.npz. Writes shots/out/geometry.json (one row per shot, same order)
and shots/out/samples.json.

Geometry (hitter always at the near end, y < 22; net at y = 22; feet):
  For a direction a (degrees, 0 = straight up the court, + = toward x = 20) from the contact point:
    to_net(a)   distance along the floor to the net
    net_in(a)   net height where that line crosses it: 34 in at the middle, 36 in at the posts
    room(a)     distance from the net to the far boundary along that line: what the hitter has to work with
  window        the range of a that crosses the net between the posts
  Each shot with a target also gets: the direction it used, where that sits in the window (0 = left edge, 1 = right edge),
  the room on that line, how much of the room it used (1 = on the line), and the distance from the target to the nearest line.

Samples:
  random      n_random hits drawn uniformly from every row of the table. This is the one that measures label quality.
  geometric   the n_geometric shots with a target that had the least room on the line they used (sharpest angles),
              excluding any already in the random sample. A separate, deliberately hard suite; not a measure of the average.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NET = 22.0


def ray(h, a_deg):
    """to_net, net height (in), room after the net, for directions a_deg from point h. NaN where the line misses the net."""
    a = np.radians(np.asarray(a_deg, float))
    dx, dy = np.sin(a), np.cos(a)
    s_net = (NET - h[1]) / dy
    xn = h[0] + s_net * dx
    ok = (xn >= 0) & (xn <= 20)
    s_back = (44 - h[1]) / dy
    s_side = np.where(dx > 0, (20 - h[0]) / np.where(dx == 0, 1, dx), np.where(dx < 0, (0 - h[0]) / np.where(dx == 0, 1, dx), np.inf))
    s_out = np.minimum(s_back, s_side)
    net_in = 34 + 2 * np.abs(xn - 10) / 10
    return np.where(ok, s_net, np.nan), np.where(ok, net_in, np.nan), np.where(ok, s_out - s_net, np.nan)


def describe(h, t):
    lo = np.degrees(np.arctan2(0 - h[0], NET - h[1])); hi = np.degrees(np.arctan2(20 - h[0], NET - h[1]))
    out = {"window_deg": [round(float(lo), 1), round(float(hi), 1)]}
    prof = ray(h, np.arange(-80, 81, 5))
    out["profile_5deg"] = {"to_net": [None if np.isnan(v) else round(float(v), 1) for v in prof[0]],
                           "net_in": [None if np.isnan(v) else round(float(v), 2) for v in prof[1]],
                           "room": [None if np.isnan(v) else round(float(v), 1) for v in prof[2]]}
    if t is None:
        return out
    a = float(np.degrees(np.arctan2(t[0] - h[0], t[1] - h[1])))
    to_net, net_in, room = (float(v) for v in ray(h, a))
    dist = float(np.hypot(t[0] - h[0], t[1] - h[1]))
    out.update({"dir_deg": round(a, 1), "window_pos": round((a - lo) / (hi - lo), 3),
                "to_net": None if np.isnan(to_net) else round(to_net, 2), "net_in": None if np.isnan(net_in) else round(net_in, 2),
                "room": None if np.isnan(room) else round(room, 2),
                "room_used": None if np.isnan(room) or room <= 0 else round((dist - to_net) / room, 3),
                "line_margin": round(float(min(t[0], 20 - t[0], 44 - t[1])), 2)})
    return out


def flight_moves(rows):
    """How far the opponent nearest the target moves toward it between the hit and the ball arriving."""
    cache, res = {}, []
    for r in rows:
        if r["target"] is None or not r["dt"]:
            res.append(None); continue
        if r["clip"] not in cache:
            cache[r["clip"]] = np.load(ROOT / f"batch/out/{r['clip']}_tracks.npz")["smooth"]
        g = cache[r["clip"]]
        f0 = int(round(r["t"] * 60)); f1 = min(len(g) - 1, int(round((r["t"] + r["dt"]) * 60)))
        far = r["side"] == "far"
        opp = (0, 1) if far else (2, 3)
        fl = lambda p: np.array([20 - p[0], 44 - p[1]]) if far else np.array(p, float)  # noqa: E731
        t = np.array(r["target"])
        d0 = [np.linalg.norm(fl(g[f0, q]) - t) for q in opp]; d1 = [np.linalg.norm(fl(g[f1, q]) - t) for q in opp]
        if np.isnan(d0 + d1).any():
            res.append(None); continue
        q = int(np.argmin(d1))
        res.append({"closer_ft": round(float(d0[q] - d1[q]), 2), "start_ft": round(float(d0[q]), 2), "moved_ft": round(float(np.linalg.norm(g[f1, opp[q]] - g[f0, opp[q]])), 2)})
    return res


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    n_rand = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n_geo = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    rows = json.loads((ROOT / "shots/out/shots.json").read_text())
    geo = [describe(r["hitter_xy"], r["target"]) for r in rows]
    mv = flight_moves(rows)
    for g, m in zip(geo, mv):
        g["opponent_during_flight"] = m
    (ROOT / "shots/out/geometry.json").write_text(json.dumps(geo))
    rng = np.random.default_rng(seed)
    rand = sorted(int(i) for i in rng.choice(len(rows), n_rand, replace=False))
    cand = [i for i, g in enumerate(geo) if g.get("room") is not None and rows[i]["target_kind"] == "bounce" and i not in rand]
    geom = sorted(cand, key=lambda i: geo[i]["room"])[:n_geo]
    (ROOT / "shots/out/samples.json").write_text(json.dumps({"seed": seed, "random": rand, "geometric": geom}, indent=1))
    b = [g for g, r in zip(geo, rows) if r["target_kind"] == "bounce" and g.get("room") is not None]
    q = lambda k, src: np.percentile([g[k] for g in src if g.get(k) is not None], [10, 50, 90]).round(2).tolist()  # noqa: E731
    print(f"{len(rows)} shots; {len(b)} bounce targets on a line that crosses the net")
    print("  direction used, deg (10/50/90%):", q("dir_deg", b)); print("  place in the window 0..1:", q("window_pos", b))
    print("  room on the line used, ft:", q("room", b)); print("  share of the room used:", q("room_used", b)); print("  distance to nearest line, ft:", q("line_margin", b))
    print("  past the line (out):", sum(g["line_margin"] < 0 for g in b))
    for kind in ("bounce", "volley"):
        m = [g["opponent_during_flight"] for g, r in zip(geo, rows) if r["target_kind"] == kind and g["opponent_during_flight"]]
        print(f"  {kind}: nearest opponent starts {np.median([x['start_ft'] for x in m]):.1f} ft from the target, gets {np.median([x['closer_ft'] for x in m]):.1f} ft closer in flight (median; n={len(m)})")
    print("  random sample kinds:", {k: sum(rows[i]["target_kind"] == k for i in rand) for k in ("bounce", "volley", "none")})
    print("  geometric sample room, ft:", [geo[i]["room"] for i in geom])


if __name__ == "__main__":
    main()
