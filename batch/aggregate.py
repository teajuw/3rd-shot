"""Summarise batch/out/results.json: stage failures, events, and flight quality.

Usage: python aggregate.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "batch/out/results.json").read_text())

print(f"shots: {len(res)}, footage {sum(r['duration_s'] for r in res):.0f} s, compute {sum(r['seconds'] for r in res) / 60:.0f} min")
for st in ("players", "ball", "segment"):
    print(f"  stage {st}: ok on {sum(bool(r['stages'].get(st)) for r in res)}/{len(res)}")
late = [(r["tag"], r["player_init_frame"]) for r in res if r.get("player_init_frame", 0) > 30]
print(f"  player tracker started late (>0.5 s) on {len(late)} shots: {late}")
print(f"  ball detected: median {np.median([r['ball']['detected_pct'] for r in res if 'ball' in r]):.0f}% of frames per shot")

flights, events, rows = [], [], []
n_str = n_fit = 0
for r in res:
    for s in r["stretches"]:
        n_str += 1
        row = {"shot": r["tag"], "t0": round(s["t0"], 1), "t1": round(s["t1"], 1), "hits": s["hits"], "bounces": s["bounces"]}
        if s.get("fit"):
            n_fit += 1
            f = json.loads((ROOT / s["fit"]).read_text())
            arcs = f["arcs"]
            for a in arcs:
                a["suspect"] = (a["reproj_rms_px"] or 0) > 6 or bool(a["net"] and a["net"]["clearance_in"] < -4)
                a["shot"] = r["tag"]
            flights += arcs
            events += f["events"]
            row.update(flights=len(arcs), suspect=sum(a["suspect"] for a in arcs), median_px=round(f["reproj_median_px"], 1))
        rows.append(row)

print(f"\nstretches of 3 s or more: {n_str}; with at least 3 events and a completed fit: {n_fit}")
print("shot      window        hits bounces flights suspect median_px")
for w in rows:
    print(f"  {w['shot']:<8} {w['t0']:5.1f}-{w['t1']:5.1f}  {w['hits']:4d} {w['bounces']:6d} {w.get('flights', '-'):>7} {w.get('suspect', '-'):>7} {w.get('median_px', '-'):>8}")

if flights:
    F = flights
    sus = [a for a in F if a["suspect"]]
    print(f"\nflights: {len(F)}; suspect {len(sus)} ({100 * len(sus) / len(F):.0f}%)")
    print(f"  reprojection rms per flight: median {np.median([a['reproj_rms_px'] for a in F if a['reproj_rms_px']]):.1f} px, "
          f"75th pct {np.percentile([a['reproj_rms_px'] for a in F if a['reproj_rms_px']], 75):.1f}")
    cross = [a["net"]["clearance_in"] for a in F if a["net"]]
    print(f"  net crossings: {len(cross)}; below the tape by more than 4 in: {sum(c < -4 for c in cross)}; below at all: {sum(c < 0 for c in cross)}")
    bn = [e for e in events if e["kind"] == "bounce"]
    print(f"  bounces: {len(bn)}; in court {sum(e['in_court'] for e in bn)}")
    hits = [e for e in events if e["kind"] == "hit"]
    hz = np.array([e["height_ft"] for e in hits])
    print(f"  hits: {len(hits)}; contact height median {np.median(hz):.1f} ft, below 0.3 ft: {(hz < 0.3).sum()}, above 8 ft: {(hz > 8).sum()}")
    sp = np.array([a["launch_speed_mph"] for a in F if a["from"] == "hit"])
    print(f"  launch speed after a hit: median {np.median(sp):.0f} mph, 90th pct {np.percentile(sp, 90):.0f}, over 70 mph: {(sp > 70).sum()}")
    kinds = {}
    for a in F:
        k = f"{a['from']}->{a['to']}" + (" (<0.15 s)" if a["duration_s"] < 0.15 else "")
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        kinds[k][1] += a["suspect"]
    print("  suspect by flight type:", {k: f"{v[1]}/{v[0]}" for k, v in sorted(kinds.items())})
    per = {}
    for a in F:
        per.setdefault(a["shot"], [0, 0])
        per[a["shot"]][0] += 1
        per[a["shot"]][1] += a["suspect"]
    print("  suspect by shot:", {k: f"{v[1]}/{v[0]}" for k, v in per.items()})
