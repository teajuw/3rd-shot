"""Score the pipeline's automatic hits and bounces against hand labels.

Labels: one JSON per clip under <labels_dir>/labels/<clip>.json, as exported from the
Rally Labeller page ({"events": [{"f": clip_frame, "kind": near|far|bounce|end}], "done": bool}).
Pipeline: batch/out/<clip>_s*_events.json (times on the 50 Hz unique-frame clock).

A predicted event matches a labelled one of the same type if it is within TOL frames
(60 fps); each label is matched at most once. Only clips marked done are scored.

Usage: python score.py [labels_dir=eval/labels] [split=dev|test|all] [events_dir=batch/out]
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOL = 5   # frames at 60 fps = 83 ms


def load_labels(d, clip):
    p = Path(d) / "labels" / f"{clip}.json"
    if not p.exists():
        return None
    x = json.loads(p.read_text())
    x = x.get("data", x)
    return x if x.get("done") else None


def predicted(events_dir, clip):
    segp = ROOT / events_dir / f"{clip}_segments.json"
    if not segp.exists():
        return []
    seg = json.loads(segp.read_text())
    t, keep = np.array(seg["t"]), np.array(seg["keep"], float)
    out = []
    for f in sorted(glob.glob(str(ROOT / events_dir / f"{clip}_s*_events.json"))):
        for e in json.loads(Path(f).read_text())["events"]:
            fr = float(np.interp(e["t"], t, keep))
            kind = "bounce" if e["kind"] == "bounce" else ("far" if e["player"] >= 2 else "near")
            out.append({"f": fr, "kind": kind})
    return out


def match(pred, gold):
    """Greedy nearest matching within TOL. Returns (pairs, unmatched_pred, unmatched_gold)."""
    cand = sorted((abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold) if abs(p - g) <= TOL)
    up, ug, pairs = set(range(len(pred))), set(range(len(gold))), []
    for d, i, j in cand:
        if i in up and j in ug:
            up.discard(i); ug.discard(j); pairs.append((i, j))
    return pairs, sorted(up), sorted(ug)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    labels_dir = sys.argv[1] if len(sys.argv) > 1 else "eval/labels"
    split = sys.argv[2] if len(sys.argv) > 2 else "dev"
    events_dir = sys.argv[3] if len(sys.argv) > 3 else "batch/out"
    clips = [c for c in json.loads((ROOT / "eval/clips.json").read_text())["clips"] if split == "all" or c["split"] == split]
    tot = {"hit": [0, 0, 0], "bounce": [0, 0, 0]}
    side_ok = side_n = 0
    dts, scored = [], 0
    print(f"split={split}  tolerance ±{TOL} frames")
    print("clip      hits tp/fp/fn   bounces tp/fp/fn   side ok   point-over marks")
    for c in clips:
        L = load_labels(ROOT / labels_dir, c["id"])
        if L is None:
            print(f"  {c['id']:<8} not labelled yet")
            continue
        scored += 1
        P = predicted(events_dir, c["id"])
        row = {}
        for name, kinds in (("hit", ("near", "far")), ("bounce", ("bounce",))):
            g = [e for e in L["events"] if e["kind"] in kinds]
            p = [e for e in P if e["kind"] in kinds]
            pairs, up, ug = match([e["f"] for e in p], [e["f"] for e in g])
            row[name] = (len(pairs), len(up), len(ug))
            for k in range(3):
                tot[name][k] += row[name][k]
            dts += [p[i]["f"] - g[j]["f"] for i, j in pairs]
            if name == "hit":
                ok = sum(p[i]["kind"] == g[j]["kind"] for i, j in pairs)
                side_ok += ok; side_n += len(pairs); row["side"] = f"{ok}/{len(pairs)}"
        ends = sum(e["kind"] == "end" for e in L["events"])
        print(f"  {c['id']:<8} {row['hit'][0]:>3}/{row['hit'][1]}/{row['hit'][2]:<8} {row['bounce'][0]:>5}/{row['bounce'][1]}/{row['bounce'][2]:<8} {row['side']:>7}   {ends}")
    if not scored:
        print("\nNo finished clips in this split yet.")
        return
    ph, rh, fh = prf(*tot["hit"])
    pb, rb, fb = prf(*tot["bounce"])
    print(f"\nclips scored: {scored}/{len(clips)}")
    print(f"hits:    precision {ph:.2f}  recall {rh:.2f}  F1 {fh:.2f}   (tp {tot['hit'][0]}, false {tot['hit'][1]}, missed {tot['hit'][2]})")
    print(f"bounces: precision {pb:.2f}  recall {rb:.2f}  F1 {fb:.2f}   (tp {tot['bounce'][0]}, false {tot['bounce'][1]}, missed {tot['bounce'][2]})")
    print(f"hitter side correct on matched hits: {side_ok}/{side_n}")
    if dts:
        print(f"timing on matches: median {np.median(dts):+.1f} frames, median |error| {np.median(np.abs(dts)):.1f} frames")
    print(f"SCORE (mean of hit F1 and bounce F1): {(fh + fb) / 2:.3f}")


if __name__ == "__main__":
    main()
