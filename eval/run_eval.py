"""Run an event detector on every labelled clip, grade it, and write one hill-climb round.

Usage: python eval/run_eval.py <round_name> <detector> [description]
  detector: batch   the events the unattended batch run produced (auto_events.py, stretches >= 3 s)
            decode  fit3d/decode_events.py on the whole shot
Writes .claude/hillclimb/events/<round_name>/{results.jsonl, summary.json, traces/<clip>_rep0.json}.

Grading: a predicted event is right if a label of the same class (near hit, far hit,
bounce) lies within TOL frames; each label is used once. Labels live in eval/labels and
are read only here, never by a detector. By default only events in play are scored
(between a Serve mark and a Point over mark); ALL_EVENTS=1 scores every label.
"""
import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".claude/hillclimb/events"
V, C = "data/video/IpUxCoBHg_g.mp4", "calib/out/t270_calibration.json"
TOL = 5
LIVE = os.environ.get("ALL_EVENTS") != "1"     # default: score only events in play


def labels(clip):
    p = ROOT / "eval/labels/labels" / f"{clip}.json"
    if not p.exists():
        return None
    x = json.loads(p.read_text())
    x = x.get("data", x)
    ev = sorted((e for e in x.get("events", []) if e["kind"] not in ("end", "start")), key=lambda e: e["f"])
    return (ev, live_windows(x.get("events", []), LIVE_FALLBACK.get(clip))) if ev else None


# Points in the clips labelled before the Serve mark existed, read off the labels with Trevor's
# note that he had also marked dead-ball bounces: b_158 opens mid-point; b_404's serve is the far hit at 365.
LIVE_FALLBACK = {"b_158": [(0, 96)], "b_404": [(365, 1239)], "b_1095": [(0, 10**9)]}


def live_windows(events, fallback):
    """[(first, last)] frame ranges in play: Serve mark to Point over mark. A clip may open or end mid-point."""
    marks = sorted((e["f"], e["kind"]) for e in events if e["kind"] in ("start", "end"))
    if not any(k == "start" for _, k in marks) and fallback:
        return fallback
    wins, open_at = [], 0 if (marks and marks[0][1] == "end") or not marks else None
    for f, k in marks:
        if k == "start":
            open_at = f
        elif open_at is not None:
            wins.append((open_at, f)); open_at = None
    if open_at is not None:
        wins.append((open_at, 10**9))
    return wins


def in_play(f, wins):
    return any(a - TOL <= f <= b + TOL for a, b in wins)


def detect_batch(clip, out):
    seg = json.loads((ROOT / f"batch/out/{clip}_segments.json").read_text())
    t, keep = np.array(seg["t"]), np.array(seg["keep"], float)
    ev = []
    for f in sorted(glob.glob(str(ROOT / f"batch/out/{clip}_s*_events.json"))):
        for e in json.loads(Path(f).read_text())["events"]:
            ev.append({"f": float(np.interp(e["t"], t, keep)), "kind": "bounce" if e["kind"] == "bounce" else ("far" if e["player"] >= 2 else "near")})
    return ev, {}


def detect_decode(clip, out):
    o = out / "events" / f"{clip}.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["uv", "run", "python", "fit3d/decode_events.py", V, f"batch/out/{clip}_segments.json",
                        f"batch/out/{clip}_tracks.npz", C, str(o)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    d = json.loads(o.read_text())
    return [{**e, "kind": e["kind"] if e["kind"] == "bounce" else e["side"]} for e in d["events"]], d


def match(pred, gold):
    cand = sorted((abs(p["f"] - g["f"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)
                  if p["kind"] == g["kind"] and abs(p["f"] - g["f"]) <= TOL)
    up, ug, pairs = set(range(len(pred))), set(range(len(gold))), []
    for _, i, j in cand:
        if i in up and j in ug:
            up.discard(i); ug.discard(j); pairs.append((i, j))
    return pairs, sorted(up), sorted(ug)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def main():
    name, det = sys.argv[1], sys.argv[2]
    desc = sys.argv[3] if len(sys.argv) > 3 else det
    out = FLOW / name
    (out / "traces").mkdir(parents=True, exist_ok=True)
    clips = json.loads((ROOT / "eval/clips.json").read_text())["clips"]
    rows, tot = [], np.zeros((3, 3), int)       # rows: all / hits / bounces ; cols tp fp fn
    for c in clips:
        gold = labels(c["id"])
        if gold is None:
            continue
        t0 = time.time()
        gold_all, wins = gold
        pred, raw = (detect_batch if det == "batch" else detect_decode)(c["id"], out)
        if LIVE:
            gold = [e for e in gold_all if in_play(e["f"], wins)]
            pred = [e for e in pred if in_play(e["f"], wins)]
        else:
            gold = gold_all
        pairs, up, ug = match(pred, gold)
        k = np.zeros((3, 3), int)
        for which, items in ((0, [gold[j]["kind"] for _, j in pairs]), (1, [pred[i]["kind"] for i in up]), (2, [gold[j]["kind"] for j in ug])):
            for kind in items:
                k[0, which] += 1
                k[2 if kind == "bounce" else 1, which] += 1
        tot += k
        dt = [pred[i]["f"] - gold[j]["f"] for i, j in pairs]
        grade = {"event_f1": round(f1(*k[0]), 3), "hit_f1": round(f1(*k[1]), 3), "bounce_f1": round(f1(*k[2]), 3),
                 "timing_abs_frames": round(float(np.median(np.abs(dt))), 2) if dt else None}
        rows.append({"prompt_id": c["id"], "rep": 0, "prompt": f"clip {c['id']} ({c['duration_s']} s, {len(gold)} labelled events)",
                     "tags": [c["split"]], "grade": grade, "counts": {"tp": int(k[0, 0]), "false": int(k[0, 1]), "missed": int(k[0, 2])},
                     "latency_s": round(time.time() - t0, 1)})
        lines = [f"clip {c['id']}: tp {k[0, 0]}, false {k[0, 1]}, missed {k[0, 2]}", "MATCHED (pred frame, label frame, kind):"]
        lines += [f"  {pred[i]['f']:7.1f} {gold[j]['f']:5d} {gold[j]['kind']}" for i, j in sorted(pairs, key=lambda p: gold[p[1]]["f"])]
        lines += ["FALSE (predicted, no label):"] + [f"  {json.dumps(pred[i])}" for i in up]
        lines += ["MISSED (label, no prediction):"] + [f"  {gold[j]['f']:5d} {gold[j]['kind']}" for j in ug]
        if raw.get("candidates"):
            lines += ["CANDIDATES:"] + ["  " + json.dumps({a: (round(b, 2) if isinstance(b, float) else b) for a, b in cd.items() if a not in ("vb", "va", "t")}) for cd in raw["candidates"]]
        (out / "traces" / f"{c['id']}_rep0.json").write_text(json.dumps([{"role": "user", "content": rows[-1]["prompt"]}, {"role": "assistant", "content": "\n".join(lines)}], indent=1))
        print(f"  {c['id']:<8} [{c['split']}] event F1 {grade['event_f1']:.2f}  hits {grade['hit_f1']:.2f}  bounces {grade['bounce_f1']:.2f}  tp/false/missed {k[0, 0]}/{k[0, 1]}/{k[0, 2]}")
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps({"description": desc, "target": "code", "pooled": {
        "event_f1": round(f1(*tot[0]), 3), "hit_f1": round(f1(*tot[1]), 3), "bounce_f1": round(f1(*tot[2]), 3),
        "tp": int(tot[0, 0]), "false": int(tot[0, 1]), "missed": int(tot[0, 2])}}, indent=1))
    print(f"{name}: pooled event F1 {f1(*tot[0]):.3f} (hits {f1(*tot[1]):.3f}, bounces {f1(*tot[2]):.3f}); tp {tot[0, 0]}, false {tot[0, 1]}, missed {tot[0, 2]} over {len(rows)} clips")


if __name__ == "__main__":
    main()
