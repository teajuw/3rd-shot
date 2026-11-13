"""Deterministic cut detection and baseline-camera labelling.

Inputs
  signal .npz from signal.py (per-frame change to the previous frame, thumbnail size)
  a calibration .json from calib/calibrate.py (the locked-off baseline camera)

Method
  1. Events: frames whose change exceeds a threshold, grouped when they are within a few
     frames of each other. A 1-2 frame event is a hard cut; a longer one is a transition
     (wipe, dissolve, flash).
  2. Shots: the stretches between events.
  3. Court score (10 samples/s, half-res): fraction of points along the calibrated court
     lines that sit on a bright thin line. High = the baseline camera is on screen.
     This labels each shot and also cross-checks the cuts: a shot whose court score
     flips partway through contains a boundary the change signal missed.

Usage: python detect.py <video> <signal.npz> <calibration.json> <out_prefix>
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calib"))
from calibrate import LINES, project  # noqa: E402

FPS = 60
T_ACTIVE = 8.0      # mean abs RGB change that counts as "something big happened"
T_HIST = 0.15       # or colour-histogram distance above this
T_GROW = 2.5        # once an event is found, extend it outward while change stays above this
GAP = 4             # frames: active frames closer than this belong to one event
MIN_SHOT = 12       # frames: shorter stretches are absorbed into the surrounding event
SCORE_STEP = 6      # court score every 6th frame = 10 Hz
SW, SH = 960, 540
BASELINE_ON = 0.55  # court score above this = baseline camera


def find_events(diff, hist):
    active = np.flatnonzero((diff > T_ACTIVE) | (hist > T_HIST))
    events = []
    for i in active:
        if events and i - events[-1][1] <= GAP:
            events[-1][1] = int(i)
        else:
            events.append([int(i), int(i)])
    # hysteresis: wipes start and end gently, so grow each event while change stays elevated
    for ev in events:
        if ev[1] - ev[0] + 1 <= 2:
            continue  # hard cuts are a single frame; nothing to grow
        # (frames are often duplicated, 30 fps content in a 60 fps file, so look past small gaps)
        while True:
            w = np.flatnonzero(diff[max(1, ev[0] - GAP):ev[0]] > T_GROW)
            if len(w) == 0:
                break
            ev[0] = max(1, ev[0] - GAP) + int(w[0])
        while True:
            w = np.flatnonzero(diff[ev[1] + 1:ev[1] + 1 + GAP] > T_GROW)
            if len(w) == 0:
                break
            ev[1] = ev[1] + 1 + int(w[-1])
    # absorb very short stretches between events
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] - 1 < MIN_SHOT:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def court_sampler(calib):
    e = calib["no_distortion"]
    p = np.r_[e["f_px"], 0.0, e["rvec"], e["tvec"]]
    size = tuple(calib["image_size"])
    on, off_a, off_b = [], [], []
    for a, b in LINES.values():
        a, b = np.array(a, float), np.array(b, float)
        n = int(np.linalg.norm(b - a) / 0.5)
        ss = (np.arange(n) + 0.5) / n
        P = a[None] + ss[:, None] * (b - a)[None]
        uv = project(p, P, size)
        t = project(p, P + (b - a)[None] * 1e-3, size) - uv
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        s = SW / size[0]
        on.append(uv * s)
        off_a.append(uv * s + 4.0 * nrm)
        off_b.append(uv * s - 4.0 * nrm)

    def idx(pts):
        pts = np.concatenate(pts).round().astype(int)
        return np.clip(pts[:, 1], 0, SH - 1), np.clip(pts[:, 0], 0, SW - 1)

    return idx(on), idx(off_a), idx(off_b)


def court_scores(video, calib, cache):
    if cache.exists():
        return np.load(cache)["score"]
    on, oa, ob = court_sampler(calib)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video,
                            "-vf", f"select='not(mod(n\\,{SCORE_STEP}))',setpts=N/TB,scale={SW}:{SH}:flags=area",
                            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, bufsize=1 << 22)
    n = SW * SH * 3
    out = []
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        w = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3).min(axis=2).astype(np.int16)
        contrast = w[on] - np.maximum(w[oa], w[ob])
        out.append(float((contrast > 25).mean()))
    score = np.array(out, np.float32)
    np.savez(cache, score=score)
    return score


def main():
    video, sig_path, calib_path, prefix = sys.argv[1:5]
    z = np.load(sig_path)
    diff, hist = z["diff"], z["hist"]
    n = len(diff)
    calib = json.loads(Path(calib_path).read_text())
    score = court_scores(video, calib, Path(prefix + "_court.npz"))

    events = find_events(diff, hist)
    # shots are the gaps between events
    bounds = [(-1, -1)] + [tuple(e) for e in events] + [(n, n)]
    shots = []
    for (ps, pe), (ns, ne) in zip(bounds[:-1], bounds[1:]):
        s, e = pe + 1, ns - 1          # first and last frame of the shot
        if e - s + 1 < 1:
            continue
        sc = score[int(np.ceil(s / SCORE_STEP)): e // SCORE_STEP + 1]
        if len(sc) == 0:
            sc = score[min(s // SCORE_STEP, len(score) - 1):][:1]
        on = sc > BASELINE_ON
        frac = float(on.mean())
        label = "baseline" if frac >= 0.9 else "other" if frac <= 0.1 else "mixed"
        if e - s + 1 < FPS and label != "baseline":
            label = "short"  # under 1 s and not the baseline camera: wipe graphics, flash cutaways
        shots.append({"start_frame": s, "end_frame": e, "start_s": round(s / FPS, 3), "end_s": round((e + 1) / FPS, 3),
                      "duration_s": round((e - s + 1) / FPS, 3), "label": label,
                      "court_score_median": round(float(np.median(sc)), 3), "baseline_fraction": round(frac, 3),
                      "ends_with": None})
    ev_by_start = {s: (s, e) for s, e in events}
    for sh in shots:
        ev = ev_by_start.get(sh["end_frame"] + 1)
        if ev:
            length = ev[1] - ev[0] + 1
            sh["ends_with"] = "cut" if length <= 2 else f"transition_{length}f"

    out = {"video": video, "fps": FPS, "frames": n,
           "params": {"T_GROW": T_GROW, "T_ACTIVE": T_ACTIVE, "T_HIST": T_HIST, "GAP": GAP, "MIN_SHOT": MIN_SHOT, "BASELINE_ON": BASELINE_ON},
           "events": [{"start_frame": s, "end_frame": e, "start_s": round(s / FPS, 3), "frames": e - s + 1,
                       "kind": "cut" if e - s + 1 <= 2 else "transition"} for s, e in events],
           "shots": shots}
    Path(prefix + "_shots.json").write_text(json.dumps(out, indent=1))
    with open(prefix + "_shots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shots[0].keys()))
        w.writeheader()
        w.writerows(shots)

    kinds = [e["kind"] for e in out["events"]]
    print(f"{n} frames, {len(events)} events: {kinds.count('cut')} hard cuts, {kinds.count('transition')} transitions")
    print(f"{len(shots)} shots")
    for lab in ("baseline", "other", "short", "mixed"):
        ss = [s for s in shots if s["label"] == lab]
        tot = sum(s["duration_s"] for s in ss)
        print(f"  {lab:<9} {len(ss):>4} shots, {tot:7.1f}s ({100 * tot * FPS / n:4.1f}%)"
              + (f", median {np.median([s['duration_s'] for s in ss]):.1f}s, longest {max(s['duration_s'] for s in ss):.1f}s" if ss else ""))
    print("court score distribution (10 Hz):", np.histogram(score, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])[0])
    mixed = [s for s in shots if s["label"] == "mixed"]
    for s in mixed[:25]:
        print(f"  mixed: {s['start_s']:.2f}-{s['end_s']:.2f}s baseline_fraction={s['baseline_fraction']}")


if __name__ == "__main__":
    main()
