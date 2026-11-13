"""Run the whole pipeline, unchanged and with no hand steps, on a random sample of
baseline-camera shots, and collect what each stage produced.

For each shot: players -> ball candidates -> ball track -> segmentation -> for every
stretch of ball visibility of 3 s or more: automatic events -> 3D fit.

Usage: python run_many.py <n_shots> <seed>
"""
import json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V, C = "data/video/IpUxCoBHg_g.mp4", "calib/out/t270_calibration.json"
USED = (267.25, 1021.15)     # shots already looked at

def sh(args, log):
    r = subprocess.run(["uv", "run", "python", *args], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "NO_RENDER": "1"})
    log.write(f"\n$ {' '.join(args)}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}\n")
    return r.returncode == 0, r.stdout

def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    shots = json.loads((ROOT / "cuts/out/IpUxCoBHg_g_shots.json").read_text())["shots"]
    pool = [s for s in shots if s["label"] == "baseline" and 8 <= s["duration_s"] <= 35
            and all(abs(s["start_s"] - u) > 1 for u in USED)]
    random.seed(seed)
    pick = sorted(random.sample(pool, n), key=lambda s: s["start_s"])
    out = ROOT / "batch/out"
    results = []
    for k, s in enumerate(pick):
        T = f"b_{int(s['start_s'])}"
        S, N = s["start_frame"] + 1, s["end_frame"] - s["start_frame"] - 2
        rec = {"tag": T, "start_s": s["start_s"], "duration_s": s["duration_s"], "frames": N, "stages": {}, "stretches": []}
        t0 = time.time()
        with open(out / f"{T}.log", "w") as log:
            ok, o = sh(["players/track_players.py", V, C, str(S), str(N), f"batch/out/{T}"], log)
            rec["stages"]["players"] = ok
            if ok:
                for line in o.splitlines():
                    if "initialised" in line:
                        rec["player_init_frame"] = int(line.split()[-1])
            ok2, _ = sh(["ball/candidates.py", V, str(S), str(N), f"batch/out/{T}_cands.npz"], log)
            ok2 = ok2 and sh(["ball/track_ball.py", V, f"batch/out/{T}_cands.npz", f"batch/out/{T}", "fused"], log)[0]
            rec["stages"]["ball"] = ok2
            if ok2:
                rec["ball"] = json.loads((out / f"{T}_fused_stats.json").read_text())
            ok3 = ok and ok2 and sh(["fit3d/segment.py", f"batch/out/{T}_fused_track.npz", f"batch/out/{T}_tracks.npz",
                                     "cuts/out/IpUxCoBHg_g_signal.npz", f"batch/out/{T}"], log)[0]
            rec["stages"]["segment"] = ok3
            if ok3:
                seg = json.loads((out / f"{T}_segments.json").read_text())
                tt = seg["t"]
                for j, (a, b) in enumerate(seg["stretches"]):
                    ta, tb = tt[a], tt[b]
                    if tb - ta < 3.0:
                        continue
                    st = {"t0": ta, "t1": tb}
                    tag = f"batch/out/{T}_s{j}"
                    okE, _ = sh(["fit3d/auto_events.py", V, f"batch/out/{T}_segments.json", f"batch/out/{T}_tracks.npz", C,
                                 f"{ta:.2f}", f"{tb:.2f}", tag + "_events.json"], log)
                    ev = json.loads((ROOT / (tag + "_events.json")).read_text())["events"] if okE else []
                    st["hits"] = sum(e["kind"] == "hit" for e in ev)
                    st["bounces"] = sum(e["kind"] == "bounce" for e in ev)
                    if okE and len(ev) >= 3:
                        okF, _ = sh(["fit3d/fit.py", f"batch/out/{T}_segments.json", tag + "_events.json",
                                     f"batch/out/{T}_tracks.npz", C, tag], log)
                        st["fit_ok"] = okF
                        if okF:
                            st["fit"] = tag + "_fit.json"
                    rec["stretches"].append(st)
        rec["seconds"] = round(time.time() - t0)
        results.append(rec)
        (out / "results.json").write_text(json.dumps(results, indent=1))
        print(f"[{k + 1}/{n}] {T} {s['duration_s']:.1f}s -> {rec['seconds']}s, stages {rec['stages']}, stretches {len(rec['stretches'])}", flush=True)

if __name__ == "__main__":
    main()
