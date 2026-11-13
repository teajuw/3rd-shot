"""Front stages only (players, ball, segmentation, event decoding) on every baseline-camera shot of the match.

Usage: uv run python batch/run_front.py [min_seconds=4]
Resumable: a shot whose events file exists is skipped. Outputs go to batch/out/<tag>_*, tag = b_<start second>.
No 3D ball fit is run; the shot-target dataset does not need one.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V, C = "data/video/IpUxCoBHg_g.mp4", "calib/out/t270_calibration.json"


def sh(args, log):
    r = subprocess.run(["uv", "run", "python", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "NO_RENDER": "1"})
    log.write(f"\n$ {' '.join(args)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
    return r.returncode == 0


def main():
    min_s = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    shots = json.loads((ROOT / "cuts/out/IpUxCoBHg_g_shots.json").read_text())["shots"]
    pool = sorted([s for s in shots if s["label"] == "baseline" and s["duration_s"] >= min_s], key=lambda s: s["start_s"])
    out = ROOT / "batch/out"
    status = out / "front_status.jsonl"
    for k, s in enumerate(pool):
        T = f"b_{int(s['start_s'])}"
        ev = out / f"{T}_decoded.json"
        if ev.exists():
            continue
        S, N = s["start_frame"] + 1, s["end_frame"] - s["start_frame"] - 2
        t0 = time.time(); ok = True
        with open(out / f"{T}.front.log", "w") as log:
            if not (out / f"{T}_tracks.npz").exists():
                ok = ok and sh(["players/track_players.py", V, C, str(S), str(N), f"batch/out/{T}"], log)
            if ok and not (out / f"{T}_fused_track.npz").exists():
                ok = sh(["ball/candidates.py", V, str(S), str(N), f"batch/out/{T}_cands.npz"], log) and sh(["ball/track_ball.py", V, f"batch/out/{T}_cands.npz", f"batch/out/{T}", "fused"], log)
            if ok and not (out / f"{T}_segments.json").exists():
                ok = sh(["fit3d/segment.py", f"batch/out/{T}_fused_track.npz", f"batch/out/{T}_tracks.npz", "cuts/out/IpUxCoBHg_g_signal.npz", f"batch/out/{T}"], log)
            if ok:
                ok = sh(["fit3d/decode_events.py", V, f"batch/out/{T}_segments.json", f"batch/out/{T}_tracks.npz", C, str(ev)], log)
        with open(status, "a") as f:
            f.write(json.dumps({"tag": T, "start_s": s["start_s"], "duration_s": s["duration_s"], "ok": bool(ok), "seconds": round(time.time() - t0)}) + "\n")
        print(k + 1, len(pool), T, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
