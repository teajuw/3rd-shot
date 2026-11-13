"""Pass 1: decode the whole video at thumbnail size and record, per frame, how much it
differs from the previous frame. Output: out/<id>_signal.npz with
  diff  mean absolute RGB difference to the previous frame (0-255)
  hist  chi-square distance between 8x8x8 colour histograms of consecutive frames

Usage: python signal.py <video> <out.npz>
"""
import subprocess
import sys
import time

import numpy as np

TW, TH = 96, 54


def main():
    video, out = sys.argv[1], sys.argv[2]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={TW}:{TH}:flags=area",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=1 << 20)
    n = TW * TH * 3
    prev = prev_h = None
    diffs, hists = [], []
    t0 = time.time()
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        f = np.frombuffer(buf, np.uint8).reshape(TH, TW, 3)
        q = (f >> 5).astype(np.int32)
        h = np.bincount((q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).ravel(), minlength=512).astype(np.float32)
        h /= h.sum()
        if prev is None:
            diffs.append(0.0)
            hists.append(0.0)
        else:
            diffs.append(float(np.abs(f.astype(np.int16) - prev).mean()))
            hists.append(float(((h - prev_h) ** 2 / (h + prev_h + 1e-9)).sum()))
        prev, prev_h = f.astype(np.int16), h
    np.savez(out, diff=np.array(diffs, np.float32), hist=np.array(hists, np.float32))
    print(f"{len(diffs)} frames in {time.time() - t0:.0f}s ({len(diffs) / (time.time() - t0):.0f} fps)")


if __name__ == "__main__":
    main()
