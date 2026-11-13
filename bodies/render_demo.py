"""Render the 3D scene of one rally to a video: court, four bodies, paddles and ball, with a slowly orbiting camera.

Usage: uv run --with pyvista python bodies/render_demo.py <grounded_prefix> <paddle_prefix> <out.mp4> [t0=20] [t1=30] [--view orbit|broadcast|court|side|top] [--still frame.png]
Reads <grounded_prefix>.npz and _verts.npy (bodies/ground.py) and <paddle_prefix>.npz (bodies/paddle.py).
Nothing from the broadcast is drawn: every pixel comes from the reconstruction.
Court coordinates: x across 0..20, y along 0..44 (net at 22), z up, feet.
"""
import argparse
import subprocess

import numpy as np
import pyvista as pv

COL = ["#ffc83c", "#3cc8ff", "#e65ae6", "#5ae678"]
W, H = 1280, 720


def paddle_parts():
    face = pv.Box(bounds=(-0.27, 0.655, -0.026, 0.026, -0.33, 0.33))
    grip = pv.Cylinder(center=(-0.455, 0, 0), direction=(1, 0, 0), radius=0.06, height=0.40, resolution=12)
    return face, grip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("paddle"); ap.add_argument("out")
    ap.add_argument("t0", nargs="?", type=float, default=20.0); ap.add_argument("t1", nargs="?", type=float, default=30.0)
    ap.add_argument("--view", default="orbit", choices=["orbit", "broadcast", "court", "side", "top"])
    ap.add_argument("--still", default=None, help="write one frame from the middle to this PNG and stop")
    a = ap.parse_args()
    g = np.load(a.run + ".npz"); V = np.load(a.run + "_verts.npy", mmap_mode="r")
    pd = np.load(a.paddle + ".npz", allow_pickle=True)
    fps = float(g["fps"]); roots = g["root_world"]; ball = pd["ball"]
    f0, f1 = int(a.t0 * fps), min(len(V), int(a.t1 * fps))
    faces = np.hstack([np.full((len(g["faces"]), 1), 3), g["faces"]]).astype(np.int64).ravel()

    pl = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
    pl.set_background("#10161b")
    pl.add_light(pv.Light(position=(-20, -30, 60), focal_point=(10, 22, 0), intensity=0.9, positional=False))
    pl.add_light(pv.Light(position=(40, 60, 40), focal_point=(10, 22, 0), intensity=0.35, positional=False))
    pl.add_light(pv.Light(light_type="headlight", intensity=0.25))
    flat = lambda x0, y0, x1, y1, z, c: pl.add_mesh(pv.Box(bounds=(x0, x1, y0, y1, z - 0.01, z)), color=c, lighting=False)  # noqa: E731
    flat(-25, -28, 45, 72, -0.02, "#1d2a44"); flat(0, 0, 20, 44, 0, "#2f5fb3"); flat(0, 15, 20, 29, 0.002, "#3a6fc9")
    for x0, y0, x1, y1 in [(0, 0, 20, 0), (0, 44, 20, 44), (0, 0, 0, 44), (20, 0, 20, 44), (0, 15, 20, 15), (0, 29, 20, 29), (10, 0, 10, 15), (10, 29, 10, 44)]:
        flat(x0 - 0.085, y0 - 0.085, x1 + 0.085, y1 + 0.085, 0.012, "white")
    # net: 36 in at the posts, 34 in at the middle
    xs = np.linspace(-1, 21, 23); top = (34 + 2 * np.abs(xs - 10) / 11) / 12
    pts = np.vstack([np.c_[xs, np.full(23, 22.0), np.zeros(23)], np.c_[xs, np.full(23, 22.0), top]])
    quads = np.hstack([[4, i, i + 1, 23 + i + 1, 23 + i] for i in range(22)])
    pl.add_mesh(pv.PolyData(pts, quads), color="white", opacity=0.22, lighting=False)
    pl.add_mesh(pv.Spline(pts[23:], 60).tube(radius=0.07), color="white", lighting=False)
    for x in (-1, 21):
        pl.add_mesh(pv.Cylinder(center=(x, 22, 1.5), direction=(0, 0, 1), radius=0.09, height=3.0), color="#cfd6dc")

    bodies, shadows, faces_p, grips = [], [], [], []
    for p in range(4):
        m = pv.PolyData(np.asarray(V[f0, p], np.float32) + roots[f0, p], faces)
        pl.add_mesh(m, color=COL[p], smooth_shading=True, specular=0.15, specular_power=12, diffuse=0.85, ambient=0.25)
        s = pv.PolyData(m.points.copy(), faces)
        pl.add_mesh(s, color="black", opacity=0.28, lighting=False)
        fa, gr = paddle_parts()
        pl.add_mesh(fa, color="#22282d", specular=0.2); pl.add_mesh(gr, color="#f0f0f0")
        bodies.append(m); shadows.append(s); faces_p.append((fa, fa.points.copy())); grips.append((gr, gr.points.copy()))
    bm = pv.Sphere(radius=0.2, theta_resolution=16, phi_resolution=12); b0 = bm.points.copy()
    ball_actor = pl.add_mesh(bm, color="#d8f03a", lighting=False)
    bs = pv.Disc(inner=0, outer=0.2, c_res=16); s0 = bs.points.copy()
    bshadow = pl.add_mesh(bs, color="black", opacity=0.35, lighting=False)
    trail = pv.PolyData(np.zeros((2, 3))); trail.lines = np.array([2, 0, 1])
    trail_actor = pl.add_mesh(trail, color="#d8f03a", opacity=0.55, line_width=3, lighting=False)

    def draw(f, u):
        for p in range(4):
            pts = np.asarray(V[f, p], np.float32) + roots[f, p]
            bodies[p].points = pts
            bodies[p].compute_normals(inplace=True, cell_normals=False)
            sh = pts.copy(); sh[:, 0] += 0.25 * sh[:, 2]; sh[:, 1] += 0.2 * sh[:, 2]; sh[:, 2] = 0.02     # shadow from the main light
            shadows[p].points = sh
            c, ax, n = pd["centre"][f, p], pd["axis"][f, p], pd["normal"][f, p]
            ax = ax / np.linalg.norm(ax); n = n - ax * (n @ ax); n = n / np.linalg.norm(n)
            R = np.stack([ax, n, np.cross(ax, n)], 1)
            for mesh, rest in (faces_p[p], grips[p]):
                mesh.points = rest @ R.T + c
        b = ball[f]; ok = not np.isnan(b[0])
        ball_actor.SetVisibility(ok); bshadow.SetVisibility(ok)
        if ok:
            bm.points = b0 + b
            bs.points = s0 + [b[0] + 0.25 * b[2], b[1] + 0.2 * b[2], 0.025]
        tr = ball[max(f0, f - 14):f + 1]; tr = tr[~np.isnan(tr[:, 0])]
        trail_actor.SetVisibility(ok and len(tr) > 1)
        if ok and len(tr) > 1:
            trail.points = tr; trail.lines = np.hstack([[len(tr)], np.arange(len(tr))])
        if a.view == "orbit":                             # from behind one corner across the near baseline
            az = np.radians(-150 + 80 * u)
            r, hgt = 46 - 5 * np.sin(np.pi * u), 15 + 3 * np.sin(np.pi * u)
            pos, foc, up, fov = (10 + r * np.sin(az), 22 + r * np.cos(az), hgt), (10, 22, 3.0), (0, 0, 1), 30
        elif a.view == "broadcast":                       # where the broadcast camera stands
            pos, foc, up, fov = (9.84, -90.6, 29.4), (10, 22, 2), (0, 0, 1), 11.3
        elif a.view == "court":                           # eye height behind the near baseline, drifting sideways
            pos, foc, up, fov = (4 + 12 * u, -14, 5.5), (10, 24, 3.2), (0, 0, 1), 38
        elif a.view == "side":                            # level with the net, low
            pos, foc, up, fov = (-38, 22 - 6 + 12 * u, 6), (10, 22, 3), (0, 0, 1), 30
        else:                                             # top
            pos, foc, up, fov = (10, 22, 95), (10, 22, 0), (1, 0, 0), 30
        pl.camera.position = pos; pl.camera.focal_point = foc; pl.camera.up = up; pl.camera.view_angle = fov
        pl.camera.clipping_range = (1, 400)
        pl.render()
        return pl.screenshot(None, return_img=True)

    if a.still:
        import PIL.Image
        fm = (f0 + f1) // 2
        PIL.Image.fromarray(draw(fm, 0.5)).save(a.still); print("wrote", a.still); return
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        ff.stdin.write(np.ascontiguousarray(draw(f, (f - f0) / max(1, f1 - f0 - 1))[:, :, :3]).tobytes())
    ff.stdin.close(); ff.wait()
    print("wrote", a.out, f1 - f0, "frames at", fps, "fps")


if __name__ == "__main__":
    main()
