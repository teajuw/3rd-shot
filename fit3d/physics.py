"""Ball flight model and camera projection (feet, seconds).

Flight: gravity plus quadratic drag, no spin. Drag constant from the research notes:
Cd about 0.33 for the outdoor 40-hole ball, 74 mm diameter, 24.5 g, air 1.2 kg/m^3
-> k = rho*Cd*A/(2m) = 0.035 per metre = 0.0106 per foot.
"""
import cv2
import numpy as np

G = 32.174
K_DRAG = 0.0106
R_BALL = 0.121          # ft (74 mm diameter)
NET_Y = 22.0
SUB = 2                 # RK4 substeps per 50 Hz sample


def _acc(v):
    a = -K_DRAG * np.linalg.norm(v) * v
    a[2] -= G
    return a


def simulate(p0, v0, n, dt):
    """Positions at n samples spaced dt, starting at p0 with velocity v0. Returns (n,3), (n,3)."""
    P = np.empty((n, 3))
    V = np.empty((n, 3))
    p, v = np.array(p0, float), np.array(v0, float)
    h = dt / SUB
    for i in range(n):
        P[i], V[i] = p, v
        for _ in range(SUB):
            k1v = _acc(v)
            k2v = _acc(v + 0.5 * h * k1v)
            k3v = _acc(v + 0.5 * h * k2v)
            k4v = _acc(v + h * k3v)
            p = p + h * (v + h / 6 * (k1v + k2v + k3v))
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return P, V


class Camera:
    def __init__(self, calib):
        e = calib["no_distortion"]
        self.f = e["f_px"]
        self.R, _ = cv2.Rodrigues(np.array(e["rvec"], float))
        self.t = np.array(e["tvec"], float)
        self.cx, self.cy = calib["image_size"][0] / 2, calib["image_size"][1] / 2
        self.C = -self.R.T @ self.t

    def project(self, P):
        c = P @ self.R.T + self.t
        return np.stack([self.f * c[:, 0] / c[:, 2] + self.cx, self.f * c[:, 1] / c[:, 2] + self.cy], 1)

    def ray(self, px):
        d = self.R.T @ np.array([(px[0] - self.cx) / self.f, (px[1] - self.cy) / self.f, 1.0])
        return d / np.linalg.norm(d)

    def at_depth(self, px, y):
        """Point on the pixel's ray at court coordinate Y = y."""
        d = self.ray(px)
        return self.C + d * (y - self.C[1]) / d[1]

    def on_plane(self, px, z):
        d = self.ray(px)
        return self.C + d * (z - self.C[2]) / d[2]
