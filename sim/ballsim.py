"""Event-based pickleball ball simulator (feet, seconds, court frame: x across, y along, z up).

Flight   gravity, quadratic drag, and lift from spin (Magnus) with the spin held constant in flight.
Bounce   an impulse at the floor: restitution that falls with impact speed, and Coulomb friction that
         either slides or grips (thin hollow shell, moment of inertia 2/3 m r^2), which also changes spin.
Net      a flight that crosses the net plane below the tape stops there.
Paddle   an impulse against a moving flat face with its own restitution and friction.

Contacts are handled as events (find the crossing time, apply the impulse, carry on), not by
time-stepping a soft contact, so the time step can stay coarse.
"""
from dataclasses import dataclass, replace

import numpy as np

G = 32.174
R_BALL = 0.121            # ft
NET_Y = 22.0
ALPHA = 2.0 / 3.0         # I / (m r^2) for a thin shell
AREA_OVER_2M = 0.0106 / 0.33     # rho * A / (2 m), per foot (outdoor ball: 74 mm, 24.5 g, air 1.2 kg/m^3)


@dataclass(frozen=True)
class Params:
    cd: float = 0.33          # drag coefficient
    cl_slope: float = 0.195   # lift coefficient = cl_slope * (r * spin / speed)   (Steyn et al.)
    e0: float = 0.66          # floor restitution at zero impact speed
    e1: float = 0.0           # loss of restitution per ft/s of impact speed
    mu: float = 0.4           # floor friction
    kt: float = -1.0          # if >= 0: ignore friction and scale tangential speed by kt (the old constant model)
    e_paddle: float = 0.43
    mu_paddle: float = 0.4

    def with_(self, **kw):
        return replace(self, **kw)


def accel(v, w, p):
    a = -AREA_OVER_2M * p.cd * np.sqrt(v @ v) * v
    if w is not None:
        a = a + AREA_OVER_2M * p.cl_slope * R_BALL * np.cross(w, v)
    a[2] -= G
    return a


def flight(p0, v0, T, p=Params(), w=None, h=0.004):
    """Integrate for T seconds. Returns times (n,), positions (n,3), velocities (n,3)."""
    n = max(2, int(np.ceil(T / h)) + 2)
    P = np.empty((n, 3)); V = np.empty((n, 3))
    x, v = np.array(p0, float), np.array(v0, float)
    for i in range(n):
        P[i], V[i] = x, v
        k1 = accel(v, w, p); k2 = accel(v + 0.5 * h * k1, w, p); k3 = accel(v + 0.5 * h * k2, w, p); k4 = accel(v + h * k3, w, p)
        x = x + h * (v + h / 6 * (k1 + k2 + k3))
        v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.arange(n) * h, P, V


def at(ts, A, t):
    return np.stack([np.interp(t, ts, A[:, k]) for k in range(3)], -1)


def impact(v, w, n, e, mu, kt=-1.0):
    """Impulse against a fixed surface with unit normal n. v, w are the ball's velocity and spin before; returns after."""
    w = np.zeros(3) if w is None else np.asarray(w, float)
    vn = v @ n
    vt = v - vn * n
    jn = (1 + e) * abs(vn)                       # normal impulse per unit mass
    if kt >= 0:
        return kt * vt - e * vn * n, w
    u = vt + np.cross(w, -R_BALL * n)            # velocity of the contact point along the surface
    dv = -u * ALPHA / (1 + ALPHA)                # what would bring the contact point to rest (grip)
    lim = mu * jn
    if np.linalg.norm(dv) > lim:                 # not enough friction: slide
        dv = -lim * u / max(np.linalg.norm(u), 1e-9)
    dw = np.cross(-R_BALL * n, dv) / (ALPHA * R_BALL ** 2)
    return vt + dv - e * vn * n, w + dw


def bounce(v, w, p=Params()):
    e = float(np.clip(p.e0 - p.e1 * abs(v[2]), 0.2, 0.95))
    return impact(np.asarray(v, float), w, np.array([0.0, 0.0, 1.0]), e, p.mu, p.kt)


def paddle_hit(v, w, normal, v_paddle, p=Params()):
    """Ball leaving a flat paddle face moving at v_paddle (velocity of the face at the contact point)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    rel = np.asarray(v, float) - np.asarray(v_paddle, float)
    if rel @ n > 0:
        n = -n                                   # use the side the ball is arriving on
    out, w2 = impact(rel, w, n, p.e_paddle, p.mu_paddle)
    return out + np.asarray(v_paddle, float), w2


def net_height(x):
    return (34 + 2 * abs(x - 10) / 10) / 12      # 34 in at the centre, 36 in at the posts


def simulate(p0, v0, T, p=Params(), w=None, h=0.004, max_bounces=6):
    """Fly, bounce and stop at the net. Returns times, positions, and a list of events (kind, time, position)."""
    T_all, P_all, events = [], [], []
    x, v, t0 = np.array(p0, float), np.array(v0, float), 0.0
    for _ in range(max_bounces + 1):
        ts, P, V = flight(x, v, T - t0, p, w, h)
        below = np.flatnonzero((P[1:, 2] < R_BALL) & (V[1:, 2] < 0))
        cross = np.flatnonzero(np.sign(P[:-1, 1] - NET_Y) != np.sign(P[1:, 1] - NET_Y))
        k_b = below[0] + 1 if len(below) else None
        k_n = None
        for k in cross:
            f = (NET_Y - P[k, 1]) / (P[k + 1, 1] - P[k, 1])
            q = P[k] + f * (P[k + 1] - P[k])
            if q[2] - R_BALL < net_height(q[0]) and -1 <= q[0] <= 21 and (k_b is None or k < k_b):
                k_n = (k, f, q); break
        if k_n is not None:
            k, f, q = k_n
            T_all.append(t0 + ts[:k + 1]); P_all.append(P[:k + 1]); events.append(("net", t0 + ts[k] + f * h, q))
            break
        if k_b is None:
            T_all.append(t0 + ts); P_all.append(P); break
        f = (P[k_b - 1, 2] - R_BALL) / (P[k_b - 1, 2] - P[k_b, 2])
        q = P[k_b - 1] + f * (P[k_b] - P[k_b - 1]); vq = V[k_b - 1] + f * (V[k_b] - V[k_b - 1])
        T_all.append(t0 + ts[:k_b]); P_all.append(P[:k_b])
        t0 = t0 + ts[k_b - 1] + f * h
        events.append(("bounce", t0, q))
        v, w = bounce(vq, w, p)
        x = np.array([q[0], q[1], R_BALL])
        if t0 >= T or abs(v[2]) < 0.5:
            break
    return np.concatenate(T_all), np.concatenate(P_all), events
