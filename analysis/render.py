"""A small painter's-algorithm renderer on top of matplotlib: shaded tubes around space curves and
coloured spheres, drawn as depth-sorted polygons (rasterised inside an otherwise vector PDF)."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import PolyCollection


class Camera:
    """Orthographic camera looking at the origin from azimuth/elevation (degrees)."""

    def __init__(self, azim=30.0, elev=25.0, light=(-0.45, 0.75, 0.6)):
        a, e = np.radians(azim), np.radians(elev)
        self.v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # towards the viewer
        f = -self.v
        self.right = np.cross(f, [0.0, 0.0, 1.0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, f)
        L = light[0] * self.right + light[1] * self.up + light[2] * self.v
        self.L = L / np.linalg.norm(L)

    def project(self, P):
        """P (..., 3) -> screen coordinates (..., 2) and depth (...) (larger = nearer)."""
        return np.stack([P @ self.right, P @ self.up], axis=-1), P @ self.v


class Scene:
    def __init__(self):
        self.quads, self.normals, self.colors, self.shine = [], [], [], []

    def add(self, quads, normals, colors, shine=1.0):
        self.quads.append(quads)
        self.normals.append(normals)
        self.colors.append(np.broadcast_to(colors, (len(quads), 3)))
        self.shine.append(np.full(len(quads), shine))

    def add_tube(self, P, radius, color, sides=12, shine=1.0, closed=True):
        """Tube of the given radius around the curve P (N, 3); closed curves without repeated endpoint."""
        q, n = tube_mesh(np.asarray(P, dtype=float), radius, sides, closed)
        self.add(q, n, np.asarray(color, dtype=float), shine)

    def add_surface(self, X, colors, shine=0.3):
        """Quadrilateral mesh of a closed surface given by vertices X (I, J, 3), periodic in J;
        colors (I-1, J, 3) per face; normals point away from the origin (for spheres)."""
        A, B = X[:-1], X[1:]
        C, D = np.roll(B, -1, axis=1), np.roll(A, -1, axis=1)
        quads = np.stack([A, B, C, D], axis=2).reshape(-1, 4, 3)
        cen = quads.mean(axis=1)
        nrm = cen / np.linalg.norm(cen, axis=1, keepdims=True)
        self.add(quads, nrm, colors.reshape(-1, 3), shine)

    def draw(self, ax, cam, cull=True, ambient=0.32, diffuse=0.68, specular=0.35, power=30, lw=0.25):
        Q = np.concatenate(self.quads)
        N = np.concatenate(self.normals)
        C = np.concatenate(self.colors)
        S = np.concatenate(self.shine)
        if cull:
            keep = N @ cam.v > -1e-3
            Q, N, C, S = Q[keep], N[keep], C[keep], S[keep]
        xy, depth = cam.project(Q)
        order = np.argsort(depth.mean(axis=1))
        xy, N, C, S = xy[order], N[order], C[order], S[order]
        dif = np.clip(N @ cam.L, 0, 1)
        h = cam.L + cam.v
        h /= np.linalg.norm(h)
        spec = np.clip(N @ h, 0, 1) ** power
        rgb = np.clip(C * (ambient + diffuse * dif)[:, None] + (specular * S * spec)[:, None], 0, 1)
        pc = PolyCollection(xy, facecolors=rgb, edgecolors=rgb, linewidths=lw, rasterized=True)
        ax.add_collection(pc)
        lo, hi = xy.reshape(-1, 2).min(axis=0), xy.reshape(-1, 2).max(axis=0)
        return lo, hi


def tube_mesh(P, radius, sides=12, closed=True):
    """Quads and outward normals of a tube around a curve, with a parallel-transported frame (for a
    closed curve the closing twist is spread evenly along the curve; an open tube has no caps)."""
    Nn = len(P)
    T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    if not closed:
        T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    n = np.empty_like(P)
    n[0] = a - (a @ T[0]) * T[0]
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, Nn):
        v = n[i - 1] - (n[i - 1] @ T[i]) * T[i]
        n[i] = v / np.linalg.norm(v)
    nN = n[-1] - (n[-1] @ T[0]) * T[0]
    nN /= np.linalg.norm(nN)
    twist = np.arctan2(np.cross(nN, n[0]) @ T[0], nN @ n[0]) if closed else 0.0
    ang = twist * np.arange(Nn) / Nn
    b = np.cross(T, n)
    n, b = np.cos(ang)[:, None] * n + np.sin(ang)[:, None] * b, -np.sin(ang)[:, None] * n + np.cos(ang)[:, None] * b
    beta = 2 * np.pi * np.arange(sides) / sides
    R = np.cos(beta)[None, :, None] * n[:, None, :] + np.sin(beta)[None, :, None] * b[:, None, :]   # (N, s, 3)
    V = P[:, None, :] + radius * R
    i1, k1 = np.roll(np.arange(Nn), -1), np.roll(np.arange(sides), -1)
    quads = np.stack([V, V[i1], V[i1][:, k1], V[:, k1]], axis=2)
    nrm = R + R[i1] + R[i1][:, k1] + R[:, k1]
    if not closed:
        quads, nrm = quads[:-1], nrm[:-1]
    quads, nrm = quads.reshape(-1, 4, 3), nrm.reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return quads, nrm


def sphere_vertices(x, phi):
    """Vertices of the unit sphere with heights x (I,) and longitudes phi (J,): shape (I, J, 3)."""
    r = np.sqrt(np.clip(1 - x ** 2, 0, 1))
    return np.stack([r[:, None] * np.cos(phi)[None, :], r[:, None] * np.sin(phi)[None, :],
                     np.broadcast_to(x[:, None], (len(x), len(phi)))], axis=-1)


def add_cap(scene, y, color, angle=0.13, lift=1.01, shine=0.2):
    """A small coloured spherical cap of angular radius ``angle`` centred at the unit vector y
    (a marker on a unit sphere)."""
    y = np.asarray(y, dtype=float)
    V = sphere_vertices(np.linspace(np.cos(angle), 1, 4), np.linspace(0, 2 * np.pi, 12, endpoint=False)) * lift
    e1 = np.cross([0, 0, 1.0], y) if abs(y[2]) < 0.99 else np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(y, e1)
    Vr = V[..., :1] * e1 + V[..., 1:2] * e2 + V[..., 2:] * y
    scene.add_surface(Vr, np.broadcast_to(np.asarray(color, dtype=float), (3, 12, 3)).copy(), shine=shine)


def pole_marker(ax, cam, y, color, size=4.0):
    """A vector marker (dot with dark rim) at the point y of the unit sphere, if it faces the camera."""
    y = np.asarray(y, dtype=float)
    if y @ cam.v > -1e-9:                    # visible, or on the outline
        xy, _ = cam.project(y)
        ax.plot([xy[0]], [xy[1]], "o", ms=size, mfc=color, mec=(0.1, 0.1, 0.1), mew=0.6, zorder=5)


def finish(ax, lo, hi, pad=0.04):
    c, h = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + pad)
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_aspect("equal")
    ax.axis("off")


def resample_closed(P, n):
    """Resample a closed polygon P (3, N) to n points equally spaced in arc length."""
    Q = np.concatenate([P, P[:, :1]], axis=1)
    seg = np.linalg.norm(np.diff(Q, axis=1), axis=0)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, Q[c]) for c in range(3)])
