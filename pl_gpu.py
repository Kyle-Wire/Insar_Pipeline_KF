#!/usr/bin/env python3
"""
GPU phase linking of MiaplPy patches (speed plan Phase 2,
review_2026-11-10/PLAN.md; 11 Nov 2026).

The same per-pixel algorithm as our patched MiaplPy build's `process_patch_c`
(utils.pyx, overlap-datum + speed patches), batched over pixels on the GPU with
JAX. It writes the same files into <work>/inverted/PATCHES/PATCH_<i>/
(phase_ref, shp, tempCoh, mask_ps, ps_products, then flag), so MiaplPy's
concatenation, resume and everything downstream are unchanged. A patch it
writes also gets kernel.json naming this kernel.

Runs in the `dolphin` conda env (CUDA JAX); it imports nothing from the
pipeline package. Per pixel, as MiaplPy does it:

  SHPs      two-sample KS test of the pixel's amplitudes against each
            neighbour in its window (distance <= MiaplPy's ks_lut threshold),
            then the 8-connected group of similar neighbours holding the pixel
  coherence the full N x N correlation of the SHPs (est_corr: zero covariance
            -> zero correlation, as divide_elementwise does)
  > ps_shp SHPs   overlapping-window sequential EMI (windows of ministack
            images plus `lag` shared images, the last one takes the rest;
            regularise |C| by doubling diagonal loading from 1e-6 up to 100
            Cholesky tries, else EVD), each window joined to the previous by
            the circular mean phase difference on the shared images;
            tempCoh[0] = mean window PTA coherence, tempCoh[1] = full-stack PTA
            coherence, amplitude = mean SHP amplitude
  <= ps_shp SHPs  MiaplPy's PS test on the full matrix: top eigenvalue share >
            95 % and amplitude dispersion < 0.42 -> PS (own phase), else the
            top eigenvector. The matrix has rank <= ps_shp, so its eigenpairs
            come from the ps_shp x ps_shp Gram matrix (same eigenvalues, the
            eigenvector mapped back), not from an N x N eigendecomposition
  masked    (mask 0) tempCoh 0.1, SHP 1, the pixel's own phase

An angle of NaN counts as 0, as cargf_r does. Only sequential_EMI (and EMI,
which MiaplPy switches to below 10 images) is implemented; any other method
is refused so the caller keeps the CPU route.

    python pl_gpu.py run --slc-stack .../inputs/slcStack.h5 --out-dir .../inverted \
        --range-window 15 --azimuth-window 15 --patch-size 200 --method sequential_EMI \
        --ministack 10 --lag 3 --ps-shp 10 --mask .../inputs/skipMask.h5
    python pl_gpu.py probe        # JSON: is a CUDA device usable?
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # the GPU is shared with the Windows desktop

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np

KERNEL = "pl_gpu"
KERNEL_VERSION = 1          # bump on any change that can move a result
SUPPORTED_METHODS = ("sequential_EMI", "EMI")
ALPHA = 0.01                # MiaplPy's KS significance (CPhaseLink.__init__)
PATCH_FILES = ("phase_ref", "shp", "tempCoh", "mask_ps", "ps_products")


def kernel_label() -> str:
    return f"{KERNEL}-{KERNEL_VERSION}"


# ------------------------------------------------------------ MiaplPy's setup, ported exactly

def ks_threshold(n_image: int, alpha: float = ALPHA) -> float:
    """utils.ks_lut_cy(n, n, alpha) with its C float/double steps."""
    f32 = np.float32
    N = f32((n_image * n_image) / (n_image + n_image))
    sq = math.sqrt(float(N))
    a = f32(alpha)
    for d in np.arange(0.01, 1, 0.001, dtype=np.float32):
        value = f32(float(d) * (sq + 0.12 + 0.11 / sq))
        p = f32(0)
        for t in range(1, 101):
            p = f32(float(p) + ((-1) ** (t - 1)) * math.exp(-2 * (float(value) ** 2) * (t ** 2)))
        p = f32(2 * p)
        p = min(max(p, f32(0)), f32(1))
        if p <= a:
            return float(d)
    return 0.1


def ks_max_count(n_image: int, threshold: float) -> int:
    """Largest integer ECDF gap k with float32(k / n) <= threshold: MiaplPy's
    `abs(temp1 - temp2) / nx` stored in a float, compared with `<=`."""
    k = np.arange(n_image + 1)
    ok = (k / n_image).astype(np.float32) <= np.float32(threshold)
    return int(k[ok].max())


def patch_boxes(length: int, width: int, patch_size: int, range_window: int, azimuth_window: int) -> list:
    """CPhaseLink.patch_slice: (col0, row0, col1, row1, index), the last
    row/column of patches taking the remainder."""
    r1 = np.arange(0, length - azimuth_window, patch_size, dtype=np.int32)
    c1 = np.arange(0, width - range_window, patch_size, dtype=np.int32)
    r2, c2 = r1 + patch_size, c1 + patch_size
    r2[-1], c2[-1] = length, width
    out, idx = [], 0
    for i in range(r1.size):
        for t in range(c1.size):
            out.append((int(c1[t]), int(r1[i]), int(c2[t]), int(r2[i]), idx))
            idx += 1
    return out


def windows(n_image: int, method: str, ministack: int, lag: int) -> list:
    """(start, first, last) of each phase-linking window: images [start, last)
    are linked, [first, last) are kept (sequential_phase_linking_cy)."""
    if method == "EMI":
        return [(0, 0, n_image)]
    k = n_image // ministack
    out = []
    for s in range(k):
        first = s * ministack
        last = n_image if s == k - 1 else first + ministack
        a = 0 if s == 0 or first - lag < 0 else first - lag
        out.append((a, first, last))
    return out


# ------------------------------------------------------------ GPU kernels

def _jax():
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")   # never TF32: float32 like LAPACK/BLAS
    return jax


def probe() -> dict:
    """Is a CUDA device usable here? Never raises."""
    try:
        jax = _jax()
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            return {"gpu": False, "reason": "JAX sees no GPU", "devices": [str(d) for d in jax.devices()]}
        import jax.numpy as jnp
        x = jnp.ones((64, 64), jnp.complex64)
        float(jnp.abs((x @ x.conj().T)[0, 0]))
        stats = {}
        try:
            stats = devs[0].memory_stats() or {}
        except Exception:                                           # noqa: BLE001
            pass
        return {"gpu": True, "device": str(devs[0]), "kind": getattr(devs[0], "device_kind", ""),
                "jax": jax.__version__, "bytes_limit": stats.get("bytes_limit")}
    except Exception as exc:                                        # noqa: BLE001
        return {"gpu": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}


class Kernels:
    """The jitted pieces for one stack shape (n_image, windows, ps_shp)."""

    def __init__(self, n_image: int, method: str, ministack: int, lag: int, ps_shp: int,
                 range_window: int, azimuth_window: int):
        jax = _jax()
        import jax.numpy as jnp
        from jax import lax
        self.jax, self.jnp, self.lax = jax, jnp, lax
        self.n = n = int(n_image)
        self.wins = windows(n, method, ministack, lag)
        self.ps_shp = int(ps_shp)
        self.hr, self.hc = (azimuth_window - 1) // 2, (range_window - 1) // 2
        dy, dx = np.meshgrid(np.arange(-self.hr, self.hr + 1), np.arange(-self.hc, self.hc + 1), indexing="ij")
        self.dy, self.dx = dy.ravel().astype(np.int32), dx.ravel().astype(np.int32)
        self.nk = self.dy.size
        self.center = self.nk // 2
        self._ks = jax.jit(self._ks_impl)
        self._ranks = jax.jit(self._ranks_impl)
        self._fill = jax.jit(self._fill_impl)
        self._link = jax.jit(self._link_impl)

    # --- SHP test
    def _ranks_impl(self, s):
        jnp, jax = self.jnp, self.jax
        return jax.vmap(lambda a: jnp.searchsorted(a, a, side="right"))(s).astype(jnp.int32)

    def _ks_impl(self, s, r, u, v):
        """Integer KS gaps max|#A<=x - #B<=x| over x in A u B for the pairs (u, v)."""
        jnp, jax = self.jnp, self.jax
        A, B, rA, rB = s[u], s[v], r[u], r[v]
        cB = jax.vmap(lambda b, a: jnp.searchsorted(b, a, side="right"))(B, A).astype(jnp.int32)
        cA = jax.vmap(lambda a, b: jnp.searchsorted(a, b, side="right"))(A, B).astype(jnp.int32)
        return jnp.maximum(jnp.max(jnp.abs(rA - cB), -1), jnp.max(jnp.abs(cA - rB), -1))

    def _fill_impl(self, sim):
        """8-connected component of `sim` (P, h, w) holding the centre."""
        jnp, lax = self.jnp, self.lax
        P, h, w = sim.shape
        seed = jnp.zeros_like(sim).at[:, h // 2, w // 2].set(True) & sim

        def grow(x):
            p = jnp.pad(x, ((0, 0), (1, 1), (1, 1)))
            g = jnp.zeros_like(x)
            for i in range(3):
                for j in range(3):
                    g = g | p[:, i:i + h, j:j + w]
            return g & sim

        def cond(c):
            return c[1]

        def body(c):
            x, _ = c
            y = grow(x)
            return y, jnp.any(y != x)

        out, _ = lax.while_loop(cond, body, (seed, jnp.array(True)))
        return out

    # --- linking
    @staticmethod
    def _angle_r(jnp, z):
        a = jnp.angle(z)
        return jnp.where(jnp.isnan(a), 0.0, a)

    def _gam(self, phi, vec):
        """gam_pta_c: 2/(n^2-n) * sum_{i<k} cos(phi_ik - (a_i - a_k)), phi = angle_r(C)."""
        jnp = self.jnp
        n = phi.shape[-1]
        a = self._angle_r(jnp, vec)
        t = jnp.cos(phi - (a[..., :, None] - a[..., None, :]))
        t = jnp.where(jnp.triu(jnp.ones((n, n), bool), 1), t, 0.0)
        return jnp.sum(t, axis=(-2, -1)) * 2 / (n * n - n)

    def _regularize(self, A):
        """regularize_matrix_cy: (ok, matrix) after up to 100 Cholesky tries."""
        jnp, lax = self.jnp, self.lax
        s = A.shape[-1]
        eye = jnp.eye(s, dtype=A.dtype)

        def chol_ok(M):
            L = jnp.linalg.cholesky(M)
            return jnp.all(jnp.isfinite(L), axis=(-2, -1))

        ok0 = chol_ok(A)
        en0 = jnp.full(A.shape[:-2], 1e-6, A.dtype)

        def cond(c):
            _, _, ok, t = c
            return (t < 99) & ~jnp.all(ok)

        def body(c):
            M, en, ok, t = c
            M = jnp.where(ok[..., None, None], M, M + en[..., None, None] * eye)
            en = jnp.where(ok, en, en * 2)
            return M, en, ok | chol_ok(M), t + 1

        M, _, ok, _ = lax.while_loop(cond, body, (A, en0, ok0, jnp.array(0)))
        return ok, M

    def _link_window(self, C):
        """phase_linking_process_cy for EMI on a batch of windows C (..., s, s)."""
        jnp, jax = self.jnp, self.jax
        A = jnp.abs(C)
        ok, R = self._regularize(A)
        L = jnp.linalg.cholesky(jnp.where(ok[..., None, None], R, jnp.eye(R.shape[-1], dtype=R.dtype)))
        eye = jnp.broadcast_to(jnp.eye(R.shape[-1], dtype=R.dtype), R.shape)
        inv = jax.scipy.linalg.cho_solve((L, True), eye)
        M = inv.astype(C.dtype) * C
        _, v_emi = jnp.linalg.eigh(M)
        _, v_evd = jnp.linalg.eigh(C)
        v = jnp.where(ok[..., None], v_emi[..., :, 0], v_evd[..., :, -1])
        x0 = jnp.exp(1j * self._angle_r(jnp, v[..., 0])).astype(C.dtype)
        res = v * jnp.conj(x0)[..., None]
        q = self._gam(self._angle_r(jnp, C), res)
        return res, q

    def _link_impl(self, Z, Am, pr, pc, shp, nshp):
        """Batch of pixels: Z (rows, cols, n) complex64 of the patch region and
        Am its cabsf amplitudes; pr, pc (b,) pixel position in Z; shp (b, nk)
        bool; nshp (b,)."""
        jnp = self.jnp
        n = self.n
        rows = jnp.clip(pr[:, None] + self.dy[None, :], 0, Z.shape[0] - 1)
        cols = jnp.clip(pc[:, None] + self.dx[None, :], 0, Z.shape[1] - 1)
        X = Z[rows, cols]                                   # (b, nk, n)
        AX = Am[rows, cols]
        m = shp.astype(jnp.float32)
        Xm = X * m[..., None]
        ns = nshp.astype(jnp.float32)
        cov = jnp.einsum("bkn,bkm->bnm", Xm, jnp.conj(Xm)) / ns[:, None, None]
        d = jnp.sqrt(jnp.abs(jnp.diagonal(cov, axis1=1, axis2=2)))
        corr = jnp.where(cov == 0, 0, cov / (d[:, :, None] * d[:, None, :]).astype(cov.dtype))
        phi = self._angle_r(jnp, corr)
        amp = jnp.sum(AX * m[..., None], axis=1) / ns[:, None]

        # sequential EMI: windows grouped by size so each group is one batched solve
        by_size = {}
        for k, (a, f, l) in enumerate(self.wins):
            by_size.setdefault(l - a, []).append(k)
        res = [None] * len(self.wins)
        q = [None] * len(self.wins)
        for s, ks in by_size.items():
            C = jnp.stack([corr[:, self.wins[k][0]:self.wins[k][2], self.wins[k][0]:self.wins[k][2]] for k in ks], 1)
            r, qq = self._link_window(C)
            for j, k in enumerate(ks):
                res[k], q[k] = r[:, j], qq[:, j]
        vr = jnp.zeros((X.shape[0], n), corr.dtype)
        for k, (a, f, l) in enumerate(self.wins):
            shift = jnp.zeros(X.shape[0], jnp.float32)
            if k > 0 and f > a:
                acc = jnp.sum(jnp.exp(1j * (jnp.angle(vr[:, a:f]) - jnp.angle(res[k][:, :f - a]))), axis=1)
                shift = jnp.angle(acc)
            vr = vr.at[:, f:l].set(res[k][:, f - a:l - a] * jnp.exp(1j * shift)[:, None].astype(corr.dtype))
        q_avg = sum(q) / len(self.wins)
        q_full = self._gam(phi, vr)

        # PS candidates: eigenpairs of the rank <= ps_shp matrix from its Gram matrix
        kps = self.ps_shp
        order = jnp.argsort(~shp, axis=1, stable=True)[:, :kps]           # SHP columns first, window order
        Xs = jnp.take_along_axis(Xm, order[..., None], axis=1)           # (b, kps, n)
        dsafe = jnp.where(d > 0, d, 1.0)
        Y = jnp.where((d > 0)[:, None, :], Xs / dsafe[:, None, :], 0) / jnp.sqrt(ns)[:, None, None]
        G = jnp.einsum("bkn,bjn->bkj", jnp.conj(Y), Y)                  # (b, kps, kps) = Y^H Y with Y (n x kps)
        lam, W = jnp.linalg.eigh(G)
        lam1, lam2 = lam[:, -1], lam[:, -2] if kps > 1 else jnp.zeros_like(lam[:, -1])
        u = jnp.einsum("bkn,bk->bn", Y, W[:, :, -1]) / jnp.sqrt(jnp.where(lam1 > 0, lam1, 1.0))[:, None]
        u = jnp.where((lam1 > 0)[:, None], u, 0)
        x0 = jnp.exp(1j * self._angle_r(jnp, u[:, 0])).astype(u.dtype)
        vps = u * jnp.conj(x0)[:, None]
        s = jnp.sqrt(jnp.sum(jnp.abs(lam) ** 2, axis=1))
        top = lam1 * (100 / s)
        z = X[:, self.center, :]                                         # the pixel itself
        az = AX[:, self.center, :]
        ampd = jnp.std(az, axis=1) / jnp.mean(az, axis=1)
        ampd = jnp.where(ampd > 1, 1.0, ampd)
        is_ps = (top > 95) & (ampd < 0.42)
        g_ps = self._gam(phi, vps)
        tq_ps = jnp.where(is_ps, 1.0, jnp.where(g_ps == 1, 0.95, g_ps))
        own = z * jnp.conj(z[:, :1])
        v_ps = jnp.where(is_ps[:, None], own, vps)

        cand = nshp <= kps
        vec = jnp.where(cand[:, None], v_ps, vr)
        amp_out = jnp.where(cand[:, None], az, amp)
        tc0 = jnp.where(cand, tq_ps, q_avg)
        tc1 = jnp.where(cand, tq_ps, q_full)
        out = amp_out.astype(jnp.complex64) * jnp.exp(1j * jnp.angle(vec)).astype(jnp.complex64)
        out = out.at[:, 0].set(amp_out[:, 0].astype(jnp.complex64))
        tc0, tc1 = jnp.maximum(tc0, 0), jnp.maximum(tc1, 0)
        ps_prod = jnp.where(cand[:, None], jnp.stack([ampd, lam1, lam2, top], 1), 0.0)
        mask_ps = cand & is_ps
        return out, tc0, tc1, ps_prod, mask_ps


# ------------------------------------------------------------ one patch

def _region(box, length, width, hr, hc):
    c0, r0, c1, r1 = box[:4]
    return max(0, r0 - hr), min(length, r1 + hr), max(0, c0 - hc), min(width, c1 + hc)


def read_patch(slc_ds, mask_ds, box, length, width, hr, hc):
    """The patch region (rows, cols, n) complex64, its sorted amplitudes, the
    box mask (1 = link)."""
    er0, er1, ec0, ec1 = _region(box, length, width, hr, hc)
    z = np.asarray(slc_ds[:, er0:er1, ec0:ec1], dtype=np.complex64)
    amp = cabsf(z)
    a = np.sort(amp.reshape(z.shape[0], -1).T, axis=1)      # (pixels, n)
    c0, r0, c1, r1 = box[:4]
    if mask_ds is None:
        mask = np.ones((r1 - r0, c1 - c0), bool)
    else:
        mask = np.asarray(mask_ds[r0:r1, c0:c1]) != 0
    return {"z": np.ascontiguousarray(np.moveaxis(z, 0, -1)), "amp": np.ascontiguousarray(np.moveaxis(amp, 0, -1)),
            "amp_sorted": np.ascontiguousarray(a), "mask": mask, "region": (er0, er1, ec0, ec1)}


def cabsf(z: np.ndarray) -> np.ndarray:
    """|z| as C's cabsf (glibc hypotf: the double-precision hypot rounded to
    float), which MiaplPy uses. numpy's complex64 abs differs by 1 ulp on about
    a third of values, enough to move a KS gap that sits on the threshold."""
    return np.hypot(z.real.astype(np.float64), z.imag.astype(np.float64)).astype(np.float32)


def _pad_to(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-pad the leading axes of `a` to `shape`, so every patch reuses the
    same compiled kernels (padding is never indexed as a neighbour)."""
    if a.shape[:len(shape)] == tuple(shape):
        return a
    out = np.zeros(tuple(shape) + a.shape[len(shape):], a.dtype)
    out[tuple(slice(0, k) for k in a.shape[:len(shape)])] = a
    return out


def link_patch(K: Kernels, data: dict, box, threshold_count: int, pixel_batch: int, pair_batch: int,
               region_shape: tuple | None = None) -> dict:
    jnp = K.jnp
    c0, r0, c1, r1 = box[:4]
    er0, er1, ec0, ec1 = data["region"]
    H, W = er1 - er0, ec1 - ec0
    HP, WP = region_shape or (H, W)
    n = K.n
    s = jnp.asarray(_pad_to(data["amp_sorted"], (HP * WP,)))
    r = K._ranks(s)

    # KS gaps on the half-window of offsets; the other half by symmetry
    gap = np.full((K.nk, H, W), np.iinfo(np.int32).max, np.int32)
    gap[K.center] = 0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    us, vs, slots = [], [], []
    for o in range(K.nk):
        dy, dx = int(K.dy[o]), int(K.dx[o])
        if not (dy > 0 or (dy == 0 and dx > 0)):
            continue
        ok = (yy + dy < H) & (xx + dx >= 0) & (xx + dx < W)
        u = (yy[ok] * W + xx[ok]).astype(np.int32)
        us.append(u)
        vs.append(u + dy * W + dx)
        slots.append((o, ok))
    u_all, v_all = np.concatenate(us), np.concatenate(vs)
    m = u_all.size
    pad = (-m) % pair_batch
    u_p = np.concatenate([u_all, np.zeros(pad, np.int32)])
    v_p = np.concatenate([v_all, np.zeros(pad, np.int32)])
    d_all = np.empty(u_p.size, np.int32)
    for i in range(0, u_p.size, pair_batch):
        d_all[i:i + pair_batch] = np.asarray(K._ks(s, r, jnp.asarray(u_p[i:i + pair_batch]),
                                                 jnp.asarray(v_p[i:i + pair_batch])))
    d_all = d_all[:m]
    pos = 0
    opp = {(int(K.dy[o]), int(K.dx[o])): o for o in range(K.nk)}
    for (o, ok), u in zip(slots, us):
        d = d_all[pos:pos + u.size]
        pos += u.size
        dy, dx = int(K.dy[o]), int(K.dx[o])
        gap[o][ok] = d                                         # pair seen from u
        o2 = opp[(-dy, -dx)]
        g2 = gap[o2]
        g2[(yy[ok] + dy), (xx[ok] + dx)] = d                   # same pair seen from u + o
    br0, bc0 = r0 - er0, c0 - ec0
    bh, bw = r1 - r0, c1 - c0
    sim = gap[:, br0:br0 + bh, bc0:bc0 + bw] <= threshold_count          # (nk, bh, bw)
    sim = np.moveaxis(sim, 0, -1).reshape(bh * bw, 2 * K.hr + 1, 2 * K.hc + 1)

    shp = np.empty((bh * bw, K.nk), bool)
    for i in range(0, sim.shape[0], 8192):
        chunk = sim[i:i + 8192]
        shp[i:i + 8192] = np.asarray(K._fill(jnp.asarray(_pad_to(chunk, (8192,)))))[:chunk.shape[0]].reshape(-1, K.nk)
    nshp = shp.sum(1).astype(np.int32)

    mask = data["mask"].ravel()
    py, px = np.divmod(np.arange(bh * bw), bw)
    pr, pc = (py + br0).astype(np.int32), (px + bc0).astype(np.int32)
    todo = np.flatnonzero(mask)
    out = np.empty((bh * bw, n), np.complex64)
    tc = np.zeros((2, bh * bw), np.float32)
    psp = np.zeros((4, bh * bw), np.float32)
    mps = np.zeros(bh * bw, bool)
    Z, Am = jnp.asarray(_pad_to(data["z"], (HP, WP))), jnp.asarray(_pad_to(data["amp"], (HP, WP)))
    for i in range(0, todo.size, pixel_batch):
        sel = todo[i:i + pixel_batch]
        k = sel.size
        idx = np.concatenate([sel, np.full(pixel_batch - k, sel[0])]) if k < pixel_batch else sel
        o, t0, t1, pp, mp = K._link(Z, Am, jnp.asarray(pr[idx]), jnp.asarray(pc[idx]), jnp.asarray(shp[idx]),
                                    jnp.asarray(nshp[idx]))
        out[sel] = np.asarray(o)[:k]
        tc[0, sel], tc[1, sel] = np.asarray(t0)[:k], np.asarray(t1)[:k]
        psp[:, sel] = np.asarray(pp)[:k].T
        mps[sel] = np.asarray(mp)[:k]
    # masked pixels: MiaplPy's else-branch
    rest = np.flatnonzero(~mask)
    if rest.size:
        zp = data["z"][pr[rest], pc[rest]]
        out[rest] = zp * np.conj(zp[:, :1])
        tc[:, rest] = 0.1
        nshp[rest] = 1
    return {"phase_ref": np.ascontiguousarray(np.moveaxis(out.reshape(bh, bw, n), -1, 0)),
            "shp": nshp.reshape(bh, bw).astype(np.int32),
            "tempCoh": tc.reshape(2, bh, bw),
            "mask_ps": mps.reshape(bh, bw),
            "ps_products": psp.reshape(4, bh, bw)}


def write_patch(out_dir: Path, index: int, res: dict, meta: dict) -> Path:
    """The patch files, each written to a temporary name and renamed; flag.npy
    last, so a patch is either complete or not flagged."""
    d = Path(out_dir) / "PATCHES" / f"PATCH_{index:04d}"
    d.mkdir(parents=True, exist_ok=True)
    for name in PATCH_FILES:
        tmp = d / f".{name}.tmp.npy"
        np.save(tmp, res[name])
        os.replace(tmp, d / f"{name}.npy")
    (d / "kernel.json").write_text(json.dumps(meta, indent=1))
    tmp = d / ".flag.tmp.npy"
    np.save(tmp, [1])
    os.replace(tmp, d / "flag.npy")
    return d


# ------------------------------------------------------------ self-check against MiaplPy on the CPU
# Run before a project links on the GPU (miaplpy_backend.verify_gpu_kernel) and
# by tests/test_pl_gpu.py: one synthetic patch linked by MiaplPy's own
# process_patch_c (the installed build, in its env) and by this kernel, compared
# with the limits the speed plan's Gate 2(a) set before any result. On real
# study_areav2 patches (11 Nov 2026): SHP counts identical on every pixel,
# phase p99 <= 0.0014 rad against the record.
SELFCHECK_SHAPE = (60, 48, 48)               # dates, rows, cols: 6 mini-stacks of 10
SELFCHECK_LIMITS = {"shp_identical_min": 0.995, "phase_median_max": 0.01, "phase_p99_max": 0.05,
                    "ps_mask_mismatch_max_frac": 0.005}

CPU_REFERENCE = r'''
import sys, json, numpy as np, h5py
from miaplpy.lib import utils as iut
sys.path.insert(0, sys.argv[3])
import pl_gpu
stack, out = sys.argv[1], sys.argv[2]
with h5py.File(stack, "r") as h:
    n, L, W = h["slc"].shape
class Obj:
    def read(self, datasetName=None, box=None, print_msg=False):
        x0, y0, x1, y1 = box
        with h5py.File(stack, "r") as h:
            return np.ascontiguousarray(h["slc"][:, y0:y1, x0:x1])
rows = np.arange(-7, 8, dtype=np.int32); ref = np.array([7], dtype=np.int32)
iut.process_patch_c(np.array([0, 0, W, L, 0], dtype=np.int32), 15, 15, W, L, n, Obj(), pl_gpu.ks_threshold(n),
                    rows, rows, ref, ref, b"sequential_EMI", n // 10, 10, 10, b"ks", out.encode(), 3, b"/nonexistent")
import miaplpy
print(json.dumps({"miaplpy": getattr(miaplpy, "__version__", None)}))
'''


def synthetic_stack(path: Path, seed: int = 7, shape: tuple = SELFCHECK_SHAPE) -> Path:
    """A small stack with what the SHP test and the PS test must tell apart:
    two distributed-scatterer fields of different brightness and coherence,
    a decorrelating strip, bright stable points, and a no-data corner."""
    import h5py
    n, L, W = shape
    rng = np.random.default_rng(seed)

    def cn(*s):
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)

    defo = np.cumsum(rng.normal(0, 0.15, n))                    # a slow per-date phase history
    common = cn(n, 1, 1) * np.exp(1j * defo)[:, None, None]
    z = np.empty((n, L, W), np.complex64)
    for (r0, r1, c0, c1, scale, coh) in [(0, L, 0, W // 2, 1.0, 0.8), (0, L, W // 2, W, 2.5, 0.5),
                                         (L // 2 - 4, L // 2 + 4, 0, W, 1.5, 0.1)]:
        part = np.sqrt(coh) * common + np.sqrt(1 - coh) * cn(n, r1 - r0, c1 - c0)
        z[:, r0:r1, c0:c1] = (scale * part).astype(np.complex64)
    for r, c in rng.integers(4, min(L, W) - 4, size=(12, 2)):
        z[:, r, c] = (40 * (1 + 0.05 * rng.standard_normal(n)) * np.exp(1j * (defo + rng.normal(0, 0.1, n)))).astype(np.complex64)
    for r, c in rng.integers(4, min(L, W) - 4, size=(12, 2)):
        # lone bright noisy pixels: PS candidates (few SHPs) that fail the dispersion test
        z[:, r, c] = (12 * cn(n)).astype(np.complex64)
    z[:, :5, :7] = 0                                             # no data, as at a burst edge
    with h5py.File(path, "w") as h:
        h["slc"] = z
        h["date"] = np.array([f"2020{(i // 28) + 1:02d}{(i % 28) + 1:02d}".encode() for i in range(n)])
    return Path(path)


def compare_patch_dirs(a: Path, b: Path) -> dict:
    """One patch folder against another: SHP counts, PS mask, phase, coherence."""
    o = {k: np.load(Path(a) / f"{k}.npy") for k in PATCH_FILES}
    w = {k: np.load(Path(b) / f"{k}.npy") for k in PATCH_FILES}
    dph = np.abs(np.angle(o["phase_ref"] * np.conj(w["phase_ref"])))[1:]
    dtc = np.abs(o["tempCoh"] - w["tempCoh"])
    return {"pixels": int(o["shp"].size), "shp_identical": float((o["shp"] == w["shp"]).mean()),
            "ps_mask_mismatch_frac": float((o["mask_ps"] != w["mask_ps"]).mean()),
            "phase_median_rad": float(np.median(dph)), "phase_p99_rad": float(np.percentile(dph, 99)),
            "phase_max_rad": float(dph.max()), "tempcoh_p99": float(np.percentile(dtc, 99)),
            "nan_mismatch": int((np.isnan(o["phase_ref"]) != np.isnan(w["phase_ref"])).sum())}


def selfcheck_verdict(m: dict) -> tuple[bool, str]:
    lim = SELFCHECK_LIMITS
    bad = []
    if m["shp_identical"] < lim["shp_identical_min"]:
        bad.append(f"SHP counts identical on {m['shp_identical']:.1%} (< {lim['shp_identical_min']:.1%})")
    if m["ps_mask_mismatch_frac"] > lim["ps_mask_mismatch_max_frac"]:
        bad.append(f"PS mask differs on {m['ps_mask_mismatch_frac']:.1%}")
    if m["phase_median_rad"] > lim["phase_median_max"]:
        bad.append(f"phase median {m['phase_median_rad']:.4f} rad")
    if m["phase_p99_rad"] > lim["phase_p99_max"]:
        bad.append(f"phase p99 {m['phase_p99_rad']:.4f} rad")
    if m["nan_mismatch"]:
        bad.append(f"{m['nan_mismatch']} NaN mismatches")
    return (not bad), "; ".join(bad) or "within the Gate 2(a) limits"


# ------------------------------------------------------------ driver

def is_out_of_memory(exc: BaseException) -> bool:
    """XLA's allocation failure (RESOURCE_EXHAUSTED), however JAX wraps it."""
    text = f"{type(exc).__name__}: {exc}"
    return "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()


def default_batches(n_image: int, budget_gb: float) -> tuple[int, int]:
    """Pixels per linking batch (~6 N x N complex64 arrays each) and KS pairs
    per batch, inside budget_gb of device memory."""
    per_px = 6 * n_image * n_image * 8 + 2 * 225 * n_image * 8
    px = int(max(16, min(1024, budget_gb * 1e9 // per_px)))
    px = 1 << (px.bit_length() - 1)
    pairs = int(max(1024, min(65536, budget_gb * 1e9 // (8 * n_image * 4))))
    pairs = 1 << (pairs.bit_length() - 1)
    return px, pairs


def run(args) -> dict:
    import h5py
    method = args.method
    with h5py.File(args.slc_stack, "r") as f:
        n, length, width = f["slc"].shape
    if n < 10 and method.startswith("sequential"):
        method = method.split("sequential_")[1]                   # MiaplPy does the same below 10 images
    if method not in SUPPORTED_METHODS:
        raise SystemExit(f"pl_gpu: method {method!r} is not implemented on the GPU (only {SUPPORTED_METHODS})")
    thr = ks_threshold(n)
    kmax = ks_max_count(n, thr)
    boxes = patch_boxes(length, width, args.patch_size, args.range_window, args.azimuth_window)
    if args.boxes:
        boxes = [tuple(int(v) for v in b.split(",")) for b in args.boxes]
    elif args.patches:
        want = {int(v) for v in args.patches.split(",")}
        boxes = [b for b in boxes if b[4] in want]
    out_dir = Path(args.out_dir)
    todo = [b for b in boxes if not (out_dir / "PATCHES" / f"PATCH_{b[4]:04d}" / "flag.npy").exists()]
    if args.reverse:
        todo = todo[::-1]
    px, pairs = default_batches(n, args.memory_gb)
    px = args.pixel_batch or px
    pairs = args.pair_batch or pairs
    K = Kernels(n, method, args.ministack, args.lag, args.ps_shp, args.range_window, args.azimuth_window)
    regions = [_region(b, length, width, K.hr, K.hc) for b in todo] or [(0, 1, 0, 1)]
    region_shape = (max(r[1] - r[0] for r in regions), max(r[3] - r[2] for r in regions))
    meta = {"kernel": kernel_label(), "method": method, "n_image": int(n), "ks_threshold": thr, "ks_max_gap": kmax,
            "ministack": args.ministack, "lag": args.lag, "ps_shp": args.ps_shp,
            "range_window": args.range_window, "azimuth_window": args.azimuth_window, "device": probe().get("device")}
    print(f"pl_gpu: {len(todo)} of {len(boxes)} patches to link, {n} images, KS threshold {thr:.3f} "
          f"(gap <= {kmax}), batches {px} px / {pairs} pairs", flush=True)

    mask_f = h5py.File(args.mask, "r") if args.mask and os.path.exists(args.mask) else None
    mask_ds = None
    if mask_f is not None:
        mask_ds = mask_f["mask"] if "mask" in mask_f else mask_f[next(k for k in mask_f if mask_f[k].ndim == 2)]
    f = h5py.File(args.slc_stack, "r")
    ds = f["slc"]
    q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def reader():
        try:
            for b in todo:
                if stop.is_set():
                    break
                t = time.time()
                d = read_patch(ds, mask_ds, b, length, width, K.hr, K.hc)
                d["read_s"] = time.time() - t
                q.put((b, d))
        except BaseException as exc:                                # noqa: BLE001
            q.put((None, exc))
            return
        q.put((None, None))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    done, t_all = 0, time.time()
    try:
        while True:
            b, d = q.get()
            if b is None:
                if isinstance(d, BaseException):
                    raise d
                break
            t = time.time()
            while True:
                try:
                    res = link_patch(K, d, b, kmax, px, pairs, region_shape)
                    break
                except Exception as exc:                            # noqa: BLE001
                    if not is_out_of_memory(exc) or (px <= 16 and pairs <= 1024):
                        raise
                    # the desktop took device memory back: same patch, half the batch
                    px, pairs = max(16, px // 2), max(1024, pairs // 2)
                    meta["batches_halved"] = meta.get("batches_halved", 0) + 1
                    print(f"pl_gpu: out of device memory on PATCH_{b[4]:04d}; retrying with {px} px / {pairs} "
                          "pairs per batch", flush=True)
            secs = time.time() - t
            write_patch(out_dir, b[4], res, {**meta, "box": list(b[:4]), "seconds": round(secs, 1),
                                              "read_seconds": round(d["read_s"], 1)})
            done += 1
            print(f"    Phase inversion of PATCH_{b[4]:04d} is Completed on GPU in {secs:.1f} s "
                  f"(read {d['read_s']:.1f} s) [{done}/{len(todo)}]", flush=True)
            if args.max_patches and done >= args.max_patches:
                break
    finally:
        stop.set()
        f.close()
        if mask_f is not None:
            mask_f.close()
    return {"patches": done, "seconds": round(time.time() - t_all, 1), **meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pl_gpu.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    r = sub.add_parser("run")
    r.add_argument("--slc-stack", required=True)
    r.add_argument("--out-dir", required=True, help="MiaplPy's <work>/inverted")
    r.add_argument("--mask", default="")
    r.add_argument("--range-window", type=int, default=15)
    r.add_argument("--azimuth-window", type=int, default=15)
    r.add_argument("--patch-size", type=int, default=200)
    r.add_argument("--method", default="sequential_EMI")
    r.add_argument("--ministack", type=int, default=10)
    r.add_argument("--lag", type=int, default=3)
    r.add_argument("--ps-shp", type=int, default=10)
    r.add_argument("--memory-gb", type=float, default=4.0)
    r.add_argument("--pixel-batch", type=int, default=0)
    r.add_argument("--pair-batch", type=int, default=0)
    r.add_argument("--patches", default="", help="comma-separated patch indices (default: all unfinished)")
    r.add_argument("--boxes", nargs="*", default=[], help="c0,r0,c1,r1,index boxes instead of the patch grid (tests)")
    r.add_argument("--reverse", action="store_true", help="last patch first")
    r.add_argument("--max-patches", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        print(json.dumps(probe()))
        return 0
    print(json.dumps(run(a)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
