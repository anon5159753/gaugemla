import hashlib
import warnings

import numpy as np

import dt_decoupled
import gam_core as gc
import kvlab

ARM = "mla_ours_fisher_sigma"
SCHEMA = "mla/v1"
VERSION = "mla_ours/1 (2026-09-29)"
ROTARY_MODE = "shared"
BIAS_SET = ["dkv"]
MLA_SCALE = 0.125

def mla_scale_of(w, d_h):
    return float(d_h ** -0.5)

KV_EXCHANGE_RATE = 1.0

PHASE_PIVOT_REL = 1e-6
DEGENERATE_REL = 1e-6
TSIGT_TOL = 1e-6
PSI_CONSISTENCY_TOL = 1e-12

def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]

def varpi_of(GK_l, r):
    GK_l = np.asarray(GK_l, dtype=np.float64)
    P = r // 2
    ii = np.arange(P)
    vp = 0.5 * (GK_l[:, ii, ii] + GK_l[:, ii + P, ii + P])
    if not np.isfinite(vp).all():
        raise ValueError("non-finite rotary Fisher diagonal")
    if (vp <= 0).any():
        bad = np.argwhere(vp <= 0)
        raise ValueError(f"varpi <= 0 at (head, plane) {bad[:8].tolist()} (min {vp.min():.3e}): "
                         f"GK is not a Fisher on those planes. Never floored (build plan §4a.6).")
    return vp

def phase_pin(Vh):
    Vh = np.array(Vh, dtype=np.complex128, copy=True)
    a = np.abs(Vh)
    for j in range(Vh.shape[0]):
        amax = a[j].max()
        if not amax > 0:
            raise ValueError(f"row {j} is identically zero -- not a singular vector")
        piv = int(np.flatnonzero(a[j] >= (1.0 - PHASE_PIVOT_REL) * amax)[0])
        z = Vh[j, piv]
        Vh[j] *= np.conj(z) / abs(z)
    return Vh

def plane_spectra(Kt, GK_l, S_h, r, phase_fix=True):
    Kt = np.asarray(Kt, dtype=np.float64)
    H = Kt.shape[0]
    if r <= 0 or r % 2:
        raise ValueError(f"rotary width r = {r} must be positive and even")
    P = r // 2
    vp = varpi_of(GK_l, r)
    if vp.shape != (H, P):
        raise ValueError(f"varpi is {vp.shape}, expected ({H}, {P}): GK_l is not this layer's")
    Ks = np.stack([gc.to_complex(Kt[i], r) for i in range(H)]) @ S_h
    Vh, mu = [], []
    for p in range(P):
        _, sp, vh = np.linalg.svd(np.sqrt(vp[:, p])[:, None] * Ks[:, p, :], full_matrices=False)
        Vh.append(phase_pin(vh) if phase_fix else vh)
        mu.append(sp ** 2)
    return Vh, mu, vp

def direction_order(mu):
    n = [len(np.asarray(m)) for m in mu]
    flat = np.concatenate([np.asarray(m, dtype=np.float64) for m in mu])
    if not np.isfinite(flat).all() or (flat < 0).any():
        raise ValueError("mu must be finite and non-negative (squared singular values)")
    plane = np.concatenate([np.full(k, p, dtype=np.int64) for p, k in enumerate(n)])
    idx = np.concatenate([np.arange(k, dtype=np.int64) for k in n])
    order = np.argsort(-flat, kind="stable")
    return np.stack([plane[order], idx[order]], axis=1)

def allocate(mu, n_dir):
    order = direction_order(mu)
    n_dir = int(n_dir)
    if not 0 <= n_dir <= len(order):
        raise ValueError(f"asked for {n_dir} complex rotary directions; this layer has {len(order)} "
                         f"(= sum over planes of min(H, dm+1))")
    P = len(mu)
    kept = order[:n_dir]
    m = np.bincount(kept[:, 0], minlength=P).astype(np.int64)
    for p in range(P):
        got = np.sort(kept[kept[:, 0] == p, 1])
        if not np.array_equal(got, np.arange(m[p])):
            raise AssertionError(f"plane {p}: kept directions {got.tolist()} are not a prefix -- "
                                 f"mu is not descending within the plane")
    mu_in = float(mu[kept[-1, 0]][kept[-1, 1]]) if n_dir else float("inf")
    mu_out = float(mu[order[n_dir, 0]][order[n_dir, 1]]) if n_dir < len(order) else 0.0
    return m, {"boundary_mu_in": mu_in, "boundary_mu_out": mu_out,
               "boundary_gap": mu_in - mu_out if n_dir else float("nan")}

def rotary_maps(Qt, Kt, Vh, m, S_ih, Sigma, r):
    Qt = np.asarray(Qt, dtype=np.float64)
    Kt = np.asarray(Kt, dtype=np.float64)
    H, _, n = Qt.shape
    P = r // 2
    m = np.asarray(m, dtype=np.int64)
    if m.shape != (P,) or (m < 0).any():
        raise ValueError(f"m must be ({P},) non-negative, got {m}")
    nh = int(m.sum())
    w_rot = 2 * nh
    if w_rot == 0:
        raise ValueError("w = 0 is never built (D4): MLA's rotary cache must be non-empty")
    Sig = (np.asarray(Sigma, dtype=np.float64) + np.asarray(Sigma, dtype=np.float64).T) / 2.0
    Kc = np.stack([gc.to_complex(Kt[i], r) for i in range(H)])
    W_KR = np.zeros((w_rot, n))
    W_QR = np.zeros((H * w_rot, n))
    Psi = np.zeros((H, r, w_rot))
    half = np.repeat(np.arange(P, dtype=np.int64), m)
    slot_plane = np.concatenate([half, half])
    tsigt = np.zeros(P)
    off = 0
    for p in range(P):
        mp = int(m[p])
        if mp == 0:
            continue
        if mp > Vh[p].shape[0]:
            raise ValueError(f"plane {p}: m_p = {mp} > {Vh[p].shape[0]} available directions")
        T = Vh[p][:mp] @ S_ih
        TS = T @ Sig
        Gp = TS @ T.conj().T
        tsigt[p] = float(np.abs(Gp - np.eye(mp)).max())
        if tsigt[p] > TSIGT_TOL:
            raise ValueError(
                f"plane {p}: ||T Sigma T^H - I||_max = {tsigt[p]:.3e} > {TSIGT_TOL:g}. A kept "
                f"direction is not in range(Sigma) -- its singular value is zero (or floored), so "
                f"it carries no key energy and psi is ill-posed. Too many directions asked for.")
        Bp = (Kc[:, p, :] @ Sig) @ T.conj().T
        psi = np.linalg.solve(Gp.T, Bp.T).T
        s = off + np.arange(mp)
        W_KR[s] = T.real
        W_KR[nh + s] = T.imag
        u, v = psi.real, psi.imag
        Psi[:, p, s] = u
        Psi[:, p + P, s] = v
        Psi[:, p, nh + s] = -v
        Psi[:, p + P, nh + s] = u
        for i in range(H):
            qa, qb = Qt[i][p], Qt[i][p + P]
            W_QR[i * w_rot + s] = u[i][:, None] * qa[None, :] + v[i][:, None] * qb[None, :]
            W_QR[i * w_rot + nh + s] = -v[i][:, None] * qa[None, :] + u[i][:, None] * qb[None, :]
        off += mp
    worst = 0.0
    for i in range(H):
        ref = Psi[i].T @ Qt[i][:r]
        blk = W_QR[i * w_rot:(i + 1) * w_rot]
        worst = max(worst, float(np.abs(blk - ref).max() / max(np.abs(blk).max(), 1e-300)))
    if worst > PSI_CONSISTENCY_TOL:
        raise AssertionError(f"W_QR_i != Psi[i]^T Qt_i[:r]: rel {worst:.3e}")
    return W_KR, W_QR, Psi, slot_plane, {"tsigt_err": tsigt.tolist(), "psi_consistency": worst}

def content_basis(Qt, Kt, Vt, Wo, GK_l, GV_l, S_h, S_ih, d_c, r):
    H = Kt.shape[0]
    A_free, A_V, _ = dt_decoupled._metrics(Qt, Wo, H, r, None, None, GK_l, GV_l)
    Z_c = dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H, r, w=KV_EXCHANGE_RATE)
    _, s, Vt_c = np.linalg.svd(Z_c @ S_h, full_matrices=False)
    if d_c > Vt_c.shape[0]:
        raise ValueError(f"d_c = {d_c} exceeds the content stack's {Vt_c.shape[0]} directions")
    Wbar = Vt_c[:d_c].copy()
    C = np.linalg.qr((Wbar @ S_ih).T)[0].T
    Mread = S_h @ Wbar.T @ (Wbar @ S_ih @ C.T)
    return C, Mread, s ** 2

def build_layer(Qt, Kt, Vt, Wo, GK_l, GV_l, Sigma_l, *, w_rot, d_c, r, phase_fix=True):
    Qt = np.asarray(Qt, dtype=np.float64)
    Kt = np.asarray(Kt, dtype=np.float64)
    Vt = np.asarray(Vt, dtype=np.float64)
    H, d_k, n = Kt.shape
    d_v = Vt.shape[1]
    if Qt.shape != (H, d_k, n) or Vt.shape[0] != H or Vt.shape[2] != n:
        raise ValueError(f"inconsistent maps: Qt {Qt.shape}, Kt {Kt.shape}, Vt {Vt.shape}")
    if w_rot % 2 or w_rot <= 0:
        raise ValueError(f"w = {w_rot} must be even and positive (D4 forbids w = 0)")
    if not 0 < d_c <= n:
        raise ValueError(f"d_c = {d_c} must be in (0, {n}]")
    if not 0 < r < d_k or r % 2:
        raise ValueError(f"r = {r}: this construction needs an even partial rotary width")
    P = r // 2
    S_h, S_ih, n_floored = dt_decoupled._sigma_pair(Sigma_l, n)

    C, Mread, lam = content_basis(Qt, Kt, Vt, Wo, GK_l, GV_l, S_h, S_ih, d_c, r)
    W_UK = np.concatenate([Kt[i][r:] @ Mread for i in range(H)], axis=0)
    W_UV = np.concatenate([Vt[i] @ Mread for i in range(H)], axis=0)
    W_UK = np.concatenate([W_UK, np.zeros((W_UK.shape[0], 1))], axis=1)
    W_UV = np.concatenate([W_UV, np.zeros((W_UV.shape[0], 1))], axis=1)
    W_UQ = Qt[:, r:, :].reshape(H * (d_k - r), n)

    Vh, mu, vp = plane_spectra(Kt, GK_l, S_h, r, phase_fix=phase_fix)
    m, alloc = allocate(mu, w_rot // 2)
    n_degen = 0
    for p in range(P):
        sv = np.sqrt(mu[p][:int(m[p])])
        if len(sv) >= 2:
            rel = (sv[:-1] - sv[1:]) / np.maximum(sv[:-1], 1e-300)
            k = int((rel < DEGENERATE_REL).sum())
            if k:
                n_degen += k
                warnings.warn(f"plane {p}: {k} near-degenerate kept singular value pair(s) (rel gap "
                              f"{rel.min():.2e} < {DEGENERATE_REL:g}); the D5 phase does not pin "
                              f"the basis inside a degenerate subspace (the function is unaffected)")
    W_KR, W_QR, Psi, slot_plane, rdiag = rotary_maps(Qt, Kt, Vh, m, S_ih, Sigma_l, r)

    layer = {
        "slot_plane": np.ascontiguousarray(slot_plane, dtype=np.int64),
        "slot_head": np.full(w_rot, -1, dtype=np.int64),
        "dC": np.full(H, d_k - r, dtype=np.int64),
        "W_DKV": np.ascontiguousarray(C, dtype=np.float64),
        "W_KR": np.ascontiguousarray(W_KR, dtype=np.float64),
        "W_UQ": np.ascontiguousarray(W_UQ, dtype=np.float64),
        "W_QR": np.ascontiguousarray(W_QR, dtype=np.float64),
        "W_UK": np.ascontiguousarray(W_UK, dtype=np.float64),
        "W_UV": np.ascontiguousarray(W_UV, dtype=np.float64),
        "Psi": np.ascontiguousarray(Psi, dtype=np.float64),
    }
    lam_tail = float(lam[d_c:].sum())
    mu_tail = float(sum(mu[p][int(m[p]):].sum() for p in range(P)))
    c_gap = float(lam[d_c - 1] - lam[d_c]) if d_c < len(lam) else float("nan")
    meta = {
        "m_p": m.tolist(),
        "mu": [mp_.tolist() for mp_ in mu],
        "lam": lam.tolist(),
        "rot_boundary_mu_in": alloc["boundary_mu_in"],
        "rot_boundary_mu_out": alloc["boundary_mu_out"],
        "boundary_gap": alloc["boundary_gap"],
        "content_cut_lam_in": float(lam[d_c - 1]),
        "content_cut_lam_out": float(lam[d_c]) if d_c < len(lam) else 0.0,
        "content_cut_gap": c_gap,
        "n_floored": int(n_floored),
        "varpi_min": float(vp.min()),
        "varpi_median": float(np.median(vp)),
        "tsigt_err_max": float(max(rdiag["tsigt_err"])),
        "psi_consistency": rdiag["psi_consistency"],
        "n_degenerate_kept": n_degen,
        "t0_proxy_damage_content": lam_tail,
        "t0_proxy_damage_rotary": mu_tail,
        "t0_proxy_damage": lam_tail + mu_tail,
    }
    return layer, meta

def _check_inputs(model_geom, w, d_c, G, fisher, sigma, s2norm):
    L, H, d_k, d_v, dm, r = model_geom
    if s2norm is not None:
        raise ValueError("ours takes no s2norm (MHA2MLA's input); an unneeded input is refused")
    if fisher is None or sigma is None:
        raise ValueError("ours needs both fisher=(GK, GV) and sigma (the Fisher+Sigma objective)")
    if len(fisher) != 2:
        raise ValueError("fisher must be the pair (GK, GV)")
    GK, GV = (np.asarray(a) for a in fisher)
    if GK.shape != (L, H, d_k, d_k) or GV.shape != (L, H, d_v, d_v):
        raise ValueError(f"fisher shapes {GK.shape}, {GV.shape}; expected ({L},{H},{d_k},{d_k}) "
                         f"and ({L},{H},{d_v},{d_v})")
    if not (np.isfinite(GK).all() and np.isfinite(GV).all()):
        raise ValueError("fisher contains non-finite entries")
    if len(sigma) != L:
        raise ValueError(f"sigma has {len(sigma)} layers, the model has {L}")
    for li, S in enumerate(sigma):
        if np.shape(S) != (dm + 1, dm + 1):
            raise ValueError(f"sigma[{li}] is {np.shape(S)}, expected the augmented ({dm + 1}, {dm + 1})")
    if isinstance(w, bool) or not isinstance(w, (int, np.integer)) or \
            isinstance(d_c, bool) or not isinstance(d_c, (int, np.integer)):
        raise TypeError(f"w and d_c must be ints, got {type(w).__name__}, {type(d_c).__name__}")
    w, d_c = int(w), int(d_c)
    if w <= 0 or w % 2:
        raise ValueError(f"w = {w} must be even and positive (D4: w = 0 is never built)")
    if w // 2 > H * (r // 2):
        raise ValueError(f"w/2 = {w // 2} complex directions > H * r/2 = {H * (r // 2)} available")
    if not 0 < d_c <= dm + 1:
        raise ValueError(f"d_c = {d_c} must be in (0, {dm + 1}]")
    if G is not None:
        if isinstance(G, bool) or not isinstance(G, (int, np.integer)):
            raise TypeError(f"G must be an int or None, got {type(G).__name__}")
        B = int(G) * (d_k + d_v)
        if w + d_c != B:
            raise ValueError(f"w + d_c = {w} + {d_c} = {w + d_c} != B = G*(d_k+d_v) = {B}: the "
                             f"cache is not the matched budget")
    return GK, GV, w, d_c

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    geom = gc.geometry(model)
    L, H, d_k, d_v, dm, r = geom
    if d_k != d_v:
        raise ValueError(f"d_k = {d_k} != d_v = {d_v}; the schema's d_h is one number")
    GK, GV, w, d_c = _check_inputs(geom, w, d_c, G, fisher, sigma, s2norm)
    layers_mod = kvlab.get_layers(model)
    layers, metas = [], []
    for li, lay in enumerate(layers_mod):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        ld, md = build_layer(Qt, Kt, Vt, Wo, GK[li], GV[li], sigma[li], w_rot=w, d_c=d_c, r=r)
        layers.append(ld)
        metas.append(md)
    B = w + d_c
    scale = mla_scale_of(w, d_k)
    prov_text = (
        f"{ARM} ({VERSION}): content = top-{d_c} right singular vectors of the Fisher-weighted "
        f"joint K^nr/V stack (dt_decoupled._joint_stack, K/V exchange rate {KV_EXCHANGE_RATE}), "
        f"Sigma-whitened (pseudo-inverse, dt_decoupled._sigma_pair; "
        f"{sum(md['n_floored'] for md in metas)} directions floored over {L} layers), served "
        f"basis QR'd to orthonormal rows as dt_decoupled.py:352, read-out Sigma C^T (C Sigma C^T)^-1 "
        f"via Mread; rotary = global top {w // 2} complex directions of the per-plane "
        f"varpi-weighted Sigma-whitened key stacks (varpi = commutant half-trace of GK), Vh rows "
        f"phase-pinned (D5), psi solved per plane, split-half plane-major slots, shared k^R; "
        f"biases {BIAS_SET}; scale {scale}")
    out = {
        "schema": SCHEMA, "arm": ARM, "version": VERSION, "rotary_mode": ROTARY_MODE,
        "G": None if G is None else int(G), "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_k), "dm": int(dm), "r": int(r),
        "provenance": {"fisher_sha256": _digest((GK, GV)),
                       "sigma_sha256": _digest(sigma),
                       "s2norm_sha256": None,
                       "text": prov_text},
        "bias_set": sorted(BIAS_SET),
        "mla_scale": scale,
        "layers": layers,
        "_meta": {"layers": metas,
                  "t0_proxy_damage": float(sum(md["t0_proxy_damage"] for md in metas)),
                  "n_degenerate_kept": int(sum(md["n_degenerate_kept"] for md in metas)),
                  "varpi_min": float(min(md["varpi_min"] for md in metas)),
                  "tsigt_err_max": float(max(md["tsigt_err_max"] for md in metas))},
    }
    self_check(out)
    return out

def self_check(wd):
    if wd["schema"] != SCHEMA or wd["arm"] != ARM or wd["rotary_mode"] != ROTARY_MODE:
        raise AssertionError("schema / arm / rotary_mode mismatch")
    if wd["bias_set"] != sorted(BIAS_SET) or wd["mla_scale"] != mla_scale_of(wd["w"], wd["d_h"]):
        raise AssertionError("bias_set / mla_scale mismatch")
    w, d_c, H, r, dm, d_h = wd["w"], wd["d_c"], wd["H"], wd["r"], wd["dm"], wd["d_h"]
    n, nh, P = dm + 1, w // 2, r // 2
    if w % 2 or w + d_c != wd["B"] or (wd["G"] is not None and wd["B"] != wd["G"] * 2 * d_h):
        raise AssertionError("width rule violated")
    D = H * (d_h - r)
    if D == 0:
        raise AssertionError(f"d_h - r = 0 (r = {r} of d_h = {d_h}): W_UQ and W_UK have zero rows, "
                             f"so the content latent carries no key information and every array "
                             f"check below passes on an empty array. A full-RoPE build must come "
                             f"through the un-roping path, which gives a positive content width.")
    if w == 0:
        raise AssertionError("w = 0: slot_plane / slot_head / Psi's column axis are empty, so the "
                             "slot and Psi checks below are vacuous. D4 forbids w = 0.")
    shapes = {"slot_plane": (w,), "slot_head": (w,), "dC": (H,), "W_DKV": (d_c, n),
              "W_KR": (w, n), "W_UQ": (D, n), "W_QR": (H * w, n), "W_UK": (D, d_c + 1),
              "W_UV": (H * d_h, d_c + 1), "Psi": (H, r, w)}
    for li, lay in enumerate(wd["layers"]):
        if set(lay) != set(shapes):
            raise AssertionError(f"layer {li}: keys {sorted(lay)} != {sorted(shapes)}")
        for k, shp in shapes.items():
            a = lay[k]
            if not isinstance(a, np.ndarray) or a.shape != shp or not a.flags["C_CONTIGUOUS"]:
                raise AssertionError(f"layer {li} {k}: {getattr(a, 'shape', type(a))} != {shp}")
            want = np.int64 if k in ("slot_plane", "slot_head", "dC") else np.float64
            if a.dtype != want:
                raise AssertionError(f"layer {li} {k}: dtype {a.dtype} != {np.dtype(want)}")
            if a.dtype == np.float64 and not np.isfinite(a).all():
                raise AssertionError(f"layer {li} {k}: non-finite")
        sp = lay["slot_plane"]
        if not np.array_equal(sp[:nh], sp[nh:]) or sp.min() < 0 or sp.max() >= P:
            raise AssertionError(f"layer {li}: slot pair rule / plane range violated")
        if np.any(np.diff(sp[:nh]) < 0):
            raise AssertionError(f"layer {li}: slots are not plane-major")
        if np.bincount(sp[:nh], minlength=P).max() > H:
            raise AssertionError(f"layer {li}: more than H pairs on one plane")
        if not np.all(lay["slot_head"] == -1) or not np.all(lay["dC"] == d_h - r):
            raise AssertionError(f"layer {li}: slot_head / dC wrong for shared mode")
        if np.any(lay["W_UK"][:, -1] != 0.0) or np.any(lay["W_UV"][:, -1] != 0.0):
            raise AssertionError(f"layer {li}: a bias column outside bias_set is not exactly 0")
        Ps = lay["Psi"]
        planes = np.arange(r) % P
        off = planes[:, None] != sp[None, :]
        if np.any(Ps[:, off] != 0.0):
            raise AssertionError(f"layer {li}: Psi is not block-sparse by plane")
        s = np.arange(nh)
        pp = sp[:nh]
        if not (np.array_equal(Ps[:, pp, s], Ps[:, pp + P, s + nh]) and
                np.array_equal(Ps[:, pp + P, s], -Ps[:, pp, s + nh])):
            raise AssertionError(f"layer {li}: Psi is not complex-linear (split-half pattern)")
    return True
