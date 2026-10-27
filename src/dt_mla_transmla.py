import hashlib

import numpy as np

import gam_core as gc
import kvlab

SCHEMA = "mla/v1"
ARM = "mla_transmla_sigma"
ARM_NOQUIRK = "mla_transmla_sigma_noquirk"
VERSION = "transmla-2026-09-29.1"
ROTARY_MODE = "shared"
BIAS_SET = ["dkv"]
PIN = "MuLabPKU/TransArch/TransMLA_NeurIPS_2025@eaf4dde7a9b0592566166fd347959738e70ecefd"
DAMP_FRAC = 0.01
CANON_REL = 1e-6
ALPHA_RULES = ("code", "noquirk")
SIGMA_SYM_TOL = 1e-12
SIGMA_PSD_TOL = 1e-10
PSI_TOL = 1e-12

_PAIRINGS = ("split_half", "interleaved")

def sigma_digest(sigma):
    h = hashlib.sha256()
    for S in sigma:
        h.update(np.ascontiguousarray(S, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]

def canon_sign(V, rel=CANON_REL):
    V = np.array(V, dtype=np.float64, copy=True)
    a = np.abs(V)
    mx = a.max(axis=0)
    if np.any(mx == 0.0):
        raise ValueError("canon_sign: a zero column has no sign to canonicalise")
    piv = np.argmax(a >= (1.0 - rel) * mx[None, :], axis=0)
    flip = V[piv, np.arange(V.shape[1])] < 0
    V[:, flip] *= -1.0
    return V

def eigh_desc(Hm, damp_frac=DAMP_FRAC):
    Hm = np.asarray(Hm, dtype=np.float64)
    Hm = 0.5 * (Hm + Hm.T)
    damp = damp_frac * float(np.mean(np.diag(Hm)))
    if not (np.isfinite(damp) and damp > 0.0):
        raise ValueError(f"eigh_desc: damping {damp!r} is not a positive finite number")
    lam_d, V = np.linalg.eigh(Hm + damp * np.eye(Hm.shape[0]))
    order = np.argsort(-lam_d, kind="stable")
    return lam_d[order] - damp, canon_sign(V[:, order]), damp

def mla_scale_of(w, d_h):
    return float((d_h + w) ** -0.5)

def s_factor(w, d_h):
    return float(np.sqrt((d_h + w) / d_h))

def pairing_dims(r, pairing="split_half"):
    h = r // 2
    if pairing == "split_half":
        return np.arange(h), np.arange(h) + h
    if pairing == "interleaved":
        return 2 * np.arange(h), 2 * np.arange(h) + 1
    raise ValueError(f"unknown pairing {pairing!r}; expected one of {_PAIRINGS}")

def rotation_matrix(U, H, d_h, r, pairing="split_half"):
    xdim, ydim = pairing_dims(r, pairing)
    T = np.eye(H * d_h, dtype=np.float64)
    heads = np.arange(H) * d_h
    for p in range(r // 2):
        for d in (int(xdim[p]), int(ydim[p])):
            T[np.ix_(heads + d, heads + d)] = U[p].T
    return T

def allocate(lam_rot, m):
    h, H = lam_rot.shape
    if not (1 <= m <= h * H):
        raise ValueError(f"cannot keep {m} complex components of {h} planes x {H} heads")
    P, C = np.meshgrid(np.arange(h), np.arange(H), indexing="ij")
    P, C, lam = P.ravel(), C.ravel(), lam_rot.ravel()
    order = np.lexsort((C, P, -lam))
    top = order[:m]
    m_p = np.bincount(P[top], minlength=h).astype(np.int64)
    for p in range(h):
        got = np.sort(C[top][P[top] == p])
        if not np.array_equal(got, np.arange(m_p[p])):
            raise AssertionError(f"plane {p} kept components {got.tolist()}, not a prefix; the "
                                 f"eigenvalues are not sorted descending within the plane")
    slots = [(p, c) for p in range(h) for c in range(int(m_p[p]))]
    last = float(lam[order[m - 1]])
    nxt = float(lam[order[m]]) if m < lam.size else float("nan")
    top_lam = float(lam[order[0]])
    cut = {"lam_last_kept": last, "lam_first_dropped": nxt, "gap": last - nxt,
           "rel_gap": (last - nxt) / top_lam if top_lam > 0 else float("nan"),
           "tie": bool(m < lam.size and last == nxt)}
    return m_p, slots, cut

def _check_sigma(S, n, l):
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (n, n):
        raise ValueError(f"sigma[{l}] is {S.shape}, expected the AUGMENTED ({n}, {n})")
    if not np.isfinite(S).all():
        raise ValueError(f"sigma[{l}] has non-finite entries")
    if abs(S[-1, -1] - 1.0) > 1e-9:
        raise ValueError(f"sigma[{l}][-1, -1] = {S[-1, -1]:.6g}; it is E[1*1] of the augmented "
                         f"constant coordinate and must be 1 -- this Sigma lacks the bias column")
    scale = float(np.abs(S).max())
    asym = float(np.abs(S - S.T).max())
    if asym > SIGMA_SYM_TOL * scale:
        raise ValueError(f"sigma[{l}] is not symmetric: max|S - S^T| = {asym:.3g} "
                         f"(scale {scale:.3g})")
    S = 0.5 * (S + S.T)
    ev = np.linalg.eigvalsh(S)
    if ev[0] < -SIGMA_PSD_TOL * ev[-1]:
        raise ValueError(f"sigma[{l}] is not PSD: min eigenvalue {ev[0]:.3g}, max {ev[-1]:.3g}")
    return S

def _check_widths(w, d_c, G, H, d_h, r):
    for name, v in (("w", w), ("d_c", d_c)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"{name} must be an int, got {type(v).__name__}")
    if G is not None and (isinstance(G, bool) or not isinstance(G, (int, np.integer))):
        raise TypeError(f"G must be an int or None, got {type(G).__name__}")
    w, d_c = int(w), int(d_c)
    if w <= 0:
        raise ValueError(f"w = {w}: TransMLA needs a rotary key (D4: fail loud, never build w = 0)")
    if w % 2:
        raise ValueError(f"w = {w} is odd; the rotary key is split-half slot pairs")
    if w > H * r:
        raise ValueError(f"w = {w} exceeds H * r = {H * r}, the number of rotary rows that exist")
    n_rows = 2 * H * d_h - w
    if not (1 <= d_c <= n_rows):
        raise ValueError(f"d_c = {d_c} outside [1, {n_rows}] (2 H d_h - w rows of M)")
    B = w + d_c
    if G is not None:
        if int(G) <= 0:
            raise ValueError(f"G = {G} must be positive")
        if B != int(G) * 2 * d_h:
            raise ValueError(f"w + d_c = {B} != G * (d_k + d_v) = {int(G) * 2 * d_h}: the cache is "
                             f"not matched to budget G = {G}")
    return w, d_c, B

def build_layer(Qt, Kt, Vt, S, *, w, d_c, alpha_rule="code", r, _pairing="split_half",
                _internals=False):
    if alpha_rule not in ALPHA_RULES:
        raise ValueError(f"alpha_rule {alpha_rule!r} not in {ALPHA_RULES}")
    Qt, Kt, Vt, S = (np.asarray(a, dtype=np.float64) for a in (Qt, Kt, Vt, S))
    H, d_h, n = Qt.shape
    if Kt.shape != (H, d_h, n) or Vt.shape != (H, d_h, n):
        raise ValueError(f"Qt {Qt.shape}, Kt {Kt.shape}, Vt {Vt.shape}: expected three (H, d_h, dm+1) "
                         f"with d_v == d_k")
    if S.shape != (n, n):
        raise ValueError(f"Sigma {S.shape} does not match the maps' augmented width {n}")
    if r % 2 or not (2 <= r <= d_h):
        raise ValueError(f"rotary width r = {r} must be even and in [2, d_h = {d_h}]")
    h, m, D = r // 2, w // 2, H * d_h
    if w % 2 or not (2 <= w <= H * r) or not (1 <= d_c <= 2 * D - w):
        raise ValueError(f"widths w = {w}, d_c = {d_c} invalid for H = {H}, d_h = {d_h}, r = {r}")
    xdim, ydim = pairing_dims(r, _pairing)

    Kf = Kt.reshape(D, n)
    KS = Kf @ S
    lam_rot = np.zeros((h, H))
    U = np.zeros((h, H, H))
    damp_rot = np.zeros(h)
    Hrot = np.zeros((h, H, H))
    for p in range(h):
        ix = np.arange(H) * d_h + int(xdim[p])
        iy = np.arange(H) * d_h + int(ydim[p])
        Hp = KS[ix] @ Kf[ix].T + KS[iy] @ Kf[iy].T
        Hp = 0.5 * (Hp + Hp.T)
        Hrot[p] = Hp
        lam_rot[p], U[p], damp_rot[p] = eigh_desc(Hp)

    m_p, slots, cut = allocate(lam_rot, m)
    slot_plane = np.array([p for (p, _) in slots] * 2, dtype=np.int64)

    T = rotation_matrix(U, H, d_h, r, _pairing)
    Kr = T @ Kf
    KrS = T @ KS
    kept = np.array([c * d_h + int(xdim[p]) for (p, c) in slots]
                    + [c * d_h + int(ydim[p]) for (p, c) in slots], dtype=np.int64)
    notkept = np.setdiff1d(np.arange(D), kept)
    N = notkept.size
    if N != D - w:
        raise AssertionError(f"kept rows are not distinct: {D - N} kept for w = {w}")

    kd = np.einsum("ij,ij->i", Kr[notkept], KrS[notkept])
    Vf = Vt.reshape(D, n)
    VS = Vf @ S
    vd = np.einsum("ij,ij->i", Vf, VS)
    for name, dg in (("K_N", kd), ("V", vd)):
        if dg.min() < -1e-12 * max(dg.max(), 1e-300):
            raise ValueError(f"diag({name} S {name}^T) has a negative entry {dg.min():.3g}: "
                             f"Sigma is not PSD on these rows")
    kn, vn = np.sqrt(np.clip(kd, 0.0, None)), np.sqrt(np.clip(vd, 0.0, None))
    alpha_code = float(kn.mean() / vn[w:].mean())
    alpha_noquirk = float(kn.mean() / vn.mean())
    alpha_rms = float(np.sqrt(np.mean(kd) / np.mean(vd)))
    alpha = alpha_code if alpha_rule == "code" else alpha_noquirk
    if not (np.isfinite(alpha) and alpha > 0.0):
        raise ValueError(f"alpha = {alpha!r} is not a positive finite ratio")

    M = np.concatenate([Kr[notkept] / alpha, Vf], axis=0)
    MS = np.concatenate([KrS[notkept] / alpha, VS], axis=0)
    Ckv = MS @ M.T
    lam_kv, R, damp_kv = eigh_desc(Ckv)
    R_r = R[:, :d_c]
    W_DKV = R_r.T @ M
    A = T.T
    W_UK = alpha * (A[:, notkept] @ R_r[:N])
    W_UV = R_r[N:]

    s = s_factor(w, d_h)
    W_UQ = s * Qt.reshape(D, n)
    A_R = A[:, kept]
    W_QR = np.concatenate([s * (A_R[i * d_h:(i + 1) * d_h].T @ Qt[i]) for i in range(H)], axis=0)

    Psi = np.zeros((H, r, w))
    for sl, (p, c) in enumerate(slots):
        Psi[:, int(xdim[p]), sl] = U[p][:, c]
        Psi[:, int(ydim[p]), sl + m] = U[p][:, c]
    for i in range(H):
        ref = s * (Psi[i].T @ Qt[i][:r])
        got = W_QR[i * w:(i + 1) * w]
        err = float(np.abs(got - ref).max())
        if err > PSI_TOL * max(float(np.abs(ref).max()), 1e-300):
            raise AssertionError(f"head {i}: W_QR disagrees with s Psi^T Qt[:r] by {err:.3g}; the "
                                 f"kept rows and Psi are out of step")

    zc = np.zeros((D, 1))
    layer = {
        "slot_plane": np.ascontiguousarray(slot_plane, dtype=np.int64),
        "slot_head": np.full(w, -1, dtype=np.int64),
        "dC": np.full(H, d_h, dtype=np.int64),
        "W_DKV": np.ascontiguousarray(W_DKV, dtype=np.float64),
        "W_KR": np.ascontiguousarray(Kr[kept], dtype=np.float64),
        "W_UQ": np.ascontiguousarray(W_UQ, dtype=np.float64),
        "W_QR": np.ascontiguousarray(W_QR, dtype=np.float64),
        "W_UK": np.ascontiguousarray(np.hstack([W_UK, zc]), dtype=np.float64),
        "W_UV": np.ascontiguousarray(np.hstack([W_UV, zc]), dtype=np.float64),
        "Psi": np.ascontiguousarray(Psi, dtype=np.float64),
    }
    kv_tot = float(np.clip(lam_kv, 0.0, None).sum())
    meta = {
        "m_p": m_p.tolist(),
        "lam_rot": lam_rot,
        "damp_rot": damp_rot,
        "U": U,
        "cut": cut,
        "native_allocation": bool(w == r and np.all(m_p == 1)),
        "alpha_code": alpha_code, "alpha_noquirk": alpha_noquirk, "alpha_rms": alpha_rms,
        "alpha_rule": alpha_rule, "alpha_used": alpha,
        "s": s,
        "lam_kv_cut": (float(lam_kv[d_c - 1]),
                       float(lam_kv[d_c]) if d_c < lam_kv.size else float("nan")),
        "lam_kv_dropped_share": (float(np.clip(lam_kv[d_c:], 0.0, None).sum()) / kv_tot
                                 if kv_tot > 0 else float("nan")),
        "damp_kv": damp_kv,
        "n_lam_kv_negative": int((lam_kv < -1e-12 * lam_kv[0]).sum()),
    }
    if not _internals:
        return layer, meta
    internals = {"T": T, "Kr": Kr, "kept": kept, "notkept": notkept, "R_r": R_r, "lam_kv": lam_kv,
                 "M": M, "Hrot": Hrot, "Ckv": Ckv, "alpha": alpha}
    return layer, meta, internals

def _geometry(model):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    layers = kvlab.get_layers(model)
    rn = getattr(layers[0].attention, "rotary_ndims", None)
    if rn is None or int(rn) != r:
        raise ValueError(f"cannot confirm the rotary width: attention.rotary_ndims = {rn!r}, "
                         f"gam_core.geometry says r = {r}")
    th = gc.inv_freq(model)
    if 2 * th.size != r:
        raise ValueError(f"{th.size} rotary frequencies for r = {r}")
    if d_k != d_v:
        raise ValueError(f"d_k = {d_k} != d_v = {d_v}")
    return L, H, d_k, dm, r, layers

def _build(model, *, w, d_c, G, fisher, sigma, s2norm, alpha_rule, arm):
    if fisher is not None:
        raise ValueError("TransMLA uses no Fisher (its objective is alpha^-2 Pi_N on keys, I on "
                         "values).  _build_mla must drop the global metric for the baselines "
                         "(build plan §5, adversarial C6) -- it reached the initialiser")
    if s2norm is not None:
        raise ValueError("s2norm is MHA2MLA's input; TransMLA must not receive it")
    if sigma is None:
        raise ValueError("TransMLA is calibrated on the teacher's SIGMA (D-M5); sigma is required")
    L, H, d_h, dm, r, layers = _geometry(model)
    w, d_c, B = _check_widths(w, d_c, G, H, d_h, r)
    if len(sigma) != L:
        raise ValueError(f"sigma has {len(sigma)} layers, the model {L}")
    n = dm + 1
    out_layers, metas = [], []
    for l in range(L):
        S = _check_sigma(sigma[l], n, l)
        Qt, Kt, Vt, _ = gc.read_layer(layers[l], H, d_h, d_h, dm)
        lay, met = build_layer(Qt, Kt, Vt, S, w=w, d_c=d_c, alpha_rule=alpha_rule, r=r)
        out_layers.append(lay)
        metas.append(met)
    return {
        "schema": SCHEMA, "arm": arm, "version": VERSION, "rotary_mode": ROTARY_MODE,
        "G": None if G is None else int(G), "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_h), "dm": int(dm), "r": int(r),
        "provenance": {
            "fisher_sha256": None,
            "sigma_sha256": sigma_digest(sigma),
            "s2norm_sha256": None,
            "text": (f"TransMLA RoRoPE + BKV, reimplemented from {PIN} (partial_rope.py, lora_qkv.py, "
                     f"utils.py, transformers/mla.py) on the teacher's SIGMA (D-M5); alpha rule "
                     f"'{alpha_rule}' ({'V-slice quirk of lora_qkv.py:112 kept' if alpha_rule == 'code' else 'quirk-free: all V rows'}); "
                     f"global top-w/2 allocation on undamped eigenvalues (Eq. 19), FreqFold off, "
                     f"interleave permutation skipped; s = sqrt((d_h+w)/d_h) in the query weights "
                     f"and biases, mla_scale = (d_h+w)^-1/2 (D1)"),
        },
        "bias_set": list(BIAS_SET),
        "mla_scale": mla_scale_of(w, d_h),
        "layers": out_layers,
        "_meta": {"alpha_rule": alpha_rule, "pin": PIN, "per_layer": metas},
    }

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm,
                  alpha_rule="code", arm=ARM)

def build_weights_noquirk(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm,
                  alpha_rule="noquirk", arm=ARM_NOQUIRK)
