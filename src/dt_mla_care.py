import hashlib

import numpy as np

import dt_mla_transmla as tm
from dt_mla_transmla import canon_sign, mla_scale_of, s_factor, sigma_digest

SCHEMA = "mla/v1"
ARM = "mla_care_sigma"
VERSION = "care-2026-09-30.1"
ROTARY_MODE = "shared"
BIAS_SET = ["dkv"]
PIN = "FutureMLS-Lab/CARE@3b4534644fd2c03c549b34c49c99990a6f760e3e"

POWER = 4.0
POWER_SQRT = 0.5
PERCDAMP = 0.01
EIGH_EPS = 1e-8
CARE_DAMP_MULT = 10.0
DAMP_MODES = ("f64", "f32x10")
DECOMP_METHOD = "transmla-care"

ARM_SQRT = "mla_care_sigma_p05"
ARM_NOQUIRK = "mla_care_sigma_noquirk"
ARM_F32DAMP10 = "mla_care_sigma_f32damp10"
ARM_RELEASE = "mla_care_sigma_release"

SYM_TOL = 1e-10
CONTENT_TOL = 1e-10
DETERMINED_EPS_MULT = 1e4
F64_EPS = float(np.finfo(np.float64).eps)
SOLVERS = ("stable", "release")
SOLVER = "stable"
JSV_JOBA = 0
STABLE_REL_GAP_MIN = 1e-10

def damped_psd_eigh(Hz, percdamp=PERCDAMP, eps=EIGH_EPS, damp_mode="f64"):
    if damp_mode not in DAMP_MODES:
        raise ValueError(f"damp_mode {damp_mode!r} not in {DAMP_MODES}")
    Hz = np.asarray(Hz, dtype=np.float64)
    if Hz.ndim != 2 or Hz.shape[0] != Hz.shape[1]:
        raise ValueError(f"damped_psd_eigh: H must be square, got {Hz.shape}")
    if not np.isfinite(Hz).all():
        raise ValueError("damped_psd_eigh: H has non-finite entries")
    scale = float(np.abs(Hz).max())
    asym = float(np.abs(Hz - Hz.T).max())
    if scale <= 0.0:
        raise ValueError("damped_psd_eigh: H is identically zero")
    if asym > SYM_TOL * scale:
        raise ValueError(f"damped_psd_eigh: H is not symmetric to roundoff (max|H - H^T| = {asym:.3g}, "
                         f"scale {scale:.3g}); it is not a second moment")
    mult = 1.0 if damp_mode == "f64" else CARE_DAMP_MULT
    work = Hz if damp_mode == "f64" else Hz.astype(np.float32)
    damp = float(percdamp * np.mean(np.diag(work)) * mult)
    if not (np.isfinite(damp) and damp > 0.0):
        raise ValueError(f"damped_psd_eigh: damping {damp!r} is not a positive finite number")
    A = np.array(work, copy=True)
    ix = np.arange(A.shape[0])
    A[ix, ix] += np.asarray(damp, dtype=A.dtype)
    ev, V = np.linalg.eigh(A)
    ev = ev.astype(np.float64)
    n_clamped = int((ev < eps).sum())
    if n_clamped:
        raise ValueError(
            f"damped_psd_eigh: the eigenvalue clamp min={eps:g} fired on {n_clamped} of {ev.size} "
            f"eigenvalues (min {ev.min():.3g}, damp {damp:.3g}). That constant is ABSOLUTE while our "
            f"SIGMA is per-token normalised and CARE's H is a raw token sum, so a clamp that fires "
            f"makes the normalisation change the basis. Do not paper over it.")
    info = {"damp": damp, "damp_mode": damp_mode, "n_clamped": n_clamped,
            "ev_min": float(ev.min()), "ev_max": float(ev.max()),
            "n_ev_below_damp": int((ev < 2.0 * damp).sum()), "asym_rel": asym / scale,
            "mean_diag": float(np.mean(np.diag(work)))}
    return ev, V.astype(np.float64), info

def matrix_power_from_eigh(ev, V, power):
    ev = np.asarray(ev, dtype=np.float64)
    P = (np.asarray(V, dtype=np.float64) * (ev ** float(power))[None, :]) @ np.asarray(V, dtype=np.float64).T
    if not np.isfinite(P).all():
        raise ValueError(f"matrix_power_from_eigh: H**{power} overflowed (ev max {ev.max():.3g})")
    return P

def svd_project_basis(W, Hz, rank, *, percdamp=PERCDAMP, power=POWER, eps=EIGH_EPS,
                      damp_mode="f64", canon=True, solver=SOLVER):
    if solver not in SOLVERS:
        raise ValueError(f"solver {solver!r} is not one of {SOLVERS}")
    W = np.asarray(W, dtype=np.float64)
    rank = int(rank)
    ev, V, info = damped_psd_eigh(Hz, percdamp=percdamp, eps=eps, damp_mode=damp_mode)
    if not (1 <= rank <= min(W.shape[1], ev.size)):
        raise ValueError(f"svd_project_basis: rank {rank} outside [1, {min(W.shape[1], ev.size)}]")
    if W.shape[1] != ev.size:
        raise ValueError(f"svd_project_basis: W has {W.shape[1]} input dims, H is {ev.size}x{ev.size}")
    if solver == "release":
        Hp = matrix_power_from_eigh(ev, V, power)
        y = Hp @ W.T
        if not np.isfinite(y).all():
            raise ValueError(f"svd_project_basis: H**{power} W^T is not finite")
        U, S, _ = np.linalg.svd(y, full_matrices=False)
        basis = np.ascontiguousarray(U[:, :rank])
    else:
        from scipy.linalg import lapack
        V = np.asarray(V, dtype=np.float64)
        X = np.asfortranarray((W @ V) * ((ev / ev.max()) ** float(power))[None, :])
        if X.shape[0] < X.shape[1]:
            raise ValueError(f"dgejsv needs rows >= cols; X is {X.shape}")
        sva, _u, v, work, _iw, info_j = lapack.dgejsv(X, joba=JSV_JOBA, jobu=3, jobv=0)
        if info_j != 0:
            raise RuntimeError(f"dgejsv failed (info {info_j})")
        order = np.argsort(-sva, kind="stable")
        S = sva[order] * (work[0] / work[1])
        basis = np.ascontiguousarray(V @ v[:, order[:rank]])
        orth = float(np.abs(basis.T @ basis - np.eye(rank)).max())
        orth_tol = 1e-8 if damp_mode == "f64" else 1e-5
        if orth > orth_tol:
            raise AssertionError(f"stable basis is not orthonormal ({orth:.3g} > {orth_tol:g}, "
                                 f"damp_mode {damp_mode})")
    if canon:
        basis = np.ascontiguousarray(canon_sign(basis))
    s_kept = S[:rank]
    tail = S[rank:]
    cut_rel = float(s_kept[-1] / S[0]) if S[0] > 0 else float("nan")
    rel_gap = float((s_kept[-1] - tail[0]) / s_kept[-1]) if tail.size and s_kept[-1] > 0 else float("nan")
    info.update(power=float(power), rank=rank, canon=bool(canon), solver=solver,
                sigma0=float(S[0]), sigma_last_kept=float(s_kept[-1]),
                sigma_first_dropped=float(tail[0]) if tail.size else float("nan"),
                sigma_min=float(S[-1]),
                cut_rel_gap=float((s_kept[-1] - tail[0]) / S[0]) if tail.size and S[0] > 0
                else float("nan"),
                cut_gap_rel_to_cut=rel_gap,
                sigma_cut_over_sigma0=cut_rel, sigma_cut_eps=cut_rel / F64_EPS,
                basis_determined=bool(cut_rel / F64_EPS >= DETERMINED_EPS_MULT) if solver == "release"
                else bool(not tail.size or rel_gap > STABLE_REL_GAP_MIN))
    return basis, S, info

def join_up_projection(A, notkept, alpha, H, d_h):
    A = np.asarray(A, dtype=np.float64)
    notkept = np.asarray(notkept, dtype=np.int64)
    D, N = H * d_h, int(notkept.size)
    if A.shape != (D, D):
        raise ValueError(f"join_up_projection: A is {A.shape}, expected ({D}, {D})")
    AK = float(alpha) * A[:, notkept]
    Wj = np.zeros((2 * D, N + D), dtype=np.float64)
    for i in range(H):
        b = 2 * i * d_h
        Wj[b:b + d_h, :N] = AK[i * d_h:(i + 1) * d_h]
        Wj[b + d_h:b + 2 * d_h, N + i * d_h:N + (i + 1) * d_h] = np.eye(d_h)
    return Wj

def content_from_basis(R, M, Wj, H, d_h):
    R = np.asarray(R, dtype=np.float64)
    W_DKV = R.T @ np.asarray(M, dtype=np.float64)
    UB = np.asarray(Wj, dtype=np.float64) @ R
    W_UK = np.concatenate([UB[2 * i * d_h:2 * i * d_h + d_h] for i in range(H)], axis=0)
    W_UV = np.concatenate([UB[2 * i * d_h + d_h:2 * (i + 1) * d_h] for i in range(H)], axis=0)
    if W_UK.shape != (H * d_h, R.shape[1]) or W_UV.shape != W_UK.shape:
        raise AssertionError(f"content_from_basis: W_UK {W_UK.shape}, W_UV {W_UV.shape}")
    return W_DKV, W_UK, W_UV

def rank_of_augmented_map(M, dm):
    M = np.asarray(M, dtype=np.float64)
    n = dm + 1
    if M.shape[1] != n:
        raise ValueError(f"rank_of_augmented_map: M has {M.shape[1]} columns, expected dm+1 = {n}")
    sv = np.linalg.svd(M, compute_uv=False)
    tol = float(sv[0]) * max(M.shape) * np.finfo(np.float64).eps
    rank = int((sv > tol).sum())
    info = {"rank": rank, "n": n, "sv_max": float(sv[0]), "sv_min": float(sv[-1]),
            "sv_at_n": float(sv[n - 1]) if sv.size >= n else float("nan"),
            "tol": tol, "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")}
    bias_free = bool(np.all(M[:, n - 1] == 0.0))
    rank_core = int((np.linalg.svd(M[:, :n - 1], compute_uv=False) > tol).sum())
    info["bias_free_augmented_column"] = bias_free
    info["rank_core"] = rank_core
    if rank != n and not (bias_free and rank == n - 1 and rank_core == n - 1):
        raise ValueError(
            f"rank(M) = {rank}, not dm+1 = {n} (sv_max {sv[0]:.3g}, sv[{n - 1}] {info['sv_at_n']:.3g}, "
            f"tol {tol:.3g}; augmented column exactly zero: {bias_free}, rank of the non-augmented "
            f"block: {rank_core}). CARE's metric H_z = M SIGMA M^T lives on the LATENT, and "
            f"||Delta H_z^1/2|| = ||Delta M SIGMA^1/2|| identifies it with the composite-map error "
            f"ONLY at full column rank, or at the one admissible deficiency above. Below that the "
            f"two objectives differ and this arm would not be "
            f"the method it claims to be (MODEL_GRID_PLAN.md §0.1b, the first rider).")
    return info

def build_layer(Qt, Kt, Vt, S, *, w, d_c, r, power=POWER, alpha_rule="code", percdamp=PERCDAMP,
                eps=EIGH_EPS, damp_mode="f64", canon=True, solver=SOLVER, _internals=False):
    Qt, Kt, Vt, S = (np.asarray(a, dtype=np.float64) for a in (Qt, Kt, Vt, S))
    H, d_h, n = Qt.shape
    dm = n - 1
    if not (np.isfinite(float(power)) and float(power) >= 0.0):
        raise ValueError(f"power {power!r} must be a finite non-negative float")
    lay, met, intl = tm.build_layer(Qt, Kt, Vt, S, w=w, d_c=d_c, alpha_rule=alpha_rule, r=r,
                                    _internals=True)
    T, kept, notkept = intl["T"], intl["kept"], intl["notkept"]
    M, alpha, R_tm = intl["M"], intl["alpha"], intl["R_r"]
    D, N = H * d_h, int(notkept.size)
    if M.shape != (N + D, n):
        raise AssertionError(f"M is {M.shape}, expected {(N + D, n)}")

    rank_info = rank_of_augmented_map(M, dm)

    A = T.T
    Wj = join_up_projection(A, notkept, alpha, H, d_h)
    MS = M @ S
    Hz = MS @ M.T
    R, svals, solve = svd_project_basis(Wj, Hz, d_c, percdamp=percdamp, power=power, eps=eps,
                                       damp_mode=damp_mode, canon=canon, solver=solver)
    W_DKV, W_UK, W_UV = content_from_basis(R, M, Wj, H, d_h)

    comp = Wj @ (R @ W_DKV)
    ref_k = np.concatenate([comp[2 * i * d_h:2 * i * d_h + d_h] for i in range(H)], axis=0)
    ref_v = np.concatenate([comp[2 * i * d_h + d_h:2 * (i + 1) * d_h] for i in range(H)], axis=0)
    for nm, got, ref in (("W_UK", W_UK @ W_DKV, ref_k), ("W_UV", W_UV @ W_DKV, ref_v)):
        dev = float(np.abs(got - ref).max())
        nrm = float(np.abs(ref).max())
        if dev > CONTENT_TOL * max(nrm, 1e-300):
            raise AssertionError(f"{nm} . W_DKV != the W_join R R^T M block ({dev:.3g} vs {nrm:.3g}): "
                                 f"the head-major de-interleaving is out of step")

    zc = np.zeros((D, 1))
    lay["W_DKV"] = np.ascontiguousarray(W_DKV, dtype=np.float64)
    lay["W_UK"] = np.ascontiguousarray(np.hstack([W_UK, zc]), dtype=np.float64)
    lay["W_UV"] = np.ascontiguousarray(np.hstack([W_UV, zc]), dtype=np.float64)

    def _damage(B):
        Z = Wj - (Wj @ B) @ B.T
        return float(np.einsum("ij,ij->", Z, Z @ Hz))
    tot = float(np.einsum("ij,ij->", Wj, Wj @ Hz))
    dmg_care, dmg_tm = _damage(R), _damage(R_tm)
    ang = np.linalg.svd(R.T @ R_tm, compute_uv=False)
    meta = dict(met)
    meta.update({
        "care_power": float(power), "care_percdamp": float(percdamp), "care_damp_mode": damp_mode,
        "care_canon": bool(canon), "care_decomp_method": DECOMP_METHOD, "care_solver": solver,
        "care_solve": solve, "care_rank_M": rank_info,
        "latent_damage_care": dmg_care, "latent_damage_transmla": dmg_tm,
        "latent_damage_total": tot,
        "latent_damage_care_rel": dmg_care / tot if tot > 0 else float("nan"),
        "latent_damage_transmla_rel": dmg_tm / tot if tot > 0 else float("nan"),
        "subspace_sv_min_vs_transmla": float(ang.min()), "subspace_sv_mean_vs_transmla": float(ang.mean()),
        "n_shared_directions_vs_transmla": int((ang > 1.0 - 1e-8).sum()),
    })
    if not _internals:
        return lay, meta
    internals = {"T": T, "kept": kept, "notkept": notkept, "M": M, "Wj": Wj, "Hz": Hz, "R": R,
                 "R_transmla": R_tm, "svals": svals, "alpha": alpha, "transmla_layer": lay,
                 "transmla_internals": intl}
    return lay, meta, internals

def _build(model, *, w, d_c, G, fisher, sigma, s2norm, power, alpha_rule, damp_mode, canon, arm,
           solver=SOLVER):
    if fisher is not None:
        raise ValueError("CARE uses no Fisher (its content objective is F = I with SIGMA measured, "
                         "applied to the up-projection under H_z = M SIGMA M^T).  _build_mla must "
                         "drop the global metric for the baselines (build plan §5, adversarial C6) "
                         "-- it reached the initialiser")
    if s2norm is not None:
        raise ValueError("s2norm is MHA2MLA's input; CARE must not receive it")
    if sigma is None:
        raise ValueError("CARE is calibrated on the teacher's SIGMA (D-M5, D-C3): its H_z is "
                         "M SIGMA M^T, so sigma is required")
    L, H, d_h, dm, r, layers = tm._geometry(model)
    w, d_c, B = tm._check_widths(w, d_c, G, H, d_h, r)
    if len(sigma) != L:
        raise ValueError(f"sigma has {len(sigma)} layers, the model {L}")
    n = dm + 1
    out_layers, metas = [], []
    for l in range(L):
        S = tm._check_sigma(sigma[l], n, l)
        Qt, Kt, Vt, _ = gc_read_layer(layers[l], H, d_h, d_h, dm)
        lay, met = build_layer(Qt, Kt, Vt, S, w=w, d_c=d_c, r=r, power=power, alpha_rule=alpha_rule,
                               damp_mode=damp_mode, canon=canon, solver=solver)
        out_layers.append(lay)
        metas.append(met)
    eps_mult = [m["care_solve"]["sigma_cut_eps"] for m in metas]
    undet = [l for l, m in enumerate(metas) if not m["care_solve"]["basis_determined"]]
    gaps = [m["care_solve"]["cut_gap_rel_to_cut"] for m in metas]
    cond = {"power": float(power), "solver": solver, "sigma_cut_eps_min": float(min(eps_mult)),
            "sigma_cut_eps_median": float(np.median(eps_mult)),
            "cut_gap_rel_min": float(np.nanmin(gaps)) if np.isfinite(gaps).any() else float("nan"),
            "n_layers_not_determined": len(undet), "layers_not_determined": undet,
            "determined_eps_mult": DETERMINED_EPS_MULT, "stable_rel_gap_min": STABLE_REL_GAP_MIN}
    print(f"[{arm}] solver={solver} p={power}: the cut sits {min(eps_mult):.2g}..{max(eps_mult):.2g} "
          f"float64 eps below sigma_0 (median {np.median(eps_mult):.2g}); "
          + (f"basis determined on {L - len(undet)}/{L} layers"
             + (f" -- NOT on layers {undet}" if undet else "")
             + (" (high-relative-accuracy Jacobi resolves the ranking at any depth)"
                if solver == "stable" else
                " -- ⛔ the RELEASE path: below the threshold the kept directions are rounding noise")),
          flush=True)
    text = (f"CARE `{DECOMP_METHOD}` content solve on TransMLA's rotary path, reimplemented from {PIN} "
            f"(lora_qkv.py:156-197, 396-406, 419, 470-506; partial_rope.py:74-127) on the teacher's "
            f"SIGMA (D-M5): H_z = M SIGMA M^T with M = [W~_N/alpha ; V_f] augmented, so the bias is in "
            f"the metric; R = U[:, :d_c] of H_z^p W_join^T, p = {power} "
            f"({'NEW_SQRT_BASIS_POWER shipped default' if float(power) == POWER else 'SENSITIVITY'}), "
            f"solver '{solver}' ("
            + ("eigenbasis + dgejsv JOBA='C', high relative accuracy; the release's explicit-H^p SVD "
               "is ill-posed in float64 once d_c nears rank(H_z)" if solver == "stable"
               else "the release's explicit H^p then np.linalg.svd, lora_qkv.py:191-194") + "), "
            f""
            f"damping {percdamp_text(damp_mode)}, eigenvalue clamp {EIGH_EPS:g} asserted not to fire; "
            f"orthonormal projection (TransMLA's gauge), canon_sign {'on' if canon else 'OFF'}; "
            f"W_DKV = R^T M, [W_UK ; W_UV] = W_join R. alpha rule '{alpha_rule}' "
            f"({'V-slice quirk of lora_qkv.py:398 kept' if alpha_rule == 'code' else 'quirk-free: all V rows'}); "
            f"w from DSTAR['mla'] and w + d_c = B, NOT CARE's qk_mqa_dim / kv_lora_rank (§0.1c); "
            f"CARE-U (converter.py:203-208 refuses to save a dynamic-rank model); rotary PCA over the 8 "
            f"TRUE planes only; s = sqrt((d_h+w)/d_h) in the query weights and biases, "
            f"mla_scale = (d_h+w)^-1/2 (D1)")
    return {
        "schema": SCHEMA, "arm": arm, "version": VERSION, "rotary_mode": ROTARY_MODE,
        "G": None if G is None else int(G), "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_h), "dm": int(dm), "r": int(r),
        "provenance": {"fisher_sha256": None, "sigma_sha256": sigma_digest(sigma),
                       "s2norm_sha256": None, "text": text},
        "bias_set": list(BIAS_SET),
        "mla_scale": mla_scale_of(w, d_h),
        "layers": out_layers,
        "_meta": {"power": float(power), "solver": solver, "alpha_rule": alpha_rule, "damp_mode": damp_mode,
                  "canon": bool(canon), "percdamp": float(PERCDAMP), "eigh_eps": float(EIGH_EPS),
                  "decomp_method": DECOMP_METHOD, "pin": PIN, "conditioning": cond,
                  "per_layer": metas},
    }

def percdamp_text(damp_mode):
    return ("0.01*mean(diag H_z) in float64 (_damped_psd_eigh, lora_qkv.py:164)" if damp_mode == "f64"
            else "0.01*mean(diag H_z)*10 in float32 -- the `care` branch's form, lora_qkv.py:107-109 "
                 "(SENSITIVITY ONLY)")

def gc_read_layer(layer, H, d_k, d_v, dm):
    import gam_core as gc
    return gc.read_layer(layer, H, d_k, d_v, dm)

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None, _canon=True):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, power=POWER,
                  alpha_rule="code", damp_mode="f64", canon=_canon, arm=ARM)

def build_weights_p(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None, power=POWER_SQRT):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm,
                  power=power, alpha_rule="code", damp_mode="f64", canon=True, arm=ARM_SQRT)

def build_weights_noquirk(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, power=POWER,
                  alpha_rule="noquirk", damp_mode="f64", canon=True, arm=ARM_NOQUIRK)

def build_weights_f32damp10(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, power=POWER,
                  alpha_rule="code", damp_mode="f32x10", canon=True, arm=ARM_F32DAMP10)

def build_weights_release(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, power=POWER,
                  alpha_rule="code", damp_mode="f64", canon=True, arm=ARM_RELEASE, solver="release")

def care_digest():
    body = f"{DECOMP_METHOD}|p={POWER!r}|damp={PERCDAMP!r}|eps={EIGH_EPS!r}|{PIN}|{VERSION}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
