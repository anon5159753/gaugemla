import hashlib

import numpy as np

import dt_mla_care as CA
import dt_mla_transmla as tm
from dt_mla_transmla import canon_sign, mla_scale_of, s_factor, sigma_digest

SCHEMA = CA.SCHEMA
ARM = "mla_care_gqa_sigma"
VERSION = CA.VERSION
CONSTRUCTION = "care-gqa-2026-11-12.1"
ROTARY_MODE = CA.ROTARY_MODE
BIAS_SET = list(CA.BIAS_SET)
PIN = CA.PIN
POWER = CA.POWER
PERCDAMP = CA.PERCDAMP
DECOMP_METHOD = CA.DECOMP_METHOD

NULL_EIG_TOL = 1e-10
SYM_TOL = 1e-9
NULL_ROW_TOL = 1e-10
LIFT_TOL = 1e-12

def native_latent(M, notkept, kept, lam_rot, U, H, Hkv, d_h, r):
    M = np.asarray(M, dtype=np.float64)
    notkept = np.asarray(notkept, dtype=np.int64)
    kept = np.asarray(kept, dtype=np.int64)
    if r != d_h:
        raise ValueError(f"r = {r} != d_h = {d_h}: partial RoPE on a GQA source is not handled (every grid GQA "
                         f"model is full RoPE); a non-rotary key row is not a RoRoPE component")
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of Hkv = {Hkv}")
    n_rep = H // Hkv
    if n_rep < 2:
        raise ValueError(f"Hkv = {Hkv} = H: an MHA source takes dt_mla_care, not this module")
    D, N = H * d_h, int(notkept.size)
    if M.shape[0] != N + D:
        raise ValueError(f"M has {M.shape[0]} rows, expected N + D = {N + D}")
    grp = np.arange(H) // n_rep
    lam = np.asarray(lam_rot, dtype=np.float64)
    top = float(lam.max())
    if not top > 0.0:
        raise ValueError("the RoRoPE eigenvalues are all zero")
    lead_min = float(lam[:, Hkv - 1].min()) / top
    tail_max = float(np.abs(lam[:, Hkv:]).max()) / top
    if not lead_min > NULL_EIG_TOL or tail_max > NULL_EIG_TOL:
        raise ValueError(f"per plane the top Hkv = {Hkv} RoRoPE eigenvalues must be nonzero and the rest null: "
                         f"smallest leading {lead_min:.3g}, largest trailing {tail_max:.3g} of the top "
                         f"(tolerance {NULL_EIG_TOL:g}). A source whose KV heads are not distinct per plane has "
                         f"no unambiguous native latent here")
    U = np.asarray(U, dtype=np.float64)
    sym = 0.0
    for p in range(U.shape[0]):
        for g in range(Hkv):
            blk = U[p][grp == g, :Hkv]
            sym = max(sym, float(np.abs(blk - blk.mean(axis=0, keepdims=True)).max()))
    if sym > SYM_TOL:
        raise ValueError(f"a leading RoRoPE eigenvector differs across the copies of a KV head by {sym:.3g} "
                         f"(tolerance {SYM_TOL:g}): it is not the lift of the source's own")
    if (kept // d_h).max() >= Hkv:
        raise ValueError("TransMLA kept a null RoRoPE component (c >= Hkv) in the rotary key")
    comp = notkept // d_h
    sym_rows = np.where(comp < Hkv)[0]
    null_rows = np.where(comp >= Hkv)[0]
    if sym_rows.size != Hkv * d_h - kept.size:
        raise AssertionError(f"{sym_rows.size} leading not-kept key rows, expected Hkv*d_h - w = "
                             f"{Hkv * d_h - kept.size}")
    kn = np.linalg.norm(M[:N], axis=1)
    null_rel = float(kn[null_rows].max() / kn.max()) if null_rows.size else 0.0
    if null_rel > NULL_ROW_TOL:
        raise ValueError(f"a key row of a null RoRoPE component has norm {null_rel:.3g} of the largest "
                         f"(tolerance {NULL_ROW_TOL:g})")
    V = M[N:]
    for g in range(Hkv):
        hs = np.where(grp == g)[0]
        ref = V[hs[0] * d_h:(hs[0] + 1) * d_h]
        for i in hs[1:]:
            if not np.array_equal(V[i * d_h:(i + 1) * d_h], ref):
                raise ValueError(f"value rows of query heads {hs[0]} and {i} differ: KV head {g} is not an exact "
                                 f"copy (was the model expanded by kvlab.load_model?)")
    nu = sym_rows.size + Hkv * d_h
    Q = np.zeros((nu, N + D), dtype=np.float64)
    Q[np.arange(sym_rows.size), sym_rows] = 1.0
    c = 1.0 / np.sqrt(n_rep)
    for g in range(Hkv):
        for i in np.where(grp == g)[0]:
            Q[sym_rows.size + g * d_h + np.arange(d_h), N + i * d_h + np.arange(d_h)] = c
    lift = float(np.abs(Q.T @ (Q @ M) - M).max() / np.abs(M).max())
    if lift > LIFT_TOL:
        raise ValueError(f"the latent leaves the native coordinates: |M - Q^T Q M| / |M| = {lift:.3g} "
                         f"(tolerance {LIFT_TOL:g})")
    info = {"Hkv": int(Hkv), "n_rep": int(n_rep), "native_latent": int(nu),
            "lead_eig_min_rel": lead_min, "tail_eig_max_rel": tail_max, "eigvec_group_asym": sym,
            "null_key_row_max_rel": null_rel, "lift_resid": lift}
    return Q, info

def build_layer(Qt, Kt, Vt, S, *, Hkv, w, d_c, r, power=POWER, alpha_rule="code", percdamp=PERCDAMP,
                eps=CA.EIGH_EPS, damp_mode="f64", canon=True, solver=CA.SOLVER, _internals=False):
    Qt, Kt, Vt, S = (np.asarray(a, dtype=np.float64) for a in (Qt, Kt, Vt, S))
    H, d_h, n = Qt.shape
    dm = n - 1
    if not (np.isfinite(float(power)) and float(power) >= 0.0):
        raise ValueError(f"power {power!r} must be a finite non-negative float")
    if not 1 <= d_c <= 2 * Hkv * d_h - w:
        raise ValueError(f"d_c = {d_c} outside [1, 2*Hkv*d_h - w = {2 * Hkv * d_h - w}] (lora_qkv.py:344)")
    lay, met, intl = tm.build_layer(Qt, Kt, Vt, S, w=w, d_c=d_c, alpha_rule=alpha_rule, r=r, _internals=True)
    T, kept, notkept = intl["T"], intl["kept"], intl["notkept"]
    M, alpha, R_tm = intl["M"], intl["alpha"], intl["R_r"]
    D, N = H * d_h, int(notkept.size)
    if M.shape != (N + D, n):
        raise AssertionError(f"M is {M.shape}, expected {(N + D, n)}")
    Q, gqa = native_latent(M, notkept, kept, met["lam_rot"], met["U"], H, Hkv, d_h, r)

    A = T.T
    Wj = CA.join_up_projection(A, notkept, alpha, H, d_h)
    Mu = Q @ M
    Hu = (Mu @ S) @ Mu.T
    Wju = Wj @ Q.T
    Ru, svals, solve = CA.svd_project_basis(Wju, Hu, d_c, percdamp=percdamp, power=power, eps=eps,
                                            damp_mode=damp_mode, canon=canon, solver=solver)
    R = Q.T @ Ru
    W_DKV, W_UK, W_UV = CA.content_from_basis(R, M, Wj, H, d_h)

    comp = Wj @ (R @ W_DKV)
    ref_k = np.concatenate([comp[2 * i * d_h:2 * i * d_h + d_h] for i in range(H)], axis=0)
    ref_v = np.concatenate([comp[2 * i * d_h + d_h:2 * (i + 1) * d_h] for i in range(H)], axis=0)
    for nm, got, ref in (("W_UK", W_UK @ W_DKV, ref_k), ("W_UV", W_UV @ W_DKV, ref_v)):
        dev = float(np.abs(got - ref).max())
        nrm = float(np.abs(ref).max())
        if dev > CA.CONTENT_TOL * max(nrm, 1e-300):
            raise AssertionError(f"{nm} . W_DKV != the W_join R R^T M block ({dev:.3g} vs {nrm:.3g}): "
                                 f"the head-major de-interleaving is out of step")

    zc = np.zeros((D, 1))
    lay["W_DKV"] = np.ascontiguousarray(W_DKV, dtype=np.float64)
    lay["W_UK"] = np.ascontiguousarray(np.hstack([W_UK, zc]), dtype=np.float64)
    lay["W_UV"] = np.ascontiguousarray(np.hstack([W_UV, zc]), dtype=np.float64)

    Hz = (M @ S) @ M.T

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
        "care_solve": solve, "care_gqa": gqa,
        "latent_damage_care": dmg_care, "latent_damage_transmla": dmg_tm, "latent_damage_total": tot,
        "latent_damage_care_rel": dmg_care / tot if tot > 0 else float("nan"),
        "latent_damage_transmla_rel": dmg_tm / tot if tot > 0 else float("nan"),
        "subspace_sv_min_vs_transmla": float(ang.min()), "subspace_sv_mean_vs_transmla": float(ang.mean()),
        "n_shared_directions_vs_transmla": int((ang > 1.0 - 1e-8).sum()),
    })
    if not _internals:
        return lay, meta
    internals = {"T": T, "kept": kept, "notkept": notkept, "M": M, "Wj": Wj, "Q": Q, "Hu": Hu, "R": R, "Ru": Ru,
                 "R_transmla": R_tm, "svals": svals, "alpha": alpha, "transmla_layer": intl}
    return lay, meta, internals

def _build(model, *, w, d_c, G, fisher, sigma, s2norm, arm=ARM):
    if fisher is not None:
        raise ValueError("CARE uses no Fisher (its content objective is F = I with SIGMA measured); _build_mla "
                         "must drop the global metric for the baselines -- it reached the initialiser")
    if s2norm is not None:
        raise ValueError("s2norm is MHA2MLA's input; CARE must not receive it")
    if sigma is None:
        raise ValueError("CARE is calibrated on the teacher's SIGMA (D-M5, D-C3): sigma is required")
    Hkv = getattr(model, "_gqa_kv_heads", None)
    L, H, d_h, dm, r, layers = tm._geometry(model)
    if Hkv is None or int(Hkv) == H:
        raise ValueError(f"{arm} is CARE on a GQA source: the model must be the kvlab-expanded copy of one "
                         f"(model._gqa_kv_heads = {Hkv!r}, H = {H}). An MHA model takes dt_mla_care")
    Hkv = int(Hkv)
    w, d_c, B = tm._check_widths(w, d_c, G, H, d_h, r)
    if len(sigma) != L:
        raise ValueError(f"sigma has {len(sigma)} layers, the model {L}")
    n = dm + 1
    out_layers, metas = [], []
    for l in range(L):
        S = tm._check_sigma(sigma[l], n, l)
        Qt, Kt, Vt, _ = CA.gc_read_layer(layers[l], H, d_h, d_h, dm)
        lay, met = build_layer(Qt, Kt, Vt, S, Hkv=Hkv, w=w, d_c=d_c, r=r)
        out_layers.append(lay)
        metas.append(met)
    eps_mult = [m["care_solve"]["sigma_cut_eps"] for m in metas]
    undet = [l for l, m in enumerate(metas) if not m["care_solve"]["basis_determined"]]
    gaps = [m["care_solve"]["cut_gap_rel_to_cut"] for m in metas]
    worst = {k: max(m["care_gqa"][k] for m in metas)
             for k in ("tail_eig_max_rel", "eigvec_group_asym", "null_key_row_max_rel", "lift_resid")}
    cond = {"power": float(POWER), "solver": CA.SOLVER, "sigma_cut_eps_min": float(min(eps_mult)),
            "sigma_cut_eps_median": float(np.median(eps_mult)),
            "cut_gap_rel_min": float(np.nanmin(gaps)) if np.isfinite(gaps).any() else float("nan"),
            "n_layers_not_determined": len(undet), "layers_not_determined": undet,
            "determined_eps_mult": CA.DETERMINED_EPS_MULT, "stable_rel_gap_min": CA.STABLE_REL_GAP_MIN}
    print(f"[{arm}] Hkv={Hkv} of H={H}: native latent {2 * Hkv * d_h - w} (expanded {2 * H * d_h - w}); worst over "
          f"{L} layers: trailing eig {worst['tail_eig_max_rel']:.2g}, eigvec asym {worst['eigvec_group_asym']:.2g}, "
          f"null key row {worst['null_key_row_max_rel']:.2g}, lift {worst['lift_resid']:.2g}; solver={CA.SOLVER} "
          f"p={POWER}: basis determined on {L - len(undet)}/{L} layers"
          + (f" -- NOT on layers {undet}" if undet else ""), flush=True)
    text = (f"[{CONSTRUCTION}] CARE `{DECOMP_METHOD}` content solve on a GQA source ({Hkv} KV heads of {H}), in the source's own "
            f"latent of 2*Hkv*d_h - w = {2 * Hkv * d_h - w} coordinates as CARE's code forms it ({PIN}: "
            f"partial_rope.py:55-56, 74-83; lora_qkv.py:164-166, 406-419, 470-506), on TransMLA's rotary path "
            f"over the kvlab-expanded model and the teacher's SIGMA (D-M5): H_u = (Q M) SIGMA (Q M)^T, "
            f"R_u = U[:, :d_c] of H_u^p (W_join Q^T)^T, p = {POWER}, damping 0.01*mean(diag H_u) (CARE's "
            f"shrinkage), stable solver, canon_sign; R = Q^T R_u, W_DKV = R^T M, [W_UK ; W_UV] = W_join R. alpha "
            f"rule 'code' (the grid's TransMLA alpha, inherited); w + d_c = B (§0.1c); CARE-U; "
            f"mla_scale = (d_h+w)^-1/2 (D1)")
    return {
        "schema": SCHEMA, "arm": arm, "version": VERSION, "rotary_mode": ROTARY_MODE,
        "G": None if G is None else int(G), "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_h), "dm": int(dm), "r": int(r),
        "provenance": {"fisher_sha256": None, "sigma_sha256": sigma_digest(sigma), "s2norm_sha256": None,
                       "text": text},
        "bias_set": list(BIAS_SET),
        "mla_scale": mla_scale_of(w, d_h),
        "layers": out_layers,
        "_meta": {"power": float(POWER), "solver": CA.SOLVER, "alpha_rule": "code", "damp_mode": "f64",
                  "canon": True, "percdamp": float(PERCDAMP), "eigh_eps": float(CA.EIGH_EPS),
                  "decomp_method": DECOMP_METHOD, "pin": PIN, "construction": CONSTRUCTION, "kv_heads": Hkv,
                  "conditioning": cond,
                  "gqa_worst": worst, "per_layer": metas},
    }

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm)

def care_gqa_digest():
    body = (f"{DECOMP_METHOD}|p={POWER!r}|damp={PERCDAMP!r}|eps={CA.EIGH_EPS!r}|{PIN}|{VERSION}|{CONSTRUCTION}|"
            f"tols={NULL_EIG_TOL!r},{SYM_TOL!r},{NULL_ROW_TOL!r},{LIFT_TOL!r}")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
