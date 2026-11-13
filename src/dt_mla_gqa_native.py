import hashlib

import numpy as np

import dt_mla_care as CA
import dt_mla_care_gqa as CG
import dt_mla_mha2mla as MM
import dt_mla_transmla as tm
import gam_core as gc
import kvlab

SCHEMA = tm.SCHEMA
CONSTRUCTION = "gqa-native-2026-11-12.1"
PIN_TM = tm.PIN
PIN_CARE = CA.PIN
PIN_MM = MM.PIN
ALPHA_MODES = ("native", "grid")
SCALE_MODES = ("native", "grid")

def _groups(H, Hkv):
    if Hkv is None or isinstance(Hkv, bool) or int(Hkv) <= 0:
        raise ValueError(f"Hkv = {Hkv!r} must be a positive int")
    Hkv = int(Hkv)
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of Hkv = {Hkv}")
    n_rep = H // Hkv
    if n_rep < 2:
        raise ValueError(f"Hkv = {Hkv} = H: an MHA source takes the grid's own modules, not this one")
    return Hkv, n_rep, np.arange(H) // n_rep

def check_copies(Kt, Vt, Hkv):
    H = Kt.shape[0]
    Hkv, n_rep, _ = _groups(H, Hkv)
    for nm, X in (("key", Kt), ("value", Vt)):
        for g in range(Hkv):
            ref = X[g * n_rep]
            for i in range(g * n_rep + 1, (g + 1) * n_rep):
                if not np.array_equal(X[i], ref):
                    raise ValueError(f"{nm} maps of query heads {g * n_rep} and {i} differ: KV head {g} is not an "
                                     f"exact copy (was the model expanded by kvlab.load_model?)")
    return n_rep

def native_alpha(kn, vn, notkept, w, H, Hkv, d_h):
    Hkv, n_rep, _ = _groups(H, Hkv)
    real = (np.asarray(notkept) // d_h) < Hkv
    if int(real.sum()) != Hkv * d_h - w:
        raise AssertionError(f"{int(real.sum())} real not-kept key rows, expected Hkv*d_h - w = {Hkv * d_h - w}")
    vrows = np.concatenate([np.arange(g * n_rep * d_h, (g * n_rep + 1) * d_h) for g in range(Hkv)])
    return float((kn[real] / np.sqrt(n_rep)).mean() / vn[vrows][w:].mean())

def _transmla_base(Qt, Kt, Vt, S, *, Hkv, w, d_c, r, alpha_mode):
    if alpha_mode not in ALPHA_MODES:
        raise ValueError(f"alpha_mode {alpha_mode!r} not in {ALPHA_MODES}")
    Qt, Kt, Vt, S = (np.asarray(a, dtype=np.float64) for a in (Qt, Kt, Vt, S))
    H, d_h, n = Qt.shape
    Hkv, n_rep, _ = _groups(H, Hkv)
    check_copies(Kt, Vt, Hkv)
    if not 1 <= d_c <= 2 * Hkv * d_h - w:
        raise ValueError(f"d_c = {d_c} outside [1, 2*Hkv*d_h - w = {2 * Hkv * d_h - w}] (lora_qkv.py:50)")
    lay, met, intl = tm.build_layer(Qt, Kt, Vt, S, w=w, d_c=d_c, alpha_rule="code", r=r, _internals=True)
    T, Kr, kept, notkept = intl["T"], intl["Kr"], intl["kept"], intl["notkept"]
    _, gqa = CG.native_latent(intl["M"], notkept, kept, met["lam_rot"], met["U"], H, Hkv, d_h, r)
    D = H * d_h
    Kf = Kt.reshape(D, n)
    KS = Kf @ S
    KrS = T @ KS
    Vf = Vt.reshape(D, n)
    VS = Vf @ S
    kd = np.einsum("ij,ij->i", Kr[notkept], KrS[notkept])
    vd = np.einsum("ij,ij->i", Vf, VS)
    kn, vn = np.sqrt(np.clip(kd, 0.0, None)), np.sqrt(np.clip(vd, 0.0, None))
    a_grid = float(kn.mean() / vn[w:].mean())
    if a_grid != met["alpha_code"]:
        raise AssertionError(f"the recomputed grid alpha {a_grid!r} != dt_mla_transmla's {met['alpha_code']!r}: "
                             f"step 5a is not reproduced")
    a_nat = native_alpha(kn, vn, notkept, w, H, Hkv, d_h)
    if not (np.isfinite(a_nat) and a_nat > 0.0):
        raise ValueError(f"native alpha {a_nat!r} is not a positive finite ratio")
    alpha = a_nat if alpha_mode == "native" else a_grid
    M = np.concatenate([Kr[notkept] / alpha, Vf], axis=0)
    MS = np.concatenate([KrS[notkept] / alpha, VS], axis=0)
    parts = {"T": T, "kept": kept, "notkept": notkept, "M": M, "MS": MS, "alpha": alpha, "a_grid": a_grid,
             "a_nat": a_nat, "n_rep": n_rep, "gqa": gqa}
    return lay, met, intl, parts

def _to_source_scale(lay, n_rep):
    c = float(np.sqrt(n_rep))
    out = dict(lay)
    out["W_DKV"] = np.ascontiguousarray(lay["W_DKV"] / c)
    out["W_KR"] = np.ascontiguousarray(lay["W_KR"] / c)
    for k in ("W_UK", "W_UV", "W_QR", "Psi"):
        out[k] = np.ascontiguousarray(lay[k] * c)
    return out

def transmla_layer(Qt, Kt, Vt, S, *, Hkv, w, d_c, r, alpha_mode="native", scale_mode="native"):
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode {scale_mode!r} not in {SCALE_MODES}")
    lay, met, intl, P = _transmla_base(Qt, Kt, Vt, S, Hkv=Hkv, w=w, d_c=d_c, r=r, alpha_mode=alpha_mode)
    H, d_h, _ = np.shape(Qt)
    D, N = H * d_h, int(P["notkept"].size)
    M, alpha = P["M"], P["alpha"]
    Ckv = P["MS"] @ M.T
    lam_kv, R, damp_kv = tm.eigh_desc(Ckv)
    R_r = R[:, :d_c]
    W_DKV = R_r.T @ M
    A = P["T"].T
    W_UK = alpha * (A[:, P["notkept"]] @ R_r[:N])
    W_UV = R_r[N:]
    zc = np.zeros((D, 1))
    out = dict(lay)
    out["W_DKV"] = np.ascontiguousarray(W_DKV, dtype=np.float64)
    out["W_UK"] = np.ascontiguousarray(np.hstack([W_UK, zc]), dtype=np.float64)
    out["W_UV"] = np.ascontiguousarray(np.hstack([W_UV, zc]), dtype=np.float64)
    if scale_mode == "native":
        out = _to_source_scale(out, P["n_rep"])
    kv_tot = float(np.clip(lam_kv, 0.0, None).sum())
    meta = {k: met[k] for k in ("m_p", "cut", "alpha_code", "alpha_noquirk", "s")}
    meta.update({"alpha_grid": P["a_grid"], "alpha_native": P["a_nat"], "alpha_used": alpha,
                 "alpha_mode": alpha_mode, "scale_mode": scale_mode, "n_rep": P["n_rep"], "gqa": P["gqa"],
                 "lam_kv_cut": (float(lam_kv[d_c - 1]), float(lam_kv[d_c]) if d_c < lam_kv.size else float("nan")),
                 "lam_kv_dropped_share": (float(np.clip(lam_kv[d_c:], 0.0, None).sum()) / kv_tot
                                          if kv_tot > 0 else float("nan")),
                 "damp_kv": damp_kv})
    return out, meta

def care_layer(Qt, Kt, Vt, S, *, Hkv, w, d_c, r, alpha_mode="native", scale_mode="native", power=CA.POWER,
               percdamp=CA.PERCDAMP, eps=CA.EIGH_EPS, damp_mode="f64", canon=True, solver=CA.SOLVER):
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode {scale_mode!r} not in {SCALE_MODES}")
    if not (np.isfinite(float(power)) and float(power) >= 0.0):
        raise ValueError(f"power {power!r} must be a finite non-negative float")
    lay, met, intl, P = _transmla_base(Qt, Kt, Vt, S, Hkv=Hkv, w=w, d_c=d_c, r=r, alpha_mode=alpha_mode)
    S = np.asarray(S, dtype=np.float64)
    H, d_h, _ = np.shape(Qt)
    T, kept, notkept, M, alpha = P["T"], P["kept"], P["notkept"], P["M"], P["alpha"]
    Q, gqa = CG.native_latent(M, notkept, kept, met["lam_rot"], met["U"], H, int(Hkv), d_h, r)
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
    D = H * d_h
    zc = np.zeros((D, 1))
    out = dict(lay)
    out["W_DKV"] = np.ascontiguousarray(W_DKV, dtype=np.float64)
    out["W_UK"] = np.ascontiguousarray(np.hstack([W_UK, zc]), dtype=np.float64)
    out["W_UV"] = np.ascontiguousarray(np.hstack([W_UV, zc]), dtype=np.float64)
    if scale_mode == "native":
        out = _to_source_scale(out, P["n_rep"])
    meta = {k: met[k] for k in ("m_p", "cut", "alpha_code", "s")}
    meta.update({"alpha_grid": P["a_grid"], "alpha_native": P["a_nat"], "alpha_used": alpha,
                 "alpha_mode": alpha_mode, "scale_mode": scale_mode, "n_rep": P["n_rep"],
                 "care_power": float(power), "care_percdamp": float(percdamp), "care_damp_mode": damp_mode,
                 "care_canon": bool(canon), "care_decomp_method": CA.DECOMP_METHOD, "care_solver": solver,
                 "care_solve": solve, "care_gqa": gqa})
    return out, meta

def mha2mla_layer(Qt, Kt, Vt, score, *, Hkv, w, d_c, r):
    Qt, Kt, Vt = (np.asarray(a, dtype=np.float64) for a in (Qt, Kt, Vt))
    H, d_k, n1 = Qt.shape
    dm, h, m = n1 - 1, r // 2, w // 2
    if Kt.shape != Qt.shape or Vt.shape != Qt.shape:
        raise ValueError(f"shapes Qt {Qt.shape} Kt {Kt.shape} Vt {Vt.shape}")
    if r != d_k:
        raise ValueError(f"r = {r} != d_h = {d_k}: partial RoPE on a GQA source is not handled (every grid GQA model "
                         f"is full RoPE)")
    Hkv, n_rep, grp = _groups(H, Hkv)
    check_copies(Kt, Vt, Hkv)
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (H, h):
        raise ValueError(f"score {score.shape} != (H, r/2) = {(H, h)}")
    gscore = np.ascontiguousarray(score.reshape(Hkv, n_rep, h).sum(axis=1))
    gg, pp, sel = MM.select_pairs(gscore, w)
    Kn = np.ascontiguousarray(Kt[::n_rep])
    Vn = np.ascontiguousarray(Vt[::n_rep])
    WK = Kn.reshape(Hkv * d_k, n1)
    WV = Vn.reshape(Hkv * d_k, n1)
    rot_a = d_k * gg + pp
    rot_b = rot_a + h
    rows_rot = np.concatenate([rot_a, rot_b])
    notS = np.setdiff1d(np.arange(Hkv * d_k, dtype=np.int64), rows_rot)
    Dn = Hkv * d_k - w
    if notS.shape != (Dn,):
        raise AssertionError(f"complement has {notS.size} rows, expected {Dn}")
    stack = np.ascontiguousarray(np.concatenate([WK[notS, :dm], WV[:, :dm]], axis=0))
    if not 1 <= d_c <= min(stack.shape):
        raise ValueError(f"d_c = {d_c} must be in [1, {min(stack.shape)}] (the stack is {stack.shape})")
    U, Sv, Vh = np.linalg.svd(stack, full_matrices=False)
    up = U[:, :d_c] * Sv[:d_c]
    W_DKV = np.concatenate([Vh[:d_c], np.zeros((d_c, 1))], axis=1)
    upk = np.concatenate([up[:Dn], WK[notS, dm:]], axis=1)
    upv = np.concatenate([up[Dn:], WV[:, dm:]], axis=1)

    D = H * d_k
    keptdim = np.zeros((Hkv, d_k), dtype=bool)
    keptdim[gg, pp] = True
    keptdim[gg, pp + h] = True
    pos = -np.ones(Hkv * d_k, dtype=np.int64)
    pos[notS] = np.arange(Dn)
    W_UQ = np.zeros((D, n1))
    W_UK = np.zeros((D, d_c + 1))
    W_UV = np.zeros((D, d_c + 1))
    for i in range(H):
        g = grp[i]
        for j in range(d_k):
            row = i * d_k + j
            W_UV[row] = upv[g * d_k + j]
            if keptdim[g, j]:
                continue
            W_UQ[row] = Qt[i, j]
            W_UK[row] = upk[pos[g * d_k + j]]
    Psi = np.zeros((H, r, w))
    for s in range(m):
        hs = np.where(grp == gg[s])[0]
        Psi[hs, pp[s], s] = 1.0
        Psi[hs, pp[s] + h, s + m] = 1.0
    W_QR = np.concatenate([Psi[i].T @ Qt[i][:r] for i in range(H)], axis=0)
    lay = {
        "slot_plane": np.concatenate([pp, pp]).astype(np.int64),
        "slot_head": np.full(w, -1, dtype=np.int64),
        "dC": np.full(H, d_k, dtype=np.int64),
        "W_DKV": W_DKV, "W_KR": WK[rows_rot], "W_UQ": W_UQ, "W_QR": W_QR,
        "W_UK": W_UK, "W_UV": W_UV, "Psi": Psi,
    }
    lay = {k: np.ascontiguousarray(v) for k, v in lay.items()}
    tail = float((Sv[d_c:] ** 2).sum())
    diag = {
        "kept": [[int(a), int(b)] for a, b in zip(gg, pp)],
        "k_g": sel["k_i"].tolist(),
        "rel_gap": sel["rel_gap"], "last_kept": sel["last_kept"], "first_dropped": sel["first_dropped"],
        "sv_cut": float(Sv[d_c - 1]), "sv_next": float(Sv[d_c]) if d_c < Sv.size else None,
        "energy_kept_frac": float(1.0 - tail / float((Sv ** 2).sum())),
        "n_rep": n_rep, "zero_content_rows": int(keptdim.sum()) * n_rep,
    }
    return lay, diag

def _model_hkv(model, arm):
    Hkv = getattr(model, "_gqa_kv_heads", None)
    H = int(model.config.num_attention_heads)
    if Hkv is None or int(Hkv) == H:
        raise ValueError(f"{arm} is a baseline on a GQA source: the model must be the kvlab-expanded copy of one "
                         f"(model._gqa_kv_heads = {Hkv!r}, H = {H}). An MHA model takes the grid's own modules")
    return int(Hkv)

def _sigma_build(model, *, w, d_c, G, fisher, sigma, s2norm, arm, which):
    name = {"transmla": "TransMLA", "care": "CARE"}[which]
    if fisher is not None:
        raise ValueError(f"{name} uses no Fisher; _build_mla must drop the global metric for the baselines -- it "
                         f"reached the initialiser")
    if s2norm is not None:
        raise ValueError(f"s2norm is MHA2MLA's input; {name} must not receive it")
    if sigma is None:
        raise ValueError(f"{name} is calibrated on the teacher's SIGMA (D-M5): sigma is required")
    Hkv = _model_hkv(model, arm)
    L, H, d_h, dm, r, layers = tm._geometry(model)
    w, d_c, B = tm._check_widths(w, d_c, G, H, d_h, r)
    if len(sigma) != L:
        raise ValueError(f"sigma has {len(sigma)} layers, the model {L}")
    n = dm + 1
    out_layers, metas = [], []
    for l in range(L):
        S = tm._check_sigma(sigma[l], n, l)
        Qt, Kt, Vt, _ = gc.read_layer(layers[l], H, d_h, d_h, dm)
        if which == "transmla":
            lay, met = transmla_layer(Qt, Kt, Vt, S, Hkv=Hkv, w=w, d_c=d_c, r=r)
        else:
            lay, met = care_layer(Qt, Kt, Vt, S, Hkv=Hkv, w=w, d_c=d_c, r=r)
        out_layers.append(lay)
        metas.append(met)
    a_nat = [m_["alpha_native"] for m_ in metas]
    a_grid = [m_["alpha_grid"] for m_ in metas]
    _rr = [x / y for x, y in zip(a_nat, a_grid)]
    record = (f" RECORD alpha_native_over_grid min {min(_rr):.6f} max {max(_rr):.6f} over {L} layers; "
              f"source_scale_c {float(np.sqrt(H // Hkv)):.6f} (n_rep {H // Hkv}).")
    print(f"[{arm}] Hkv={Hkv} of H={H} (n_rep {H // Hkv}): {name} as its code builds it on the source; alpha "
          f"native / grid over {L} layers: {min(x / y for x, y in zip(a_nat, a_grid)):.3f}-"
          f"{max(x / y for x, y in zip(a_nat, a_grid)):.3f} (layer 0: {a_nat[0]:.6g} vs {a_grid[0]:.6g}); latent and "
          f"rotary key at the source's scale (/ sqrt({H // Hkv}))", flush=True)
    if which == "transmla":
        version, pin = tm.VERSION, PIN_TM
        text = (f"[{CONSTRUCTION}] TransMLA on a GQA source ({Hkv} KV heads of {H}) as its code builds it ({pin}: "
                f"partial_rope.py:55-56, 74-106; lora_qkv.py:110-117, 191-214): dt_mla_transmla's RoRoPE, allocation "
                f"and rotation on the kvlab-expanded copy, alpha over the source's own not-kept key and value columns "
                f"(the V-slice quirk kept), the joint PCA, then the latent and rotary key at the source's scale (W_DKV, "
                f"W_KR / sqrt(n_rep); W_UK, W_UV, W_QR x sqrt(n_rep)); teacher's SIGMA (D-M5); global top-w/2 "
                f"allocation (Eq. 19), FreqFold off, interleave skipped; s = sqrt((d_h+w)/d_h), mla_scale = "
                f"(d_h+w)^-1/2 (D1)")
        meta = {"alpha_rule": "code_native", "pin": pin, "construction": CONSTRUCTION, "kv_heads": Hkv,
                "per_layer": metas}
    else:
        version, pin = CA.VERSION, PIN_CARE
        text = (f"[{CONSTRUCTION}] CARE `{CA.DECOMP_METHOD}` on a GQA source ({Hkv} KV heads of {H}) as its code "
                f"builds it ({pin}): dt_mla_care_gqa's solve in the source's own latent of 2*Hkv*d_h - w = "
                f"{2 * Hkv * d_h - w} coordinates, p = {CA.POWER}, damping 0.01*mean(diag H_u), stable solver, "
                f"canon_sign, on TransMLA's rotary path and TransMLA's code alpha on the source ({PIN_TM}: "
                f"lora_qkv.py:110-117), then the latent and rotary key at the source's scale (as TransMLA's); the "
                f"teacher's SIGMA (D-M5); w + d_c = B; mla_scale = (d_h+w)^-1/2 (D1)")
        undet = [l for l, m_ in enumerate(metas) if not m_["care_solve"]["basis_determined"]]
        eps_mult = [m_["care_solve"]["sigma_cut_eps"] for m_ in metas]
        gaps = [m_["care_solve"]["cut_gap_rel_to_cut"] for m_ in metas]
        cond = {"power": float(CA.POWER), "solver": CA.SOLVER, "sigma_cut_eps_min": float(min(eps_mult)),
                "sigma_cut_eps_median": float(np.median(eps_mult)),
                "cut_gap_rel_min": float(np.nanmin(gaps)) if np.isfinite(gaps).any() else float("nan"),
                "n_layers_not_determined": len(undet), "layers_not_determined": undet,
                "determined_eps_mult": CA.DETERMINED_EPS_MULT, "stable_rel_gap_min": CA.STABLE_REL_GAP_MIN}
        print(f"[{arm}] solver={CA.SOLVER} p={CA.POWER}: basis determined on {L - len(undet)}/{L} layers"
              + (f" -- NOT on layers {undet}" if undet else ""), flush=True)
        meta = {"power": float(CA.POWER), "solver": CA.SOLVER, "alpha_rule": "code_native", "damp_mode": "f64",
                "canon": True, "percdamp": float(CA.PERCDAMP), "eigh_eps": float(CA.EIGH_EPS),
                "decomp_method": CA.DECOMP_METHOD, "pin": pin, "construction": CONSTRUCTION, "kv_heads": Hkv,
                "conditioning": cond, "per_layer": metas}
    return {
        "schema": SCHEMA, "arm": arm, "version": version, "rotary_mode": "shared",
        "G": None if G is None else int(G), "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_h), "dm": int(dm), "r": int(r),
        "provenance": {"fisher_sha256": None, "sigma_sha256": tm.sigma_digest(sigma), "s2norm_sha256": None,
                       "text": text + record},
        "bias_set": list(tm.BIAS_SET),
        "mla_scale": tm.mla_scale_of(w, d_h),
        "layers": out_layers,
        "_meta": meta,
    }

def build_transmla(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None, arm):
    return _sigma_build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, arm=arm,
                        which="transmla")

def build_care(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None, arm):
    return _sigma_build(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, arm=arm,
                        which="care")

def build_mha2mla(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None, arm):
    if fisher is not None or sigma is not None:
        raise ValueError("the MHA2MLA initialiser uses neither fisher nor sigma; the dispatch must not pass them")
    if s2norm is None:
        raise ValueError("the MHA2MLA initialiser needs s2norm (fit_s2norm on the fit split)")
    Hkv = _model_hkv(model, arm)
    L, H, d_k, dm, r = MM._geom(model)
    h = r // 2
    w, d_c, G = MM._as_int("w", w), MM._as_int("d_c", d_c), MM._as_int("G", G, allow_none=True)
    B = w + d_c
    if w % 2 or w < 2 or w > Hkv * r:
        raise ValueError(f"w = {w} must be even and in [2, Hkv * r = {Hkv * r}]")
    if d_c < 1:
        raise ValueError(f"d_c = {d_c}")
    if G is not None and B != 2 * d_k * G:
        raise ValueError(f"w + d_c = {B} != 2 d_h G = {2 * d_k * G}")
    qn, kn, n_tok = MM._check_s2norm(s2norm, L, H, h)
    digest = MM.s2norm_digest({"q": qn, "k": kn, "n_tok": n_tok})
    before = MM._attn_digest(model)
    layers_out, diags = [], []
    for l, layer in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, _ = gc.read_layer(layer, H, d_k, d_k, dm)
        lay, dg = mha2mla_layer(Qt, Kt, Vt, qn[l] * kn[l], Hkv=Hkv, w=w, d_c=d_c, r=r)
        if dg["rel_gap"] < MM.REL_GAP_WARN:
            print(f"[{arm}] WARNING layer {l}: selection cut is a near-tie, rel_gap = {dg['rel_gap']:.2e}", flush=True)
        layers_out.append(lay)
        diags.append(dg)
    if MM._attn_digest(model) != before:
        raise RuntimeError("build_mha2mla modified the model's attention parameters")
    kept_planes = np.concatenate([lay["slot_plane"][: w // 2] for lay in layers_out])
    plane_hist = np.bincount(kept_planes, minlength=h).tolist()
    gaps = np.array([dg["rel_gap"] for dg in diags])
    meta = {"pin": PIN_MM, "construction": CONSTRUCTION, "kv_heads": Hkv, "plane_hist": plane_hist,
            "min_rel_gap": float(gaps.min()), "min_rel_gap_layer": int(gaps.argmin()), "layers": diags}
    print(f"[{arm}] Hkv={Hkv} of H={H}: (KV head, plane) pairs cached once, w={w} d_c={d_c} G={G}; kept pairs per "
          f"plane {plane_hist}; min rel_gap {meta['min_rel_gap']:.2e} (layer {meta['min_rel_gap_layer']})", flush=True)
    return {
        "schema": SCHEMA, "arm": arm, "version": MM.VERSION, "rotary_mode": "shared",
        "G": G, "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_k), "dm": int(dm), "r": int(r),
        "provenance": {
            "fisher_sha256": None, "sigma_sha256": None, "s2norm_sha256": digest,
            "text": (f"[{CONSTRUCTION}] MHA2MLA ({PIN_MM}) on a GQA source ({Hkv} KV heads of {H}) as its code builds "
                     f"it: S2 score summed over each KV group (2_norm.py:206-209), global top w/2 (KV head, plane) "
                     f"pairs (n_tok={n_tok}), kept key rows of the source copied with bias and cached once, joint SVD "
                     f"in f64 of the source's bias-free [W_K[notS]; W_V] (patching_model_load.py:93-99), W_DKV = "
                     f"Vh[:d_c] (no bias), up = U S, biases b_K[notS], b_V; in TransMLA's shared layout (query head "
                     f"i reads its group's slots; the kept dims' content rows 0 on both sides); scale d_h^-1/2."),
        },
        "bias_set": list(MM.BIAS_SET),
        "mla_scale": MM.mla_scale_of(w, d_k),
        "layers": layers_out,
        "_meta": meta,
    }

def gqa_native_digest():
    body = f"{CONSTRUCTION}|{PIN_TM}|{PIN_CARE}|{PIN_MM}|{tm.VERSION}|{CA.VERSION}|{MM.VERSION}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
