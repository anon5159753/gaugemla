import numpy as np

import dt_decoupled
import dt_mla_ours as O
import dt_mla_transmla as TM
import dt_mla_unrope_width as UW
import dt_optimal
import gam_core as gc
import kvlab

ARM = "mla_ours_unrope"
SCHEMA = O.SCHEMA
VERSION = "mla_unrope/1 (2026-10-01)"
ROTARY_MODE = O.ROTARY_MODE
BIAS_SET = list(O.BIAS_SET)
WIDTH_RULE = "unrope"
READOUTS = ("verbatim", "damped")
ROTARY_ALLOCS = ("energy", "cost")
ROTARY_SOURCES = ("ours", "transmla")

def unrope_rows(unrope, P):
    u = sorted(int(p) for p in unrope)
    if len(set(u)) != len(u) or any(not 0 <= p < P for p in u):
        raise ValueError(f"un-rope set {unrope!r} must be distinct plane indices in [0, {P})")
    return np.array(u + [p + P for p in u], dtype=np.int64), u

def apply_readout(Kres, rows, mu_p, P):
    out = np.array(Kres, dtype=np.float64, copy=True)
    for p in rows:
        a, b = float(mu_p[p].real), float(mu_p[p].imag)
        x, y = Kres[:, p, :], Kres[:, p + P, :]
        out[:, p, :] = a * x - b * y
        out[:, p + P, :] = b * x + a * y
    return out

def alloc_cost(theta, T, readout, planes, P, rotary_alloc):
    if rotary_alloc not in ROTARY_ALLOCS:
        raise ValueError(f"rotary_alloc {rotary_alloc!r} not in {ROTARY_ALLOCS}")
    c = np.ones(P, dtype=np.float64)
    if rotary_alloc == "energy" or not len(planes):
        return c
    c[list(planes)] = UW.plane_cost(theta, T, readout)[list(planes)]
    if not c.any():
        raise ValueError(
            "every plane un-ropes for free (c = 0 on all of them), so the rotary block would be "
            "allocated by direction_order's plane-major tie-break -- a meaningless order dressed as "
            "a choice. No plane rotates measurably over T; check theta and T.")
    return c

def transmla_rotary(Qt, Kt, Sigma_l, *, w_rot, r):
    Qt = np.asarray(Qt, dtype=np.float64)
    Kt = np.asarray(Kt, dtype=np.float64)
    H, d_h, n = Kt.shape
    S = TM._check_sigma(Sigma_l, n, "(this layer)")
    if w_rot % 2 or not (2 <= w_rot <= H * r):
        raise ValueError(f"w = {w_rot} invalid for TransMLA's rotary key at H = {H}, r = {r}")
    h, m, D = r // 2, w_rot // 2, H * d_h
    xdim, ydim = TM.pairing_dims(r)
    Kf = Kt.reshape(D, n)
    KS = Kf @ S
    lam_rot = np.zeros((h, H))
    U = np.zeros((h, H, H))
    for p in range(h):
        ix = np.arange(H) * d_h + int(xdim[p])
        iy = np.arange(H) * d_h + int(ydim[p])
        Hp = KS[ix] @ Kf[ix].T + KS[iy] @ Kf[iy].T
        Hp = 0.5 * (Hp + Hp.T)
        lam_rot[p], U[p], _ = TM.eigh_desc(Hp)
    m_p, slots, _cut = TM.allocate(lam_rot, m)
    slot_plane = np.array([p for (p, _) in slots] * 2, dtype=np.int64)
    Kr = TM.rotation_matrix(U, H, d_h, r) @ Kf
    kept = np.array([c * d_h + int(xdim[p]) for (p, c) in slots]
                    + [c * d_h + int(ydim[p]) for (p, c) in slots], dtype=np.int64)
    W_KR = Kr[kept]
    Psi = np.zeros((H, r, w_rot))
    for sl, (p, c) in enumerate(slots):
        Psi[:, int(xdim[p]), sl] = U[p][:, c]
        Psi[:, int(ydim[p]), sl + m] = U[p][:, c]
    W_QR = np.concatenate([Psi[i].T @ Qt[i][:r] for i in range(H)], axis=0)
    _kept = set(slots)
    lam_tail = float(sum(lam_rot[p, c] for p in range(h) for c in range(H) if (p, c) not in _kept))
    return W_KR, W_QR, Psi, slot_plane, np.asarray(m_p, dtype=np.int64), lam_tail

def build_layer(Qt, Kt, Vt, Wo, GK_l, GV_l, Sigma_l, *, w_rot, d_c, r, readout, unrope, theta,
                T, rotary_alloc, rotary_source="ours"):
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
    if not 0 < r <= d_k or r % 2:
        raise ValueError(f"r = {r}: an even rotary width in (0, {d_k}] is needed")
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    P = r // 2
    rows, planes = unrope_rows(unrope, P)
    dC0 = (d_k - r) + len(rows)
    if dC0 <= 0:
        raise ValueError(
            f"dC = (d_h - r) + 2|un-roped| = ({d_k} - {r}) + {len(rows)} = {dC0}: the content "
            f"latent would carry NO key rows, which is dt_mla_ours.py:334's refusal restated on "
            f"the effective width. Un-rope at least one plane on a full-RoPE model.")
    S_h, S_ih, n_floored = dt_decoupled._sigma_pair(Sigma_l, n)

    if rotary_source not in ROTARY_SOURCES:
        raise ValueError(f"rotary_source {rotary_source!r} not in {ROTARY_SOURCES}")
    if rotary_source == "ours":
        Vh, mu, vp = O.plane_spectra(Kt, GK_l, S_h, r, phase_fix=True)
        c_alloc = alloc_cost(theta, T, readout, planes, P, rotary_alloc)
        mu_alloc = mu if rotary_alloc == "energy" else [mu[p] * c_alloc[p] for p in range(P)]
        m, alloc = O.allocate(mu_alloc, w_rot // 2)
        W_KR, W_QR, Psi, slot_plane, rdiag = O.rotary_maps(Qt, Kt, Vh, m, S_ih, Sigma_l, r)
        rot_meta = {"c_alloc": [float(x) for x in c_alloc],
                    "cost_tail": float(sum(c_alloc[p] * mu[p][int(m[p]):].sum() for p in range(P))),
                    "mu_tail": float(sum(mu[p][int(m[p]):].sum() for p in range(P)))}
    else:
        if rotary_alloc is not None:
            raise ValueError(f"rotary_source 'transmla' takes TransMLA's own allocation; rotary_alloc must be None, "
                             f"got {rotary_alloc!r}")
        W_KR, W_QR, Psi, slot_plane, m, lam_tail = transmla_rotary(Qt, Kt, Sigma_l, w_rot=w_rot, r=r)
        vp = O.varpi_of(GK_l, r)
        rdiag = {"tsigt_err": []}
        rot_meta = {"c_alloc": None, "cost_tail": lam_tail, "mu_tail": lam_tail}
    Kres = np.stack([Kt[i][:r] - Psi[i] @ W_KR for i in range(H)])
    mu_p = UW.readout_scalars(theta, T, readout)
    Kres_t = apply_readout(Kres, planes, mu_p, P)

    Gk = np.asarray(GK_l, dtype=np.float64)
    A_free = np.ascontiguousarray(Gk[:, r:, r:])
    A_V = np.ascontiguousarray(np.asarray(GV_l, dtype=np.float64))
    Z_c = _joint_stack(Kt, Vt, Kres_t, A_free, A_V, GK_l, H, r, rows)
    _, s, Vt_c = np.linalg.svd(Z_c @ S_h, full_matrices=False)
    if d_c > Vt_c.shape[0]:
        raise ValueError(f"d_c = {d_c} exceeds the content stack's {Vt_c.shape[0]} directions")
    Wbar = Vt_c[:d_c].copy()
    C = np.linalg.qr((Wbar @ S_ih).T)[0].T
    Mread = S_h @ Wbar.T @ (Wbar @ S_ih @ C.T)
    lam = s ** 2

    W_UK = np.concatenate([np.concatenate([Kt[i][r:], Kres_t[i][rows]], axis=0) @ Mread
                           for i in range(H)], axis=0)
    W_UV = np.concatenate([Vt[i] @ Mread for i in range(H)], axis=0)
    W_UK = np.concatenate([W_UK, np.zeros((W_UK.shape[0], 1))], axis=1)
    W_UV = np.concatenate([W_UV, np.zeros((W_UV.shape[0], 1))], axis=1)
    W_UQ = np.concatenate([np.concatenate([Qt[i][r:], Qt[i][rows]], axis=0)
                           for i in range(H)], axis=0)

    layer = {
        "slot_plane": np.ascontiguousarray(slot_plane, dtype=np.int64),
        "slot_head": np.full(w_rot, -1, dtype=np.int64),
        "dC": np.full(H, dC0, dtype=np.int64),
        "W_DKV": np.ascontiguousarray(C, dtype=np.float64),
        "W_KR": np.ascontiguousarray(W_KR, dtype=np.float64),
        "W_UQ": np.ascontiguousarray(W_UQ, dtype=np.float64),
        "W_QR": np.ascontiguousarray(W_QR, dtype=np.float64),
        "W_UK": np.ascontiguousarray(W_UK, dtype=np.float64),
        "W_UV": np.ascontiguousarray(W_UV, dtype=np.float64),
        "Psi": np.ascontiguousarray(Psi, dtype=np.float64),
    }
    meta = {
        "m_p": m.tolist(), "n_floored": int(n_floored), "dC0": int(dC0),
        "unrope": list(planes), "readout": readout, "rotary_alloc": rotary_alloc,
        "rotary_source": rotary_source,
        "c_alloc": rot_meta["c_alloc"],
        "cost_tail": rot_meta["cost_tail"],
        "mu_abs": [float(abs(mu_p[p])) for p in planes],
        "lam_tail": float(lam[d_c:].sum()),
        "mu_tail": rot_meta["mu_tail"],
        "res_energy": float((Kres ** 2).sum()),
        "res_energy_kept": float((Kres_t[:, rows, :] ** 2).sum()),
        "varpi_min": float(vp.min()),
        "tsigt_err_max": float(max(rdiag["tsigt_err"])) if len(rdiag["tsigt_err"]) else 0.0,
        "n_degenerate_kept": 0,
        "t0_proxy_damage": float(lam[d_c:].sum()),
    }
    return layer, meta

def _joint_stack(Kt, Vt, Kres_t, A_free, A_V, GK_l, H, r, rows):
    base = dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H, r, O.KV_EXCHANGE_RATE)
    if len(rows) == 0:
        return base
    G = np.asarray(GK_l, dtype=np.float64)
    extra = [gc.psd_pow(G[i][np.ix_(rows, rows)], 0.5) @ Kres_t[i][rows] for i in range(H)]
    return np.concatenate([base] + extra, axis=0)

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None,
                  readout, unrope, rotary_alloc, T=None, rotary_source="ours"):
    geom = gc.geometry(model)
    L, H, d_k, d_v, dm, r = geom
    GK, GV, w, d_c = O._check_inputs(geom, w, d_c, G, fisher, sigma, s2norm)
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    if rotary_source not in ROTARY_SOURCES:
        raise ValueError(f"rotary_source {rotary_source!r} not in {ROTARY_SOURCES}")
    if rotary_source == "ours" and rotary_alloc not in ROTARY_ALLOCS:
        raise ValueError(f"rotary_alloc {rotary_alloc!r} not in {ROTARY_ALLOCS}")
    if rotary_source == "transmla" and rotary_alloc is not None:
        raise ValueError(f"rotary_source 'transmla' takes TransMLA's own allocation; rotary_alloc must be None, "
                         f"got {rotary_alloc!r}")
    P = r // 2
    unrope = tuple(range(P)) if unrope == "all" else tuple(int(p) for p in unrope)
    T = dt_optimal.T_EVAL if T is None else int(T)
    theta = gc.inv_freq(model)
    if len(theta) != P:
        raise ValueError(f"inv_freq has {len(theta)} entries but r/2 = {P}")
    layers, metas = [], []
    for li, lay in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        ld, md = build_layer(Qt, Kt, Vt, Wo, GK[li], GV[li], sigma[li], w_rot=w, d_c=d_c, r=r,
                             readout=readout, unrope=unrope, theta=theta, T=T,
                             rotary_alloc=rotary_alloc, rotary_source=rotary_source)
        layers.append(ld)
        metas.append(md)
    dC0 = int(metas[0]["dC0"])
    _rot_text = (f"rotary path and Psi are {O.VERSION}'s "
                 f"unchanged; rotary slots allocated by {rotary_alloc!r} "
                 f"(cost = c_p mu with c_p = E|e^(i th d) - mu_p|^2; energy = "
                 f"mu alone, dt_mla_ours's drop-semantics order); ") if rotary_source == "ours" else (
                 f"rotary key TransMLA's, rows and allocation (rotary_source 'transmla': {TM.VERSION}'s RoRoPE "
                 f"per-plane PCA on Sigma, global top w/2 by undamped eigenvalue, Psi = its real U_p, W_QR at "
                 f"scale d_h^-1/2); ")
    out = {
        "schema": SCHEMA, "arm": ARM, "version": VERSION, "rotary_mode": ROTARY_MODE,
        "G": None if G is None else int(G), "B": int(w + d_c), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_k), "dm": int(dm), "r": int(r),
        "provenance": {"fisher_sha256": O._digest((GK, GV)), "sigma_sha256": O._digest(sigma),
                       "s2norm_sha256": None,
                       "text": (f"{ARM} ({VERSION}): dt_mla_ours's construction with the rotary "
                                f"overflow UN-ROPED into the content at read-out {readout!r} "
                                f"(mu = 1 for verbatim, E_delta[e^(i theta delta)] over T = {T} "
                                f"for damped); un-roped planes {list(unrope)} of {P}, so "
                                f"dC = {dC0} (= d_h - r + 2|unroped|); residual taken as "
                                f"Kt[:r] - Psi W_KR; " + _rot_text +
                                f"scale {float(d_k) ** -0.5}")},
        "bias_set": sorted(BIAS_SET),
        "mla_scale": float(d_k) ** -0.5,
        "layers": layers,
        "_meta": {"layers": metas, "readout": readout, "unrope": list(unrope), "T": int(T),
                  "rotary_alloc": rotary_alloc, "rotary_source": rotary_source,
                  "cost_tail": float(sum(md["cost_tail"] for md in metas)),
                  "mu_tail": float(sum(md["mu_tail"] for md in metas)),
                  "dC0": dC0,
                  "t0_proxy_damage": float(sum(md["t0_proxy_damage"] for md in metas)),
                  "res_energy": float(sum(md["res_energy"] for md in metas)),
                  "n_degenerate_kept": 0,
                  "varpi_min": float(min(md["varpi_min"] for md in metas)),
                  "tsigt_err_max": float(max(md["tsigt_err_max"] for md in metas))},
    }
    self_check(out)
    return out

def self_check(wd):
    if wd["schema"] != SCHEMA or wd["arm"] != ARM or wd["rotary_mode"] != ROTARY_MODE:
        raise AssertionError("schema / arm / rotary_mode mismatch")
    if wd["bias_set"] != sorted(BIAS_SET) or wd["mla_scale"] != float(wd["d_h"]) ** -0.5:
        raise AssertionError("bias_set / mla_scale mismatch")
    w, d_c, H, r, dm, d_h = wd["w"], wd["d_c"], wd["H"], wd["r"], wd["dm"], wd["d_h"]
    n, nh, P = dm + 1, w // 2, r // 2
    if w % 2 or w + d_c != wd["B"] or (wd["G"] is not None and wd["B"] != wd["G"] * 2 * d_h):
        raise AssertionError("width rule violated")
    dC0 = int(wd["_meta"]["dC0"])
    if dC0 != (d_h - r) + 2 * len(wd["_meta"]["unrope"]):
        raise AssertionError(f"dC0 = {dC0} != (d_h - r) + 2|unroped|")
    if dC0 == 0:
        raise AssertionError("dC0 = 0: W_UQ and W_UK have zero rows, so every array check below "
                             "passes on an empty array (dt_mla_ours.self_check's Stage-2 guard)")
    if w == 0:
        raise AssertionError("w = 0: the slot and Psi checks below are vacuous (D4 forbids it)")
    D = H * dC0
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
        if not np.all(lay["slot_head"] == -1) or not np.all(lay["dC"] == dC0):
            raise AssertionError(f"layer {li}: slot_head / dC wrong for shared mode at dC0 = {dC0}")
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
