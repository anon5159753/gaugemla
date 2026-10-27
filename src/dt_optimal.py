import numpy as np
import kvlab
import gam_core as gc

T_EVAL = 512

VARIANTS = {
    "full":    "rope-averaged Xi, per-head read-out, K and V",
    "noquery": "A1 -- unweighted A_i = I, so the subspace ignores query magnitudes",
    "geom":    "A2 -- unweighted Xi^geom = sum_i Pi_i, pure row-space geometry",
    "delta0":  "A3 -- Xi at Delta=0, no offset averaging (A12 says the averaging never completes)",
    "konly":   "A4 -- key side only, W_V/W_O untouched",
    "vonly":   "value side only, W_K/W_Q untouched -- the mirror of konly",
    "identity": "the fixed identity frame -- the control, and the worst Xi-independent target",
    "identgauge": "arbitrary block frame, tied, with the best commutant gauge (closed form)",
    "optgauge":   "optimal target, tied, with the best commutant gauge",
}

def _offset_weights(T):
    w = np.array([T - d for d in range(T)], dtype=np.float64)
    return w / w.sum()

def _rope_avg(A, r, theta, wD):
    d = A.shape[0]
    out = A.copy()
    if r == 0:
        return out
    Arr, acc, Rbar = A[:r, :r], np.zeros((r, r)), np.zeros((r, r))
    h = r // 2
    for w_, dlt in zip(wD, range(len(wD))):
        th = theta * dlt
        C, S = np.diag(np.cos(th)), np.diag(np.sin(th))
        R = np.block([[C, -S], [S, C]])
        acc += w_ * (R @ Arr @ R.T)
        Rbar += w_ * R
    out[:r, :r] = acc
    if r < d:
        out[:r, r:] = Rbar @ A[:r, r:]
        out[r:, :r] = A[r:, :r] @ Rbar.T
    return out

def _rowspace(W):
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = int((s > max(W.shape) * np.finfo(float).eps * s[0]).sum())
    return Vt[:k]

def _block_frame(j, d, ncol):
    B = np.zeros((d, ncol))
    B[np.arange(d), np.arange(d) + d * j] = 1.0
    return B

def _solve_commutant(W, Abar, T, d_k, r):
    h, A = r // 2, Abar
    Wl = W @ T.T
    P = np.zeros((d_k, d_k))
    for p in range(h):
        v1, v2 = np.zeros(d_k), np.zeros(d_k)
        v1[p], v2[p + h] = 1.0, 1.0
        m1, m2 = Wl[:, p], Wl[:, p + h]
        den = v1 @ A @ v1 + v2 @ A @ v2
        a = (v1 @ A @ m1 + v2 @ A @ m2) / den
        b = (v2 @ A @ m1 - v1 @ A @ m2) / den
        P[p, p] = P[p + h, p + h] = a
        P[p, p + h], P[p + h, p] = -b, b
    Aff, Afr = A[r:, r:], A[r:, :r]
    for c in range(r, d_k):
        P[r:, c] = Wl[r:, c] - np.linalg.solve(Aff, Afr @ (-Wl[:r, c]))
    return P

def _shared_map(maps, A, g, d, variant):
    if variant == "identity":
        return np.eye(d, maps.shape[-1])
    if variant == "geom":
        Z = np.concatenate([_rowspace(maps[i]) for i in g], axis=0)
    elif variant == "noquery":
        Z = np.concatenate([maps[i] for i in g], axis=0)
    else:
        Z = np.concatenate([gc.psd_pow(A[i], 0.5) @ maps[i] for i in g], axis=0)
    return gc.tail_svd(Z, d)[1]

def build(model, G, variant="full", groups=None, groups_by_layer=None, G_v=None, wide=False,
          GK=None, GV=None, B=None):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    if (GK is None) != (GV is None):
        raise ValueError("GK and GV must be supplied together")
    if GK is not None and variant != "full":
        raise ValueError(f"a measured metric is only defined for variant='full'; got {variant!r}. "
                         f"The geom/noquery/identity variants exist to ABLATE the metric, so "
                         f"supplying one contradicts what they are for.")
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    c_k, c_v = d_k, d_v
    if B is not None:
        if not wide:
            raise ValueError(f"a budget B = {B} was passed to the GROUPED optimal arm, which needs a real "
                             f"group count G; only wide=True (optwide) takes a budget")
        if G is not None and B != G * (d_k + d_v):
            raise ValueError(f"budget B = {B} != G*(d_k+d_v) = {G * (d_k + d_v)} at G={G}")
        if (B * d_k) % (d_k + d_v):
            raise ValueError(f"budget B = {B} does not split as d_k:d_v = {d_k}:{d_v} into whole widths")
    if wide:
        if groups is not None or groups_by_layer is not None or G_v is not None:
            raise ValueError("wide=True fixes the partition at one group over all heads and widens "
                             "both sides symmetrically, so groups/groups_by_layer/G_v cannot also "
                             "be given")
        if variant in ("identgauge", "optgauge"):
            raise ValueError(f"wide=True is not defined for variant {variant!r}: it serves by TYING "
                             f"through a read-out confined to the d_k-square RoPE commutant, and a "
                             f"widened target has no counterpart inside that commutant")
        c_k, c_v = ((G * d_k, G * d_v) if B is None else
                    (B * d_k // (d_k + d_v), B * d_v // (d_k + d_v)))
        for side, c, d in (("key", c_k, d_k), ("value", c_v, d_v)):
            bound = min(H * d, dm + 1)
            if c > bound:
                raise ValueError(f"wide {side} width {c} at G={G} B={B} exceeds the stack's rank bound "
                                 f"{bound} (H*d={H * d}, dm+1={dm + 1}); the target cannot be wider "
                                 f"than the thing it is fitted to")
        groups = [list(range(H))]
    if groups is None:
        groups = kvlab.groups_of(H, G)
    if groups_by_layer is not None and len(groups_by_layer) != L:
        raise ValueError(f"groups_by_layer has {len(groups_by_layer)} entries, model has {L} layers")
    for gl in (groups_by_layer if groups_by_layer is not None else [groups]):
        assert sorted(i for g in gl for i in g) == list(range(H)), "groups must partition heads"
    groups_v = groups if G_v is None else kvlab.groups_of(H, G_v)
    theta = gc.inv_freq(model)
    wD = _offset_weights(T_EVAL)

    for li, lay in enumerate(layers):
        groups_l = groups_by_layer[li] if groups_by_layer is not None else groups
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        Qh, Oh = Qt.copy(), Wo.copy()
        Kb, Vb = [], []

        if variant == "vonly":
            Ktied = Kt
        else:
            if GK is not None:
                AK = np.ascontiguousarray(GK[li])
            else:
                AK = np.stack([Qt[i] @ Qt[i].T for i in range(H)])
                if variant != "delta0":
                    AK = np.stack([_rope_avg(AK[i], r, theta, wD) for i in range(H)])
            Kh = Kt.copy()
            if variant in ("identgauge", "optgauge"):
                for j, g in enumerate(groups_l):
                    T = (_block_frame(j, d_k, Kt.shape[-1]) if variant == "identgauge"
                         else _shared_map(Kt, AK, g, d_k, "full"))
                    for i in g:
                        Kh[i] = _solve_commutant(Kt[i], AK[i], T, d_k, r) @ T
                Ktied = Kh
                Vh = Vt.copy()
                AV = np.stack([Wo[i].T @ Wo[i] for i in range(H)])
                for j, g in enumerate(groups_l):
                    Tv = (_block_frame(j, d_v, Vt.shape[-1]) if variant == "identgauge"
                          else _shared_map(Vt, AV, g, d_v, "full"))
                    for i in g:
                        Vh[i] = Vt[i] @ Tv.T @ Tv
                Vtied = Vh
                gc.write_layer(lay, Qh, Ktied, Vtied, Oh, H, d_k, d_v, dm)
                continue
            for g in groups_l:
                B = _shared_map(Kt, AK, g, c_k, variant)
                Kb.append(B)
                for i in g:
                    Kh[i] = Kt[i] @ B.T @ B
            Ktied = Kh

        if variant == "konly":
            Vtied = Vt
        else:
            AV = (np.ascontiguousarray(GV[li]) if GV is not None
                  else np.stack([Wo[i].T @ Wo[i] for i in range(H)]))
            Vh = Vt.copy()
            for g in (groups_l if G_v is None else groups_v):
                Bv = _shared_map(Vt, AV, g, c_v, variant)
                Vb.append(Bv)
                for i in g:
                    Vh[i] = Vt[i] @ Bv.T @ Bv
            Vtied = Vh

        gc.write_layer(lay, Qh, Ktied, Vtied, Oh, H, d_k, d_v, dm)

    if wide and B is not None:
        return (f"optimal subspace WIDE: one group (all {H} heads), c_k={c_k} c_v={c_v}, split d_k:d_v of the "
                f"budget B={B} variant={variant} T={T_EVAL} ({VARIANTS[variant]})")
    if wide:
        return (f"optimal subspace WIDE: one group (all {H} heads), c_k={c_k}=({G}*{d_k}) "
                f"c_v={c_v}=({G}*{d_v}), cache-equivalent to G={G} variant={variant} T={T_EVAL} "
                f"({VARIANTS[variant]})")
    part = "per-layer matching" if groups_by_layer is not None else "contiguous"
    sides = f"G={G}" if G_v is None else f"G_K={G} G_V={G_v}"
    return (f"optimal subspace {sides} variant={variant} partition={part} T={T_EVAL} "
            f"({VARIANTS[variant]})")
