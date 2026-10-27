import numpy as np
import torch

import gam_core_llama
import kvlab

def geometry(model):
    if gam_core_llama.is_llama(model):
        return gam_core_llama.geometry(model)
    L, H, d_k, dm = kvlab.geometry(model)
    attn = kvlab.get_layers(model)[0].attention
    if not hasattr(attn, "rotary_ndims"):
        raise RuntimeError(
            "%s's attention exposes no `rotary_ndims`, and this reader will not guess one. "
            "Pythia's 16-of-64 was the old default and is wrong for every other family; add an "
            "accessor module for this architecture (see gam_core_llama) rather than defaulting."
            % type(model).__name__)
    return L, H, d_k, d_k, dm, int(attn.rotary_ndims)

def inv_freq(model):
    if gam_core_llama.is_llama(model):
        return gam_core_llama.inv_freq(model)
    return model.gpt_neox.rotary_emb.inv_freq.detach().double().numpy().copy()

def ln_gain(layer):
    if getattr(layer.input_layernorm, "weight", None) is None:
        return gam_core_llama.ln_gain(layer)
    return layer.input_layernorm.weight.detach().double().numpy().copy()

def split_wo(layer, H, d_v, dm):
    if hasattr(layer, "self_attn"):
        return gam_core_llama.split_wo(layer, H, d_v, dm)
    W = layer.attention.dense.weight.detach().double().numpy()
    b = layer.attention.dense.bias
    b = None if b is None else b.detach().double().numpy()
    return np.stack([W[:, i * d_v:(i + 1) * d_v] for i in range(H)]), b

def merge_wo(layer, Wo, b=None):
    if hasattr(layer, "self_attn"):
        return gam_core_llama.merge_wo(layer, Wo, b)
    W = np.concatenate([Wo[i] for i in range(Wo.shape[0])], axis=1)
    p = layer.attention.dense
    with torch.no_grad():
        p.weight.copy_(torch.from_numpy(W).to(p.weight.dtype))
        if b is not None and p.bias is not None:
            p.bias.copy_(torch.from_numpy(b).to(p.bias.dtype))

def augment(W, b):
    return np.concatenate([W, b[:, :, None]], axis=2)

def deaugment(Wt):
    return Wt[..., :-1].copy(), Wt[..., -1].copy()

SIGMA_CASES = ("A", "B", "C", "D")

SIGMA_DOC = {
    "A": "diag(I, 0)   bias dropped from the objective; the un-augmented anchor",
    "B": "I            plain augmentation: constant coordinate has a stream coordinate's moment",
    "C": "diag(I, RMS(gamma)^2)  isotropic stream, bias scaled to the typical coordinate",
    "D": "diag(gamma^2, 1)       gamma applied to stream and constant alike (consistent one)",
}

def sigma_half(case, gamma):
    g = np.abs(np.asarray(gamma, dtype=np.float64))
    dm = g.shape[0]
    if case == "A":
        return np.concatenate([np.ones(dm), [0.0]])
    if case == "B":
        return np.ones(dm + 1)
    if case == "C":
        return np.concatenate([np.ones(dm), [float(np.sqrt((g ** 2).mean()))]])
    if case == "D":
        return np.concatenate([g, [1.0]])
    raise ValueError(f"unknown Sigma case {case!r}")

def whiten(Wt, sh):
    return Wt * sh[None, :] if Wt.ndim == 2 else Wt * sh[None, None, :]

def unwhiten(Wt, sh):
    if np.any(sh == 0.0):
        raise ValueError("unwhiten called on a singular Sigma (case A) -- use the case-A branch")
    return Wt / sh[None, :] if Wt.ndim == 2 else Wt / sh[None, None, :]

def omega_commutant(c, Delta, d_k, r):
    h = r // 2
    Om = np.zeros((d_k, d_k), dtype=np.float64)
    Oi = np.zeros((d_k, d_k), dtype=np.float64)
    c = np.asarray(c, dtype=np.complex128)
    if np.any(c == 0):
        raise ValueError("singular Omega: a plane scalar is zero (CWT Prop 1(iv))")
    if not np.all(np.isfinite(c)):
        raise ValueError(
            f"non-finite Omega: {int((~np.isfinite(c)).sum())} of {c.size} plane scalars are "
            "inf/nan -- a zero inner product reached c = 1/ip (arm3B's failure mode, IA3)")
    ci = 1.0 / c
    idx = np.arange(h)
    for M, z in ((Om, c), (Oi, ci)):
        M[idx, idx] = z.real
        M[idx + h, idx + h] = z.real
        M[idx, idx + h] = -z.imag
        M[idx + h, idx] = z.imag
    if d_k > r:
        if not np.all(np.isfinite(Delta)):
            raise ValueError(
                f"non-finite Omega: the free block carries {int((~np.isfinite(Delta)).sum())} "
                "inf/nan entries (IA3)")
        Om[r:, r:] = Delta
        Oi[r:, r:] = np.linalg.inv(Delta)
    if not (np.all(np.isfinite(Om)) and np.all(np.isfinite(Oi))):
        raise ValueError("non-finite Omega: the assembled gauge or its inverse is not finite (IA3)")
    return Om, Oi

def rot(theta, delta, d_k, r):
    h = r // 2
    R = np.eye(d_k, dtype=np.float64)
    ang = np.asarray(theta, dtype=np.float64) * float(delta)
    co, si = np.cos(ang), np.sin(ang)
    idx = np.arange(h)
    R[idx, idx] = co
    R[idx + h, idx + h] = co
    R[idx, idx + h] = -si
    R[idx + h, idx] = si
    return R

def offset_weights(T):
    d = np.arange(T, dtype=np.float64)
    w = (T - d)
    return d, w / w.sum()

def damping(theta, T, r_free=True):
    d, w = offset_weights(T)
    amp = lambda om: float(abs((w * np.exp(1j * om * d)).sum()))
    n = len(theta)
    rows = []
    for j in range(n):
        rows.append((f"2t{j}", 2 * theta[j]))
        if r_free:
            rows.append((f"t{j}xfree", theta[j]))
        for k in range(n):
            if k != j:
                rows.append((f"t{j}-t{k}", abs(theta[j] - theta[k])))
                rows.append((f"t{j}+t{k}", theta[j] + theta[k]))
    out = [(lab, om, om * T, amp(om)) for lab, om in rows]
    out.sort(key=lambda x: -x[3])
    return out, out[0]

def _M(z):
    return np.array([[z.real, -z.imag], [z.imag, z.real]], dtype=np.float64)

def _T(z):
    return np.array([[z.real, z.imag], [z.imag, -z.real]], dtype=np.float64)

def abar_emp(A, theta, T, d_k, r):
    h = r // 2
    d, w = offset_weights(T)
    ph = lambda om: complex((w * np.exp(1j * om * d)).sum())
    theta = np.asarray(theta, dtype=np.float64)
    Ab = np.zeros_like(A)
    for j in range(h):
        for k in range(h):
            B = A[np.ix_([j, j + h], [k, k + h])]
            alpha = 0.5 * (B[0, 0] + B[1, 1])
            beta = 0.5 * (B[1, 0] - B[0, 1])
            p = 0.5 * (B[0, 0] - B[1, 1])
            q = 0.5 * (B[0, 1] + B[1, 0])
            blk = (_M(complex(alpha, beta) * ph(theta[j] - theta[k]))
                   + _T(complex(p, q) * ph(theta[j] + theta[k])))
            Ab[np.ix_([j, j + h], [k, k + h])] = blk
    if d_k > r:
        fr = list(range(r, d_k))
        for j in range(h):
            Mz = _M(ph(theta[j]))
            Ab[np.ix_([j, j + h], fr)] = Mz @ A[np.ix_([j, j + h], fr)]
            Ab[np.ix_(fr, [j, j + h])] = A[np.ix_(fr, [j, j + h])] @ Mz.T
        Ab[r:, r:] = A[r:, r:]
    return Ab

def abar_emp_direct(A, theta, T, d_k, r):
    d, w = offset_weights(T)
    Ab = np.zeros_like(A)
    for delta, wd in zip(d, w):
        R = rot(theta, delta, d_k, r)
        Ab += wd * (R @ A @ R.T)
    return Ab

def query_gram(Qt_c):
    return Qt_c @ Qt_c.T

def xi_emp_stack(Kt_c, Abars):
    blocks = []
    for K, Ab in zip(Kt_c, Abars):
        blocks.append(psd_pow(Ab, 0.5) @ K)
    return np.concatenate(blocks, axis=0)

def psd_pow(A, p):
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.clip(w, 0.0, None) ** p) @ V.T

def tail_svd(stack, d):
    U, s, Vt = np.linalg.svd(stack, full_matrices=False)
    tail = float((s[d:] ** 2).sum()) if len(s) > d else 0.0
    return tail, Vt[:d].copy()

def to_complex(Wt, r):
    h = r // 2
    return Wt[:h, :] + 1j * Wt[h:r, :]

def from_complex(Z, d_k, r, free=None):
    h = r // 2
    out = np.zeros((d_k, Z.shape[1]), dtype=np.float64)
    out[:h, :] = Z.real
    out[h:r, :] = Z.imag
    if free is not None:
        out[r:, :] = free
    return out

def read_layer(layer, H, d_k, d_v, dm):
    split = gam_core_llama.split_qkv if hasattr(layer, "self_attn") else kvlab.split_qkv
    Wq, Wk, Wv, bq, bk, bv = split(layer, H, d_k, dm)
    Wo, _ = split_wo(layer, H, d_v, dm)
    return augment(Wq, bq), augment(Wk, bk), augment(Wv, bv), Wo

def write_layer(layer, Qt, Kt, Vt, Wo, H, d_k, d_v, dm):
    Wq, bq = deaugment(Qt)
    Wk, bk = deaugment(Kt)
    Wv, bv = deaugment(Vt)
    qkv_merge(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm)
    merge_wo(layer, Wo)

def qkv_split(layer, H, d_k, dm):
    return (gam_core_llama.split_qkv if hasattr(layer, "self_attn") else kvlab.split_qkv)(layer, H, d_k, dm)

def qkv_merge(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm):
    return (gam_core_llama.merge_qkv if hasattr(layer, "self_attn") else kvlab.merge_qkv)(
        layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm)

def pool_layer(layer, G, H, d_k, dm, which="kv"):
    return (gam_core_llama.pool_layer if hasattr(layer, "self_attn") else kvlab.pool_layer)(
        layer, G, H, d_k, dm, which=which)

def apply_gauge(Qt, Kt, Vt, Wo, Om, Oi, Gm, Gi):
    Qo, Ko, Vo, Oo = Qt.copy(), Kt.copy(), Vt.copy(), Wo.copy()
    H = Qt.shape[0]
    for i in range(H):
        if Om is not None:
            Ko[i] = Om[i] @ Kt[i]
            Qo[i] = Oi[i].T @ Qt[i]
        if Gm is not None:
            Vo[i] = Gm[i] @ Vt[i]
            Oo[i] = Wo[i] @ Gi[i]
    return Qo, Ko, Vo, Oo

def head_group(groups, H):
    g = np.zeros(H, dtype=np.int64)
    for j, grp in enumerate(groups):
        for i in grp:
            g[i] = j
    return g

def tie(Kt, Kbar, groups=None):
    Kbar = np.asarray(Kbar)
    if Kbar.ndim == 2:
        return np.broadcast_to(Kbar, Kt.shape).copy()
    H = Kt.shape[0]
    if groups is None:
        groups = [list(range(H))]
    gi = head_group(groups, H)
    return np.stack([Kbar[gi[i]] for i in range(H)])

def layer_maps(layer, H, d_k, d_v, dm, case):
    Qt, Kt, Vt, Wo = read_layer(layer, H, d_k, d_v, dm)
    shipped = (Qt, Kt, Vt, Wo)
    if case == "A":
        return Qt[..., :dm].copy(), Kt[..., :dm].copy(), Vt[..., :dm].copy(), Wo, None, shipped
    sh = sigma_half(case, ln_gain(layer))
    return whiten(Qt, sh), whiten(Kt, sh), whiten(Vt, sh), Wo, sh, shipped

def serve_maps(arm, sh, shipped, case, groups=None):
    Qt, Kt, Vt, Wo = shipped
    H = Qt.shape[0]
    if groups is None:
        groups = [list(range(H))]
    Kb, Vb = np.atleast_3d(arm["Kbar"]), np.atleast_3d(arm["Vbar"])
    if Kb.shape[0] != len(groups):
        Kb, Vb = arm["Kbar"][None], arm["Vbar"][None]
    if case != "A":
        return (unwhiten(Kb, sh), unwhiten(Vb, sh), unwhiten(arm["Qh"], sh), arm["Oh"])
    bk = np.stack([Kt[g, :, -1].mean(axis=0) for g in groups])
    bv = np.stack([Vt[g, :, -1].mean(axis=0) for g in groups])
    Kbar = np.concatenate([Kb, bk[:, :, None]], axis=2)
    Vbar = np.concatenate([Vb, bv[:, :, None]], axis=2)
    if arm.get("Oi") is not None:
        bq = np.stack([arm["Oi"][i].T @ Qt[i][:, -1] for i in range(H)])
    else:
        bq = Qt[:, :, -1]
    Qh = np.concatenate([arm["Qh"], bq[:, :, None]], axis=2)
    return Kbar, Vbar, Qh, arm["Oh"]

def logits_of(model, ids):
    with torch.no_grad():
        return model(ids.to(next(model.parameters()).device)).logits.detach().clone()

def nll(logits, ids):
    return float(torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:]))

def rel_logit_move(a, b):
    d = (a.double() - b.double()).abs().max().item()
    s = a.double().abs().max().item()
    return d / max(s, 1e-30)
