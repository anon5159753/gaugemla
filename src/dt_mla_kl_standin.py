import contextlib
import sys

import numpy as np
import torch

import dt_mla_ours as _O
import dt_mla_unrope as _UN

SCHEMA = "kl_standin/v1"
_B2 = np.stack([np.eye(2).reshape(4), np.array([[0., -1.], [1., 0.]]).reshape(4)], -1)
_STATE = {"stand": None}

def _geometry(model):
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    Hkv = int(getattr(model, "_gqa_kv_heads", None) or getattr(cfg, "num_key_value_heads", None) or H)
    rot = [m for n, m in model.named_modules() if n.endswith("rotary_emb")]
    if len(rot) != 1:
        raise RuntimeError(f"expected one rotary_emb module, found {len(rot)}")
    if float(getattr(rot[0], "attention_scaling", 1.0)) != 1.0:
        raise RuntimeError(f"attention_scaling {rot[0].attention_scaling} != 1: cos/sin are not pure rotations")
    theta = rot[0].inv_freq.detach().double().cpu().numpy().copy()
    return L, H, Hkv, theta

def measure(model, ids, label_seed, excluded, log=print):
    L, H, Hkv, theta = _geometry(model)
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of Hkv = {Hkv}")
    nrep = H // Hkv
    P = len(theta)
    r = 2 * P
    n_doc, T = ids.shape
    if len(excluded) != n_doc or any(len(e) != L for e in excluded):
        raise ValueError(f"excluded must be (doc, layer, k) over {n_doc} docs and {L} layers")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("measure() captures the eager attention function; load the model with kvlab.load_model")
    MOD = sys.modules[type(model).__module__]
    for name in ("apply_rotary_pos_emb", "eager_attention_forward"):
        if not callable(getattr(MOD, name, None)):
            raise RuntimeError(f"{MOD.__name__} has no {name}: this capture does not apply")
    orig_rope, orig_attn = MOD.apply_rotary_pos_emb, MOD.eager_attention_forward
    dev = next(model.parameters()).device
    th = torch.tensor(theta, dtype=torch.float64, device=dev)
    ph = torch.exp(1j * th[None, :] * torch.arange(T, dtype=torch.float64, device=dev)[:, None])
    pend, cap = {}, {}

    def rope_w(q, k, *a, **kw):
        pend["qk"] = (q.detach(), k)
        return orig_rope(q, k, *a, **kw)

    def attn_w(module, query, key, value, attention_mask, **kw):
        li = module.layer_idx
        if kw.get("scaling") is None:
            raise RuntimeError("no scaling passed to the attention function")
        qin, kin = pend.pop("qk")
        cap[(li, "qin")] = qin
        kin.register_hook(lambda g, li=li: cap.__setitem__((li, "gkin"), g.detach()))
        cap[(li, "q")], cap[(li, "k")], cap[(li, "v")] = query.detach(), key.detach(), value.detach()
        cap[(li, "scale")] = float(kw["scaling"])
        out, w = orig_attn(module, query, key, value, attention_mask, **kw)
        cap[(li, "o")] = out.detach()
        out.register_hook(lambda g, li=li: cap.__setitem__((li, "go"), g.detach()))
        return out, w

    z = lambda *s: np.zeros((L, Hkv) + s)
    M4, c4, Faar, Fbbr = z(P, 4, 4), z(P, 4), z(r, r), z(r, r)
    gl, lmax = np.zeros((3, L)), np.zeros(L)
    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    n_tok = 0
    hs = [model.get_input_embeddings().register_forward_hook(lambda _m, _i, o: o.requires_grad_(True))]
    MOD.apply_rotary_pos_emb, MOD.eager_attention_forward = rope_w, attn_w
    try:
        for d in range(n_doc):
            cap.clear()
            chunk = ids[d:d + 1].to(dev)
            model.zero_grad(set_to_none=True)
            _lg = model(chunk).logits[:, :-1].float()
            with torch.no_grad():
                _v = torch.multinomial(torch.softmax(_lg.reshape(-1, _lg.shape[-1]), dim=-1).cpu(), 1,
                                       generator=torch.Generator().manual_seed(label_seed + d))
            torch.nn.functional.cross_entropy(_lg.reshape(-1, _lg.shape[-1]), _v.view(-1).to(dev)).backward()
            del _lg
            if any((li, "go") not in cap or (li, "gkin") not in cap for li in range(L)):
                raise RuntimeError("not every layer's attention was captured")
            with torch.no_grad():
                for li in range(L):
                    q = cap[(li, "q")][0].double().transpose(0, 1)
                    k = cap[(li, "k")][0].double().transpose(0, 1)
                    v = cap[(li, "v")][0].double().transpose(0, 1)
                    if k.shape[1] != H or q.shape[1] != H:
                        raise RuntimeError(f"layer {li}: {q.shape[1]} query / {k.shape[1]} key heads, expected {H}")
                    kv = k.view(T, Hkv, nrep, -1)
                    if nrep > 1 and not torch.equal(kv, kv[:, :, :1].expand_as(kv)):
                        raise RuntimeError(f"layer {li}: the expanded key copies differ")
                    ao = cap[(li, "o")][0].double()
                    go = cap[(li, "go")][0].double()
                    qin = cap[(li, "qin")][0].double().transpose(0, 1)[..., :r]
                    gkin = cap[(li, "gkin")][0].double().transpose(0, 1)[..., :r]
                    sc_ = cap[(li, "scale")]
                    s_ = torch.einsum("mhd,nhd->hmn", q, k) * sc_
                    lmax[li] = max(lmax[li], float(s_.masked_fill(~causal, 0.0).abs().max()))
                    Pm = torch.softmax(s_.masked_fill(~causal, float("-inf")), -1)
                    del s_
                    gl[1, li] = max(gl[1, li], float((torch.einsum("hmn,nhd->mhd", Pm, v) - ao).norm() / ao.norm()))
                    dP = torch.einsum("mhd,nhd->hmn", go, v)
                    G = Pm * (dP - (Pm * dP).sum(-1, keepdim=True)) * sc_
                    del dP, Pm
                    Qr = torch.complex(q[..., :P], q[..., P:r])
                    Q = Qr * ph.conj()[:, None, :]
                    Kc = torch.complex(k[:, ::nrep, :P], k[:, ::nrep, P:r]) * ph.conj()[:, None, :]
                    Qin = torch.complex(qin[..., :P], qin[..., P:])
                    gl[0, li] = max(gl[0, li], float((Qin * ph[:, None, :] - Qr).abs().norm() / Qr.abs().norm()))
                    Gc = G.to(torch.complex128)
                    del G
                    beta = (torch.einsum("hmn,mhp->nhp", Gc, Qr) * ph.conj()[:, None, :]).view(T, Hkv, nrep, P).sum(2)
                    alpha = torch.einsum("hmn,mhp->nhp", Gc, Q).view(T, Hkv, nrep, P).sum(2)
                    del Gc
                    gk = torch.complex(gkin[..., :P], gkin[..., P:]).view(T, Hkv, nrep, P).sum(2)
                    gl[2, li] = max(gl[2, li], float((beta - gk).abs().norm() / gk.abs().norm()))
                    keep = torch.ones(T, dtype=torch.bool, device=dev)
                    keep[torch.as_tensor(np.asarray(excluded[d][li], dtype=np.int64), device=dev)] = False
                    a = torch.stack([alpha.real, alpha.imag], -1)[keep]
                    b = torch.stack([beta.real, beta.imag], -1)[keep]
                    kk = torch.stack([Kc.real, Kc.imag], -1)[keep]
                    phi = (a[..., :, None] * kk[..., None, :]).reshape(*a.shape[:3], 4)
                    y = (b * kk).sum(-1)
                    M4[li] += torch.einsum("njpr,njps->jprs", phi, phi).cpu().numpy()
                    c4[li] += torch.einsum("njpr,njp->jpr", phi, y).cpu().numpy()
                    ar = torch.cat([alpha.real, alpha.imag], -1)[keep]
                    br = torch.cat([beta.real, beta.imag], -1)[keep]
                    Faar[li] += torch.einsum("nja,njb->jab", ar, ar).cpu().numpy()
                    Fbbr[li] += torch.einsum("nja,njb->jab", br, br).cpu().numpy()
            n_tok += T - len(excluded[d][0])
            if any(len(e) != len(excluded[d][0]) for e in excluded[d]):
                raise ValueError(f"document {d}: the layers leave out different token counts")
            if (d + 1) % 16 == 0 or d + 1 == n_doc:
                log(f"    stand-in stats {d + 1}/{n_doc} docs  gates g0 {gl[0].max():.1e} g1 {gl[1].max():.1e} "
                    f"g2 {gl[2].max():.1e}")
    finally:
        MOD.apply_rotary_pos_emb, MOD.eager_attention_forward = orig_rope, orig_attn
        for h in hs:
            h.remove()
        model.zero_grad(set_to_none=True)
    tol = np.where(lmax < 1e3, 1e-3, 5e-3)
    bad = [(i, li) for i in range(3) for li in range(L) if not gl[i, li] < tol[li]]
    if bad:
        raise RuntimeError(f"stand-in capture gates failed at (gate, layer) {bad[:6]}: g0 {gl[0].max():.1e} "
                           f"g1 {gl[1].max():.1e} g2 {gl[2].max():.1e}")
    return dict(schema=SCHEMA, theta=theta, M4=M4, c4=c4, Faar=Faar, Fbbr=Fbbr, n_tok=int(n_tok), nrep=int(nrep),
                gates=gl, max_abs_logit=lmax, gate_tol=tol)

def commutant(M4, c4):
    Mc = np.einsum("ra,...rs,sb->...ab", _B2, M4, _B2)
    cc = np.einsum("ra,...r->...a", _B2, c4)
    uv = np.linalg.solve(Mc, cc[..., None])[..., 0]
    return uv[..., 0] + 1j * uv[..., 1]

def standin(stats, fisher, model, tol=1e-2, log=print):
    L, H, Hkv, theta = _geometry(model)
    GK = np.asarray(fisher[0])
    P = len(theta)
    r = 2 * P
    nrep = H // Hkv
    if stats["schema"] != SCHEMA or stats["nrep"] != nrep or stats["M4"].shape != (L, Hkv, P, 4, 4):
        raise ValueError(f"stats do not belong to this model: {stats['schema']}, nrep {stats['nrep']}, "
                         f"M4 {stats['M4'].shape}")
    if not np.allclose(stats["theta"], theta, rtol=0, atol=1e-12):
        raise ValueError("stats measured at different rotary frequencies")
    s = 1.0 / (nrep * stats["n_tok"])
    err = np.zeros((L, H))
    for li in range(L):
        for i in range(H):
            ref = GK[li, i][:r, :r]
            err[li, i] = np.linalg.norm(stats["Fbbr"][li, i // nrep] * s - ref) / np.linalg.norm(ref)
    log(f"  stand-in: Fbb / (n_rep n_tok) vs each copy's GK rotary block, max rel {err.max():.1e} (tol {tol:g})")
    if not err.max() <= tol:
        li, i = np.unravel_index(int(np.argmax(err)), err.shape)
        raise RuntimeError(f"the stand-in's b is not the Fisher's gradient: layer {li} head {i} off by {err.max():.2e}")
    mu = commutant(stats["M4"], stats["c4"])
    MU = np.repeat(mu, nrep, axis=1)
    FAA = np.repeat(stats["Faar"], nrep, axis=1) * s
    return dict(schema=SCHEMA, MU=MU, FAA=FAA, theta=theta, fisher_sha256=_O._digest(fisher),
                standin_sha256=_O._digest([MU.real, MU.imag, FAA]), gk_rel_err=float(err.max()))

def install(stand):
    for k in ("schema", "MU", "FAA", "theta", "fisher_sha256", "standin_sha256"):
        if k not in stand:
            raise ValueError(f"stand-in has no {k!r}")
    if stand["schema"] != SCHEMA:
        raise ValueError(f"stand-in schema {stand['schema']!r} != {SCHEMA!r}")
    if _O._digest([stand["MU"].real, stand["MU"].imag, stand["FAA"]]) != stand["standin_sha256"]:
        raise ValueError("stand-in arrays do not hash to their recorded digest")
    _STATE["stand"] = stand

def for_fisher(fisher):
    st = _STATE["stand"]
    if st is None:
        raise RuntimeError("no KL stand-in installed: the init's stand-in block must run before a *_kl arm is built")
    if fisher is None or _O._digest(fisher) != st["fisher_sha256"]:
        raise RuntimeError(f"the installed KL stand-in was measured for Fisher {st['fisher_sha256']}, not this one")
    return st

@contextlib.contextmanager
def applied(stand, model):
    L, H, _Hkv, theta = _geometry(model)
    P = len(theta)
    if stand["MU"].shape != (L, H, P) or stand["FAA"].shape != (L, H, 2 * P, 2 * P):
        raise ValueError(f"stand-in shapes {stand['MU'].shape} / {stand['FAA'].shape} do not fit L {L}, H {H}, P {P}")
    if not np.allclose(stand["theta"], theta, rtol=0, atol=1e-12):
        raise ValueError("stand-in measured at different rotary frequencies")
    MU, FAA = stand["MU"], stand["FAA"]
    _build_layer, _apply, _joint = _UN.build_layer, _UN.apply_readout, _UN._joint_stack
    cur = {"next": 0, "li": None}

    def build_layer_w(*a, **kw):
        cur["li"] = cur["next"]
        cur["next"] += 1
        return _build_layer(*a, **kw)

    def apply_w(Kres, rows, mu_p, P_):
        mu = MU[cur["li"]]
        out = np.array(Kres, dtype=np.float64, copy=True)
        for p in rows:
            a, b = mu[:, p].real[:, None], mu[:, p].imag[:, None]
            x, y = Kres[:, p, :], Kres[:, p + P_, :]
            out[:, p, :] = a * x - b * y
            out[:, p + P_, :] = b * x + a * y
        return out

    def joint_w(Kt, Vt, Kres_t, A_free, A_V, GK_l, H_, r_, rows):
        if len(rows) == 0:
            return _joint(Kt, Vt, Kres_t, A_free, A_V, GK_l, H_, r_, rows)
        base = _UN.dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H_, r_, _UN.O.KV_EXCHANGE_RATE)
        M = FAA[cur["li"]]
        extra = [_UN.gc.psd_pow(M[i][np.ix_(rows, rows)], 0.5) @ Kres_t[i][rows] for i in range(H_)]
        return np.concatenate([base] + extra, axis=0)

    _UN.build_layer, _UN.apply_readout, _UN._joint_stack = build_layer_w, apply_w, joint_w
    try:
        yield
    finally:
        _UN.build_layer, _UN.apply_readout, _UN._joint_stack = _build_layer, _apply, _joint
    if cur["next"] != L:
        raise RuntimeError(f"the build passed {cur['next']} layers through the stand-in, expected {L}")

def stamp(out, stand, model):
    L, _H, _Hkv, _t = _geometry(model)
    pv = out["provenance"]
    pv["text"] = (f"{out['arm']}: the KL stand-in (@@KLSTAND@@, {SCHEMA}): each head's un-roped planes read out by its "
                  f"own A* = (Faa)^-1 Fab (commutant fit), in place of the ramp's mean rotation, and the un-roped rows "
                  f"weighted by Faa in the latent's stack, in place of GK's rotary block; the cost rule's price is the "
                  f"ramp's. Stand-in {stand['standin_sha256']}, measured on Fisher {stand['fisher_sha256']}. -- "
                  + pv["text"])
    layers = out["_meta"]["layers"]
    if len(layers) != L:
        raise RuntimeError(f"{len(layers)} layer metas, expected {L}")
    for li, md in enumerate(layers):
        md["mu_source"] = "kl_standin"
        md["mu_abs"] = [float(np.median(np.abs(stand["MU"][li, :, p]))) for p in md["unrope"]]
    out["_meta"]["mu_source"] = "kl_standin"
    return out
