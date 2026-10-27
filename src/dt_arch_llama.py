import hashlib
import sys

import numpy as np
import torch
from torch import nn

import dt_decoupled
import dt_decoupled_arch
import dt_gqa_arch
import dt_optimal_arch
import dt_util
import gam_core as gc
import gam_core_llama
import kvlab

_TWINS = {}

def is_llama_layout(model):
    return hasattr(kvlab.get_layers(model)[0], "self_attn")

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def _rope(x, cos, sin):
    r = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x_rot, x_pass = x[..., :r], x[..., r:]
    return torch.cat(((x_rot * cos) + (_rotate_half(x_rot) * sin), x_pass), dim=-1)

def _rotary_module(model):
    return gam_core_llama._rotary_emb(model)

def _cos_sin_all(attn, hidden_states, position_embeddings, S, T):
    cos, sin = position_embeddings
    if S == T:
        return cos, sin
    pos = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
    cos_a, sin_a = attn._rotary_ref(hidden_states, pos)
    if not (torch.equal(cos_a[:, S - T:], cos) and torch.equal(sin_a[:, S - T:], sin)):
        raise RuntimeError(f"layer {attn.layer_idx}: the cached positions are not 0..{S - 1} ending at this call's "
                           f"{T}; this module re-rotates its pre-rotation cache and refuses to guess positions")
    return cos_a, sin_a

def _attend(attn, q, k, v, attention_mask, B, T):
    out, w = attn._host_eager(attn, q, k, v, attention_mask, scaling=attn.scaling,
                              dropout=0.0 if not attn.training else attn.attention_dropout)
    return attn.o_proj(out.reshape(B, T, -1).contiguous()), w

class _GQAMixin:

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                **kwargs):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, G, d = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, G, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, G, d).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if G != H:
            k = k.index_select(1, self.head_to_group)
            v = v.index_select(1, self.head_to_group)
        return _attend(self, q, k, v, attention_mask, B, T)

class _OptimalMixin:

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                **kwargs):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, Gk, Gv, d, c_k, c_v = self.n_heads, self.n_kv_heads, self.n_kv_heads_v, self.head_dim, self.c_k, self.c_v
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        bx = self.bk_proj(hidden_states).view(B, T, Gk, c_k).transpose(1, 2)
        bvx = self.bv_proj(hidden_states).view(B, T, Gv, c_v).transpose(1, 2)
        if past_key_values is not None:
            bx, bvx = past_key_values.update(bx, bvx, self.layer_idx)
        S = bx.shape[2]
        k = torch.einsum("bhtc,hdc->bhtd", bx.index_select(1, self.head_to_group_k), self.Ck)
        v = torch.einsum("bhtc,hdc->bhtd", bvx.index_select(1, self.head_to_group_v), self.Cv)
        cos, sin = position_embeddings
        cos_s, sin_s = _cos_sin_all(self, hidden_states, position_embeddings, S, T)
        q, k = _rope(q, cos, sin), _rope(k, cos_s, sin_s)
        return _attend(self, q, k, v, attention_mask, B, T)

class _DecoupledMixin:

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                **kwargs):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, d, c_KV, d_k_R = self.n_heads, self.head_dim, self.c_KV, self.d_k_R
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        bkv = self.bkv_proj(hidden_states).view(B, T, 1, c_KV).transpose(1, 2)
        bkr = self.bkr_proj(hidden_states).view(B, T, 1, d_k_R).transpose(1, 2)
        if past_key_values is not None:
            bkr, bkv = past_key_values.update(bkr, bkv, self.layer_idx)
        S = bkv.shape[2]
        bkv_h, bkr_h = bkv.expand(-1, H, -1, -1), bkr.expand(-1, H, -1, -1)
        k_rot = torch.einsum("bhtc,hdc->bhtd", bkr_h, self.Ckr)
        k_free = torch.einsum("bhtc,hdc->bhtd", bkv_h, self.Ck)
        k = torch.cat((k_rot, k_free), dim=-1)
        v = torch.einsum("bhtc,hdc->bhtd", bkv_h, self.Cv)
        cos, sin = position_embeddings
        cos_s, sin_s = _cos_sin_all(self, hidden_states, position_embeddings, S, T)
        q, k = _rope(q, cos, sin), _rope(k, cos_s, sin_s)
        return _attend(self, q, k, v, attention_mask, B, T)

def _twin(mixin, host_cls):
    key = (mixin, host_cls)
    if key not in _TWINS:
        eager = getattr(sys.modules[host_cls.__module__], "eager_attention_forward", None)
        if eager is None:
            raise RuntimeError(f"{host_cls.__module__} has no eager_attention_forward")
        name = mixin.__name__.strip("_").replace("Mixin", "") + host_cls.__name__
        cls = type(name, (mixin, host_cls), {"_host_eager": staticmethod(eager), "__module__": __name__})
        globals()[name] = cls
        _TWINS[key] = cls
    return _TWINS[key]

def _host_attn(lay):
    a = lay.self_attn
    if isinstance(a, (_GQAMixin, _OptimalMixin, _DecoupledMixin)):
        raise RuntimeError(f"layer {a.layer_idx} is already restructured ({type(a).__name__})")
    if getattr(a, "qk_layernorm", False):
        raise RuntimeError("qk_layernorm: the key is not linear in x; no construction here applies")
    if getattr(a, "is_causal", True) is not True:
        raise RuntimeError("a non-causal attention host")
    return a

def _check_host(model, what):
    dt_util.assert_cpu(model, what)
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError(f"{what}: attention implementation is not eager; the twins need the eager kernel")
    if not gam_core_llama.is_llama(model):
        raise RuntimeError(f"{what}: not an unconverted LLaMA-layout model (q/k/v/o_proj under self_attn)")

def _finish(attn, model, mixin, H):
    attn.n_heads = H
    attn.num_key_value_groups = 1
    object.__setattr__(attn, "_rotary_ref", _rotary_module(model))
    attn.__class__ = _twin(mixin, type(attn))

def restructure_gqa(model, G, check_tie=True, groups_by_layer=None):
    _check_host(model, "restructure_gqa")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if H % G:
        raise ValueError(f"H={H} not divisible by G={G}")
    default_groups = kvlab.groups_of(H, G)
    if groups_by_layer is not None and len(groups_by_layer) != L:
        raise ValueError(f"groups_by_layer has {len(groups_by_layer)} entries, model has {L} layers")
    worst_tie = 0.0
    for li, lay in enumerate(kvlab.get_layers(model)):
        groups = groups_by_layer[li] if groups_by_layer is not None else default_groups
        attn = _host_attn(lay)
        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        Wq, Wk, Wv, bq, bk, bv = gam_core_llama.split_qkv(lay, H, d_k, dm)
        if check_tie:
            for idx in groups:
                for A in (Wk, bk, Wv, bv):
                    worst_tie = max(worst_tie, float(np.abs(A[idx] - A[idx[0]]).max()))
            if worst_tie != 0.0:
                raise RuntimeError(f"heads within a group are not identical (max |dev| = {worst_tie:.3e}): "
                                   f"restructure_gqa must run AFTER a conversion build at this G")
        heads0 = [g[0] for g in groups]
        attn.q_proj = dt_gqa_arch._linear_from(Wq.reshape(H * d_k, dm), bq.reshape(H * d_k), dm, dtype, device)
        attn.k_proj = dt_gqa_arch._linear_from(np.concatenate([Wk[i] for i in heads0]),
                                               np.concatenate([bk[i] for i in heads0]), dm, dtype, device)
        attn.v_proj = dt_gqa_arch._linear_from(np.concatenate([Wv[i] for i in heads0]),
                                               np.concatenate([bv[i] for i in heads0]), dm, dtype, device)
        attn.n_kv_heads, attn.n_kv_groups = G, H // G
        attn.register_buffer("head_to_group", torch.tensor(gc.head_group(groups, H), dtype=torch.long,
                                                           device=device), persistent=False)
        _finish(attn, model, _GQAMixin, H)
    return dict(L=L, H=H, G=G, d_k=d_k, d_v=d_v, dm=dm, r=r, per_group=H // G,
                groups=default_groups if groups_by_layer is None else groups_by_layer, worst_tie=worst_tie)

def gqa_fingerprint(model):
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        a = lay.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def restructure_optimal(model, model_name, G, G_v=None, check_tie=True, groups_by_layer=None, wide=False, B=None):
    _check_host(model, "restructure_optimal")
    G_v = G if G_v is None else G_v
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if wide and (G_v != G or groups_by_layer is not None):
        raise ValueError("wide=True fixes one group over all heads on both sides")
    if B is not None and not wide:
        raise ValueError(f"a budget B = {B} was passed to the GROUPED optimal restructure; only wide takes one")
    if B is not None and G is not None and B != G * (d_k + d_v):
        raise ValueError(f"budget B = {B} != G*(d_k+d_v) = {G * (d_k + d_v)} at G={G}")
    Bk_all, Bv_all, gbl_k, gbl_v, c_k, c_v = dt_optimal_arch._compute_shared_bases(model, G, G_v, groups_by_layer,
                                                                                   wide=wide, B=B)
    worst_dev = 0.0
    for li, lay in enumerate(kvlab.get_layers(model)):
        groups_k, groups_v = gbl_k[li], gbl_v[li]
        gk_of, gv_of = gc.head_group(groups_k, H), gc.head_group(groups_v, H)
        Qt, Kh, Vh, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        if check_tie:
            for i in range(H):
                Bk, Bv = Bk_all[li][gk_of[i]], Bv_all[li][gv_of[i]]
                worst_dev = max(worst_dev, float(np.abs(Kh[i] - Kh[i] @ Bk.T @ Bk).max()),
                                float(np.abs(Vh[i] - Vh[i] @ Bv.T @ Bv).max()))
            if worst_dev > 1e-6:
                raise RuntimeError(f"layer {li}: a head's read-out is not in its group's B rowspace (max dev "
                                   f"{worst_dev:.3e}); restructure_optimal must run AFTER dt_optimal.build")
        Ck = np.stack([Kh[i] @ Bk_all[li][gk_of[i]].T for i in range(H)])
        Cv = np.stack([Vh[i] @ Bv_all[li][gv_of[i]].T for i in range(H)])
        Wq, bq = gc.deaugment(Qt)
        attn = _host_attn(lay)
        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        Wk_g, bk_g = gc.deaugment(np.stack([Bk_all[li][j] for j in range(len(groups_k))]))
        Wv_g, bv_g = gc.deaugment(np.stack([Bv_all[li][j] for j in range(len(groups_v))]))
        attn.q_proj = dt_gqa_arch._linear_from(Wq.reshape(H * d_k, dm), bq.reshape(H * d_k), dm, dtype, device)
        attn.bk_proj = dt_gqa_arch._linear_from(Wk_g.reshape(len(groups_k) * c_k, dm),
                                                bk_g.reshape(len(groups_k) * c_k), dm, dtype, device)
        attn.bv_proj = dt_gqa_arch._linear_from(Wv_g.reshape(len(groups_v) * c_v, dm),
                                                bv_g.reshape(len(groups_v) * c_v), dm, dtype, device)
        for p in (attn.bk_proj, attn.bv_proj):
            for t in p.parameters():
                t.requires_grad_(False)
        attn.Ck = nn.Parameter(torch.from_numpy(np.ascontiguousarray(Ck)).to(device=device, dtype=dtype))
        attn.Cv = nn.Parameter(torch.from_numpy(np.ascontiguousarray(Cv)).to(device=device, dtype=dtype))
        del attn.k_proj, attn.v_proj
        attn.n_kv_heads, attn.n_kv_groups = len(groups_k), H // len(groups_k)
        attn.n_kv_heads_v, attn.n_kv_groups_v = len(groups_v), H // len(groups_v)
        attn.c_k, attn.c_v = c_k, c_v
        attn.register_buffer("head_to_group_k", torch.tensor(gk_of, dtype=torch.long, device=device),
                             persistent=False)
        attn.register_buffer("head_to_group_v", torch.tensor(gv_of, dtype=torch.long, device=device),
                             persistent=False)
        _finish(attn, model, _OptimalMixin, H)
    return dict(L=L, H=H, G=G, G_v=G_v, d_k=d_k, d_v=d_v, dm=dm, r=r, worst_dev=worst_dev)

def optimal_fingerprint(model):
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        a = lay.self_attn
        for name in ("q_proj", "bk_proj", "bv_proj", "o_proj"):
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
        h.update(a.Ck.detach().cpu().numpy().tobytes())
        h.update(a.Cv.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def restructure_decoupled(model, model_name, G, w=dt_decoupled.W_KV_DEFAULT, d_k_R=dt_decoupled.DKR_DEFAULT,
                          check_tie=True, bases=None, B=None):
    _check_host(model, "restructure_decoupled")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0:
        raise ValueError("decoupled_optwide needs rotary rows to decouple")
    if bases is None:
        Wbar_all, WKR_all, widths = dt_decoupled_arch._compute_decoupled_bases(model, G, w, d_k_R, B=B)
    else:
        Wbar_all, WKR_all, widths = bases
        if not (len(Wbar_all) == len(WKR_all) == len(widths) == L):
            raise ValueError("`bases` must come from the build() call that converted THIS model")
    worst = 0.0
    for li, lay in enumerate(kvlab.get_layers(model)):
        dkr, c_KV = widths[li]
        Wbar, WKR = Wbar_all[li], WKR_all[li]
        Qt, Kh, Vh, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        if check_tie:
            dev = 0.0
            for i in range(H):
                rot, free = Kh[i][:r, :], Kh[i][r:, :]
                dev = max(dev, float(np.abs(rot - rot @ WKR.T @ WKR).max()))
                if free.shape[0]:
                    dev = max(dev, float(np.abs(free - free @ Wbar.T @ Wbar).max()))
                dev = max(dev, float(np.abs(Vh[i] - Vh[i] @ Wbar.T @ Wbar).max()))
            worst = max(worst, dev)
            if dev > 1e-6:
                raise RuntimeError(f"layer {li}: a head's read-out is not in the rowspace it should be (max dev "
                                   f"{dev:.3e}); restructure_decoupled must run AFTER dt_decoupled.build with the same w")
        Ckr = np.stack([Kh[i][:r, :] @ WKR.T for i in range(H)])
        Ck = np.stack([Kh[i][r:, :] @ Wbar.T for i in range(H)])
        Cv = np.stack([Vh[i] @ Wbar.T for i in range(H)])
        Wq, bq = gc.deaugment(Qt)
        Wkv, bkv = gc.deaugment(Wbar[None, ...])
        Wkr, bkr = gc.deaugment(WKR[None, ...])
        attn = _host_attn(lay)
        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        attn.q_proj = dt_gqa_arch._linear_from(Wq.reshape(H * d_k, dm), bq.reshape(H * d_k), dm, dtype, device)
        attn.bkv_proj = dt_gqa_arch._linear_from(Wkv.reshape(c_KV, dm), bkv.reshape(c_KV), dm, dtype, device)
        attn.bkr_proj = dt_gqa_arch._linear_from(Wkr.reshape(dkr, dm), bkr.reshape(dkr), dm, dtype, device)
        for p in (attn.bkv_proj, attn.bkr_proj):
            for t in p.parameters():
                t.requires_grad_(False)
        for name, arr in (("Ckr", Ckr), ("Ck", Ck), ("Cv", Cv)):
            setattr(attn, name, nn.Parameter(torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)))
        del attn.k_proj, attn.v_proj
        attn.c_KV, attn.d_k_R, attn.r, attn.w_kv = c_KV, dkr, r, float(w)
        _finish(attn, model, _DecoupledMixin, H)
    _u = len(set(widths)) == 1
    return dict(L=L, H=H, G=G, d_k=d_k, d_v=d_v, dm=dm, r=r, w=w, worst_dev=worst,
                c_KV=widths[0][1] if _u else [x[1] for x in widths], d_k_R=widths[0][0] if _u else [x[0] for x in widths],
                cache_width=widths[0][0] + widths[0][1])

def decoupled_fingerprint(model):
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        a = lay.self_attn
        for name in ("q_proj", "bkv_proj", "bkr_proj", "o_proj"):
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
        for name in ("Ckr", "Ck", "Cv"):
            h.update(getattr(a, name).detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def decoupled_meta(model):
    attns = [lay.self_attn for lay in kvlab.get_layers(model)]
    dkr, ckv = [int(a.d_k_R) for a in attns], [int(a.c_KV) for a in attns]
    if len({d + c for d, c in zip(dkr, ckv)}) != 1:
        raise AssertionError("the matched-cache total differs between layers")
    if len({float(a.w_kv) for a in attns}) != 1:
        raise AssertionError("w differs between layers")
    u = len(set(dkr)) == 1
    return {"c_KV": ckv[0] if u else ckv, "d_k_R": dkr[0] if u else dkr, "r": int(attns[0].r),
            "w": float(attns[0].w_kv)}
