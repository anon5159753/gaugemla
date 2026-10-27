import hashlib

import numpy as np
import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import dt_optimal
import dt_gqa_arch
import dt_util
import gam_core as gc
import kvlab

class OptimalGQAAttention(GPTNeoXAttention):

    n_heads = None
    n_kv_heads = None
    n_kv_groups = None
    n_kv_heads_v = None
    n_kv_groups_v = None
    c_k = None
    c_v = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        position_embeddings=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        B_, T = input_shape[0], input_shape[1]
        H, Gk, Gv, d = self.n_heads, self.n_kv_heads, self.n_kv_heads_v, self.head_size

        c_k, c_v = self.c_k, self.c_v
        q = self.q_proj(hidden_states).view(B_, T, H, d).transpose(1, 2)
        bx = self.bk_proj(hidden_states).view(B_, T, Gk, c_k).transpose(1, 2)
        bvx = self.bv_proj(hidden_states).view(B_, T, Gv, c_v).transpose(1, 2)

        if layer_past is not None:
            bx, bvx = layer_past.update(bx, bvx, self.layer_idx)

        bx_h = bx.index_select(1, self.head_to_group_k)
        bvx_h = bvx.index_select(1, self.head_to_group_v)
        k = torch.einsum("bhtc,hdc->bhtd", bx_h, self.Ck)
        v = torch.einsum("bhtc,hdc->bhtd", bvx_h, self.Cv)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.dense(attn_output), attn_weights

def _compute_shared_bases(model, G, G_v, groups_by_layer=None, wide=False, B=None):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if wide and B is not None:
        c_k, c_v = B * d_k // (d_k + d_v), B * d_v // (d_k + d_v)
    else:
        c_k, c_v = (G * d_k, G * d_v) if wide else (d_k, d_v)
    if wide:
        default_k = default_v = [list(range(H))]
    else:
        default_k, default_v = kvlab.groups_of(H, G), kvlab.groups_of(H, G_v)
    gbl_k = groups_by_layer if groups_by_layer is not None else [default_k] * L
    gbl_v = groups_by_layer if groups_by_layer is not None else [default_v] * L
    theta = gc.inv_freq(model)
    wD = dt_optimal._offset_weights(dt_optimal.T_EVAL)

    Bk_all, Bv_all = [], []
    for li, lay in enumerate(layers):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        AK = np.stack([Qt[i] @ Qt[i].T for i in range(H)])
        AK = np.stack([dt_optimal._rope_avg(AK[i], r, theta, wD) for i in range(H)])
        AV = np.stack([Wo[i].T @ Wo[i] for i in range(H)])
        Bk_all.append([dt_optimal._shared_map(Kt, AK, g, c_k, "full") for g in gbl_k[li]])
        Bv_all.append([dt_optimal._shared_map(Vt, AV, g, c_v, "full") for g in gbl_v[li]])
    return Bk_all, Bv_all, gbl_k, gbl_v, c_k, c_v

def restructure_optimal(model, model_name, G, G_v=None, check_tie=True, groups_by_layer=None,
                        wide=False, B=None):
    dt_util.assert_cpu(model, "restructure_optimal")
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(
            f"attention implementation is {impl!r}; OptimalGQAAttention expands the shared "
            "projection itself and only the eager path is correct here"
        )

    G_v = G if G_v is None else G_v
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if wide and (G_v != G or groups_by_layer is not None):
        raise ValueError("wide=True fixes one group over all heads on both sides, so G_v must equal "
                         "G and groups_by_layer cannot be given -- matching dt_optimal.build")
    if B is not None and not wide:
        raise ValueError(f"a budget B = {B} was passed to the GROUPED optimal restructure; only wide takes one")
    if B is not None and G is not None and B != G * (d_k + d_v):
        raise ValueError(f"budget B = {B} != G*(d_k+d_v) = {G * (d_k + d_v)} at G={G}")
    Bk_all, Bv_all, gbl_k, gbl_v, c_k, c_v = _compute_shared_bases(model, G, G_v, groups_by_layer,
                                                                   wide=wide, B=B)

    layers = kvlab.get_layers(model)
    worst_dev = 0.0
    for li, lay in enumerate(layers):
        groups_k, groups_v = gbl_k[li], gbl_v[li]
        gk_of = gc.head_group(groups_k, H)
        gv_of = gc.head_group(groups_v, H)
        Qt, Kh, Vh, Wo = gc.read_layer(lay, H, d_k, d_v, dm)

        if check_tie:
            for i in range(H):
                Bk = Bk_all[li][gk_of[i]]
                dev = float(np.abs(Kh[i] - Kh[i] @ Bk.T @ Bk).max())
                worst_dev = max(worst_dev, dev)
                Bv = Bv_all[li][gv_of[i]]
                dev = float(np.abs(Vh[i] - Vh[i] @ Bv.T @ Bv).max())
                worst_dev = max(worst_dev, dev)
            if worst_dev > 1e-6:
                raise RuntimeError(
                    f"layer {li}: a head's read-out is not in its group's B rowspace "
                    f"(max dev {worst_dev:.3e}). restructure_optimal() must run AFTER "
                    f"dt_optimal.build(model, G={G}, variant='full', G_v={G_v}), with the SAME "
                    f"groups_by_layer if a non-contiguous grouping was used."
                )

        Ck = np.stack([Kh[i] @ Bk_all[li][gk_of[i]].T for i in range(H)])
        Cv = np.stack([Vh[i] @ Bv_all[li][gv_of[i]].T for i in range(H)])

        Wq, bq = gc.deaugment(Qt)
        attn = lay.attention
        dtype, device = attn.query_key_value.weight.dtype, attn.query_key_value.weight.device

        Bk_stack = np.stack([Bk_all[li][j] for j in range(len(groups_k))])
        Bv_stack = np.stack([Bv_all[li][j] for j in range(len(groups_v))])
        Wk_g, bk_g = gc.deaugment(Bk_stack)
        Wv_g, bv_g = gc.deaugment(Bv_stack)

        attn.q_proj = dt_gqa_arch._linear_from(Wq.reshape(H * d_k, dm), bq.reshape(H * d_k), dm,
                                                dtype, device)
        attn.bk_proj = dt_gqa_arch._linear_from(Wk_g.reshape(len(groups_k) * c_k, dm),
                                                 bk_g.reshape(len(groups_k) * c_k), dm, dtype, device)
        attn.bv_proj = dt_gqa_arch._linear_from(Wv_g.reshape(len(groups_v) * c_v, dm),
                                                 bv_g.reshape(len(groups_v) * c_v), dm, dtype, device)
        for p in (attn.bk_proj, attn.bv_proj):
            for t in p.parameters():
                t.requires_grad_(False)

        attn.Ck = nn.Parameter(torch.from_numpy(np.ascontiguousarray(Ck)).to(device=device, dtype=dtype))
        attn.Cv = nn.Parameter(torch.from_numpy(np.ascontiguousarray(Cv)).to(device=device, dtype=dtype))

        del attn.query_key_value
        attn.n_heads = H
        attn.n_kv_heads, attn.n_kv_groups = len(groups_k), H // len(groups_k)
        attn.n_kv_heads_v, attn.n_kv_groups_v = len(groups_v), H // len(groups_v)
        attn.c_k, attn.c_v = c_k, c_v
        attn.register_buffer("head_to_group_k", torch.tensor(gk_of, dtype=torch.long, device=device),
                              persistent=False)
        attn.register_buffer("head_to_group_v", torch.tensor(gv_of, dtype=torch.long, device=device),
                              persistent=False)
        attn.__class__ = OptimalGQAAttention

    return dict(L=L, H=H, G=G, G_v=G_v, d_k=d_k, d_v=d_v, dm=dm, r=r, worst_dev=worst_dev)

def to_fused_optimal(model):
    dt_util.assert_cpu(model, "to_fused_optimal")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    layers = kvlab.get_layers(model)

    for lay in layers:
        attn = lay.attention
        if not isinstance(attn, OptimalGQAAttention):
            continue
        Gk, Gv = attn.n_kv_heads, attn.n_kv_heads_v
        gk_of = attn.head_to_group_k.detach().cpu().numpy()
        gv_of = attn.head_to_group_v.detach().cpu().numpy()

        c_k = attn.bk_proj.weight.shape[0] // Gk
        c_v = attn.bv_proj.weight.shape[0] // Gv
        Bk = gc.augment(attn.bk_proj.weight.detach().cpu().double().numpy().reshape(Gk, c_k, dm),
                         attn.bk_proj.bias.detach().cpu().double().numpy().reshape(Gk, c_k))
        Bv = gc.augment(attn.bv_proj.weight.detach().cpu().double().numpy().reshape(Gv, c_v, dm),
                         attn.bv_proj.bias.detach().cpu().double().numpy().reshape(Gv, c_v))
        Ck = attn.Ck.detach().cpu().double().numpy()
        Cv = attn.Cv.detach().cpu().double().numpy()

        Kh = np.stack([Ck[i] @ Bk[gk_of[i]] for i in range(H)])
        Vh = np.stack([Cv[i] @ Bv[gv_of[i]] for i in range(H)])

        Wq = attn.q_proj.weight.detach().cpu().double().numpy().reshape(H, d_k, dm)
        bq = attn.q_proj.bias.detach().cpu().double().numpy().reshape(H, d_k)
        Wk, bk = gc.deaugment(Kh)
        Wv, bv = gc.deaugment(Vh)

        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        bias = attn.dense.bias is not None
        attn.query_key_value = nn.Linear(dm, 3 * dm, bias=bias).to(device=device, dtype=dtype)

        del attn.q_proj, attn.bk_proj, attn.bv_proj, attn.Ck, attn.Cv
        for a in ("n_heads", "n_kv_heads", "n_kv_groups", "n_kv_heads_v", "n_kv_groups_v",
                  "c_k", "c_v", "head_to_group_k", "head_to_group_v"):
            if hasattr(attn, a):
                delattr(attn, a)
        attn.__class__ = GPTNeoXAttention

        kvlab.merge_qkv(lay, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm)

    return model

def optimal_fingerprint(model):
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        a = lay.attention
        for name in ("q_proj", "bk_proj", "bv_proj", "dense"):
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
        h.update(a.Ck.detach().cpu().numpy().tobytes())
        h.update(a.Cv.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def trainable_parameters_optimal(model):
    for p in model.parameters():
        p.requires_grad_(True)
    for lay in kvlab.get_layers(model):
        a = lay.attention
        if isinstance(a, OptimalGQAAttention):
            for p in a.bk_proj.parameters():
                p.requires_grad_(False)
            for p in a.bv_proj.parameters():
                p.requires_grad_(False)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    return names, params
