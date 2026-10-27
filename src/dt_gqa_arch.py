import hashlib

import numpy as np
import torch
import torch.nn as nn

from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import kvlab
import gam_core as gc
import dt_util

class GQAAttention(GPTNeoXAttention):

    n_heads = None
    n_kv_heads = None
    n_kv_groups = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        position_embeddings=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        B, T = input_shape[0], input_shape[1]
        H, G, d = self.n_heads, self.n_kv_heads, self.head_size

        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, G, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, G, d).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if layer_past is not None:
            k, v = layer_past.update(k, v, self.layer_idx)

        if G != H:
            k = k.index_select(1, self.head_to_group)
            v = v.index_select(1, self.head_to_group)

        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.dense(attn_output), attn_weights

def _linear_from(W, b, dm, dtype, device):
    out = W.shape[0]
    lin = nn.Linear(dm, out, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(W)).to(dtype))
        lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)).to(dtype))
    return lin.to(device=device, dtype=dtype)

def restructure(model, G, check_tie=True, wrong=False, groups_by_layer=None):
    dt_util.assert_cpu(model, "restructure")
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(
            f"attention implementation is {impl!r}; GQAAttention expands K/V itself and only the "
            "eager path is correct here (the shared dispatch would repeat again if the module were "
            "named num_key_value_groups, and would broadcast-fail otherwise)"
        )

    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if H % G:
        raise ValueError(f"H={H} not divisible by G={G}")
    default_groups = kvlab.groups_of(H, G)
    per_group = H // G
    if groups_by_layer is not None and len(groups_by_layer) != L:
        raise ValueError(f"groups_by_layer has {len(groups_by_layer)} entries, model has {L} layers")

    layers = kvlab.get_layers(model)
    worst_tie = 0.0
    for li, lay in enumerate(layers):
        groups = groups_by_layer[li] if groups_by_layer is not None else default_groups
        attn = lay.attention
        p = attn.query_key_value
        dtype, device = p.weight.dtype, p.weight.device

        Wq, Wk, Wv, bq, bk, bv = kvlab.split_qkv(lay, H, d_k, dm, wrong=wrong)

        if check_tie:
            for idx in groups:
                for A in (Wk, bk, Wv, bv):
                    dev = float(np.abs(A[idx] - A[idx[0]]).max())
                    worst_tie = max(worst_tie, dev)
            if worst_tie != 0.0:
                raise RuntimeError(
                    f"heads within a group are not identical (max |dev| = {worst_tie:.3e}). "
                    "restructure() must run AFTER a conversion build; a nonzero value here means the "
                    "model was not tied at this G, or the grouping conventions disagree."
                )

        heads0 = [g[0] for g in groups]
        Wk_g = np.concatenate([Wk[i] for i in heads0], axis=0)
        Wv_g = np.concatenate([Wv[i] for i in heads0], axis=0)
        bk_g = np.concatenate([bk[i] for i in heads0], axis=0)
        bv_g = np.concatenate([bv[i] for i in heads0], axis=0)

        attn.q_proj = _linear_from(Wq.reshape(H * d_k, dm), bq.reshape(H * d_k), dm, dtype, device)
        attn.k_proj = _linear_from(Wk_g, bk_g, dm, dtype, device)
        attn.v_proj = _linear_from(Wv_g, bv_g, dm, dtype, device)

        del attn.query_key_value

        attn.n_heads = H
        attn.n_kv_heads = G
        attn.n_kv_groups = per_group
        attn.register_buffer(
            "head_to_group",
            torch.tensor(gc.head_group(groups, H), dtype=torch.long, device=device),
            persistent=False)
        attn.__class__ = GQAAttention

    return dict(L=L, H=H, G=G, d_k=d_k, d_v=d_v, dm=dm, r=r, per_group=per_group,
                groups=default_groups if groups_by_layer is None else groups_by_layer,
                worst_tie=worst_tie)

def to_fused(model):
    dt_util.assert_cpu(model, "to_fused")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    layers = kvlab.get_layers(model)

    for lay in layers:
        attn = lay.attention
        if not isinstance(attn, GQAAttention):
            continue
        G = attn.n_kv_heads
        g_of = attn.head_to_group.detach().cpu().numpy()

        Wq = attn.q_proj.weight.detach().cpu().double().numpy().reshape(H, d_k, dm)
        bq = attn.q_proj.bias.detach().cpu().double().numpy().reshape(H, d_k)
        Wk_g = attn.k_proj.weight.detach().cpu().double().numpy().reshape(G, d_k, dm)
        bk_g = attn.k_proj.bias.detach().cpu().double().numpy().reshape(G, d_k)
        Wv_g = attn.v_proj.weight.detach().cpu().double().numpy().reshape(G, d_v, dm)
        bv_g = attn.v_proj.bias.detach().cpu().double().numpy().reshape(G, d_v)

        Wk = np.stack([Wk_g[g_of[i]] for i in range(H)])
        bk = np.stack([bk_g[g_of[i]] for i in range(H)])
        Wv = np.stack([Wv_g[g_of[i]] for i in range(H)])
        bv = np.stack([bv_g[g_of[i]] for i in range(H)])

        dtype = attn.q_proj.weight.dtype
        device = attn.q_proj.weight.device
        bias = attn.dense.bias is not None
        attn.query_key_value = nn.Linear(dm, 3 * dm, bias=bias).to(device=device, dtype=dtype)

        del attn.q_proj, attn.k_proj, attn.v_proj
        for a in ("n_heads", "n_kv_heads", "n_kv_groups", "head_to_group"):
            if hasattr(attn, a):
                delattr(attn, a)
        attn.__class__ = GPTNeoXAttention

        kvlab.merge_qkv(lay, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm)

    return model

def layer_prefix(model):
    layers = kvlab.get_layers(model)
    for name, mod in model.named_modules():
        if mod is layers:
            return name
    raise RuntimeError("could not locate the layer container in named_modules()")

def gqa_fingerprint(model):
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        a = lay.attention
        for name in ("q_proj", "k_proj", "v_proj", "dense"):
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def trainable_parameters(model):
    for p in model.parameters():
        p.requires_grad_(True)
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    return names, params
