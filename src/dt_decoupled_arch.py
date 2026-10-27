import hashlib

import numpy as np
import torch
import torch.nn as nn
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import dt_decoupled
import dt_gqa_arch
import gam_core as gc
import kvlab

class DecoupledLatentAttention(GPTNeoXAttention):

    n_heads = None
    c_KV = None
    d_k_R = None
    r = None

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
        H, d, r = self.n_heads, self.head_size, self.r
        c_KV, d_k_R = self.c_KV, self.d_k_R

        q = self.q_proj(hidden_states).view(B_, T, H, d).transpose(1, 2)
        bkv = self.bkv_proj(hidden_states).view(B_, T, 1, c_KV).transpose(1, 2)
        bkr = self.bkr_proj(hidden_states).view(B_, T, 1, d_k_R).transpose(1, 2)

        if layer_past is not None:
            bkr, bkv = layer_past.update(bkr, bkv, self.layer_idx)

        bkv_h = bkv.expand(-1, H, -1, -1)
        bkr_h = bkr.expand(-1, H, -1, -1)

        k_rot = torch.einsum("bhtc,hdc->bhtd", bkr_h, self.Ckr)
        k_free = torch.einsum("bhtc,hdc->bhtd", bkv_h, self.Ck)
        k = torch.cat((k_rot, k_free), dim=-1)
        v = torch.einsum("bhtc,hdc->bhtd", bkv_h, self.Cv)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.dense(attn_output), attn_weights

def _compute_decoupled_bases(model, G, w, d_k_R, B=None):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    widths = dt_decoupled._widths_by_layer(G, H, d_k, d_v, r, d_k_R, L, B=B)
    theta = gc.inv_freq(model)
    wD = dt_decoupled.dt_optimal._offset_weights(dt_decoupled.dt_optimal.T_EVAL)

    Wbar_all, WKR_all = [], []
    for li, lay in enumerate(layers):
        d_k_R_l, c_KV_l = widths[li]
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        A_free, A_V, A_rot = dt_decoupled._metrics(Qt, Wo, H, r, theta, wD)
        Wbar_all.append(gc.tail_svd(
            dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H, r, w), c_KV_l)[1])
        WKR_all.append(gc.tail_svd(dt_decoupled._rot_stack(Kt, A_rot, H, r), d_k_R_l)[1])
    return Wbar_all, WKR_all, widths

def restructure_decoupled(model, model_name, G, w=dt_decoupled.W_KV_DEFAULT,
                          d_k_R=dt_decoupled.DKR_DEFAULT, check_tie=True, bases=None, B=None):
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(
            f"attention implementation is {impl!r}; DecoupledLatentAttention expands both cached "
            "projections itself and only the eager path is correct here")

    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0:
        raise ValueError("decoupled_optwide needs rotary rows to decouple")
    if bases is None:
        Wbar_all, WKR_all, widths = _compute_decoupled_bases(model, G, w, d_k_R, B=B)
    else:
        Wbar_all, WKR_all, widths = bases
        if not (len(Wbar_all) == len(WKR_all) == len(widths) == L):
            raise ValueError(
                f"`bases` covers {len(Wbar_all)}/{len(WKR_all)}/{len(widths)} layers, the model "
                f"has {L}. They must come from the build() call that converted THIS model.")
    layers = kvlab.get_layers(model)
    worst_overall = 0.0

    for li, lay in enumerate(layers):
        d_k_R, c_KV = widths[li]
        Wbar, WKR = Wbar_all[li], WKR_all[li]
        Qt, Kh, Vh, Wo = gc.read_layer(lay, H, d_k, d_v, dm)

        if check_tie:
            worst_dev = 0.0
            for i in range(H):
                rot, free = Kh[i][:r, :], Kh[i][r:, :]
                worst_dev = max(worst_dev, float(np.abs(rot - rot @ WKR.T @ WKR).max()))
                worst_dev = max(worst_dev, float(np.abs(free - free @ Wbar.T @ Wbar).max()))
                worst_dev = max(worst_dev, float(np.abs(Vh[i] - Vh[i] @ Wbar.T @ Wbar).max()))
            worst_overall = max(worst_overall, worst_dev)
            if worst_dev > 1e-6:
                raise RuntimeError(
                    f"layer {li}: a head's read-out is not in the rowspace it should be (max dev "
                    f"{worst_dev:.3e}). restructure_decoupled() must run AFTER "
                    f"dt_decoupled.build(model, G={G}, w={w}, d_k_R={d_k_R}) with the SAME w -- w "
                    f"scales the value block of the joint stack, so a different w picks a different "
                    f"content subspace and this is what that looks like.")

        Ckr = np.stack([Kh[i][:r, :] @ WKR.T for i in range(H)])
        Ck = np.stack([Kh[i][r:, :] @ Wbar.T for i in range(H)])
        Cv = np.stack([Vh[i] @ Wbar.T for i in range(H)])

        Wq, bq = gc.deaugment(Qt)
        Wkv, bkv = gc.deaugment(Wbar[None, ...])
        Wkr, bkr = gc.deaugment(WKR[None, ...])

        attn = lay.attention
        dtype, device = attn.query_key_value.weight.dtype, attn.query_key_value.weight.device

        attn.q_proj = dt_gqa_arch._linear_from(Wq.reshape(H * d_k, dm), bq.reshape(H * d_k), dm,
                                                dtype, device)
        attn.bkv_proj = dt_gqa_arch._linear_from(Wkv.reshape(c_KV, dm), bkv.reshape(c_KV), dm,
                                                  dtype, device)
        attn.bkr_proj = dt_gqa_arch._linear_from(Wkr.reshape(d_k_R, dm), bkr.reshape(d_k_R), dm,
                                                  dtype, device)
        for p in (attn.bkv_proj, attn.bkr_proj):
            for t in p.parameters():
                t.requires_grad_(False)

        for name, arr in (("Ckr", Ckr), ("Ck", Ck), ("Cv", Cv)):
            setattr(attn, name, nn.Parameter(
                torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)))

        del attn.query_key_value
        attn.n_heads, attn.c_KV, attn.d_k_R, attn.r = H, c_KV, d_k_R, r
        attn.w_kv = float(w)
        attn.__class__ = DecoupledLatentAttention

    _uniform = len(set(widths)) == 1
    return dict(L=L, H=H, G=G, d_k=d_k, d_v=d_v, dm=dm, r=r, w=w, worst_dev=worst_overall,
                c_KV=widths[0][1] if _uniform else [x[1] for x in widths],
                d_k_R=widths[0][0] if _uniform else [x[0] for x in widths],
                cache_width=widths[0][0] + widths[0][1])

def decoupled_fingerprint(model):
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        a = lay.attention
        for name in ("q_proj", "bkv_proj", "bkr_proj", "dense"):
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
        h.update(a.Ckr.detach().cpu().numpy().tobytes())
        h.update(a.Ck.detach().cpu().numpy().tobytes())
        h.update(a.Cv.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def decoupled_meta(model):
    attns = [lay.attention for lay in kvlab.get_layers(model)]
    dkr = [int(a.d_k_R) for a in attns]
    ckv = [int(a.c_KV) for a in attns]
    tot = {d + c for d, c in zip(dkr, ckv)}
    if len(tot) != 1:
        raise AssertionError(f"the matched-cache total differs BETWEEN layers ({sorted(tot)}). Every "
                             f"layer must hold G*(d_k+d_v) scalars per token or the arm is not "
                             f"comparable to anything at the same G.")
    uniform = len(set(dkr)) == 1
    ws = {float(a.w_kv) for a in attns}
    if len(ws) != 1:
        raise AssertionError(f"w differs between layers ({sorted(ws)}); it is one exchange rate")
    return {"c_KV": ckv[0] if uniform else ckv, "d_k_R": dkr[0] if uniform else dkr,
            "r": int(attns[0].r), "w": ws.pop()}

def trainable_parameters_decoupled(model):
    frozen = set()
    for lay in kvlab.get_layers(model):
        for p in (lay.attention.bkv_proj, lay.attention.bkr_proj):
            frozen.update(id(t) for t in p.parameters())
    return [t for t in model.parameters() if t.requires_grad and id(t) not in frozen]
