import hashlib

import numpy as np
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import dt_mla_arch as MA
import gam_core as gc
import gam_core_llama
import kvlab

def get_attn(layer):
    return layer.self_attn

def is_converted(model):
    try:
        a = get_attn(kvlab.get_layers(model)[0])
    except (AttributeError, IndexError):
        return False
    return type(a).__name__ == "LlamaMLAAttention"

def _geometry_converted(model):
    layers = kvlab.get_layers(model)
    c = model.config
    L, H, dm = len(layers), int(c.num_attention_heads), int(c.hidden_size)
    emb = gam_core_llama._rotary_emb(model)
    r = 2 * int(emb.inv_freq.shape[0])
    sc = float(getattr(emb, "attention_scaling", 1.0))
    if sc != 1.0:
        raise RuntimeError(
            f"LlamaRotaryEmbedding.attention_scaling = {sc} != 1: cos/sin carry a factor the slot "
            f"gather does not model, so every rotary slot would be scaled by it.")
    seen = {(int(get_attn(l).n_heads), int(get_attn(l).head_size)) for l in layers}
    if len(seen) != 1:
        raise RuntimeError(f"converted layers disagree on (n_heads, head_size): {sorted(seen)}")
    H_mod, dh = seen.pop()
    if H_mod != H:
        raise RuntimeError(f"converted attention says n_heads = {H_mod}, the config says {H}")
    if dh * H != dm:
        raise RuntimeError(f"head width {dh} x H {H} != hidden size {dm}")
    if r > dh:
        raise RuntimeError(f"rotary width {r} exceeds the head width {dh}")
    return L, H, dh, dh, dm, r

def geometry(model):
    if is_converted(model):
        return _geometry_converted(model)
    if not gam_core_llama.is_llama(model):
        raise RuntimeError(f"{type(model).__name__} is not a LLaMA-family model; use dt_mla_arch")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    emb = gam_core_llama._rotary_emb(model)
    r_inv = 2 * int(emb.inv_freq.shape[0])
    if r != r_inv:
        raise RuntimeError(f"rotary width disagrees: geometry says {r}, 2*len(inv_freq) says {r_inv}")
    sc = float(getattr(emb, "attention_scaling", 1.0))
    if sc != 1.0:
        raise RuntimeError(
            f"LlamaRotaryEmbedding.attention_scaling = {sc} != 1: cos/sin carry a factor the slot "
            f"gather does not model, so every rotary slot would be scaled by it. A scaled rope "
            f"type needs its own treatment, not silent absorption.")
    return L, H, d_k, d_v, dm, r

class LlamaMLAAttention(LlamaAttention):

    def _mla_qkv(self, hidden_states, position_embeddings, past_key_values=None):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, w, dh = self.n_heads, self.w, self.head_size
        cos, sin = position_embeddings
        cos_s = cos.index_select(-1, self.slot_plane)
        sin_s = sin.index_select(-1, self.slot_plane)
        c = self.dkv(hidden_states).unsqueeze(1)
        kr = self.kr(hidden_states).unsqueeze(1)
        if self.rotary_mode == "shared":
            qr = self.q_rot(hidden_states).view(B, T, H, w).transpose(1, 2)
        elif self.rotary_mode == "per_head":
            qr = self.q_rot(hidden_states).unsqueeze(1)
        else:
            raise RuntimeError(f"LlamaMLAAttention: unknown rotary_mode {self.rotary_mode!r}")
        qr, kr = apply_rotary_pos_emb(qr, kr, cos_s, sin_s)
        if past_key_values is not None:
            kr, c = past_key_values.update(kr, c, self.layer_idx)
        S = c.shape[2]
        kc = nn.functional.linear(c, self.uk, self.uk_bias)
        v = nn.functional.linear(c, self.uv, self.uv_bias).view(B, S, H, dh).transpose(1, 2)
        qc = self.q_content(hidden_states)
        if self.rotary_mode == "shared":
            d0 = self.dC0
            q = torch.cat((qc.view(B, T, H, d0).transpose(1, 2), qr), dim=-1)
            k = torch.cat((kc.view(B, S, H, d0).transpose(1, 2), kr.expand(-1, H, -1, -1)), dim=-1)
        else:
            q = torch.cat((qc, qr[:, 0]), dim=-1).index_select(-1, self.pack_idx)
            q = q.view(B, T, H, dh).transpose(1, 2)
            k = torch.cat((kc[:, 0], kr[:, 0]), dim=-1).index_select(-1, self.pack_idx)
            k = k.view(B, S, H, dh).transpose(1, 2)
        return q, k, v

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        q, k, v = self._mla_qkv(hidden_states, position_embeddings, past_key_values)
        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask, scaling=self.mla_scale,
            dropout=0.0 if not self.training else self.attention_dropout)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.dense(attn_output), attn_weights

def teacher_attn_digest(model):
    if not gam_core_llama.is_llama(model):
        raise RuntimeError("teacher_attn_digest(llama) on a non-LLaMA model")
    h = hashlib.sha256()
    for li, lay in enumerate(kvlab.get_layers(model)):
        a = get_attn(lay)
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(a, nm):
                raise RuntimeError(f"layer {li} has no {nm}: the model is already restructured")
            lin = getattr(a, nm)
            MA._tensor_bytes(h, f"{li}.{nm}.weight", lin.weight)
            if lin.bias is not None:
                MA._tensor_bytes(h, f"{li}.{nm}.bias", lin.bias)
    return h.hexdigest()

def _dense_digest(model):
    h = hashlib.sha256()
    for li, lay in enumerate(kvlab.get_layers(model)):
        a = get_attn(lay)
        d = a.dense if hasattr(a, "dense") else a.o_proj
        MA._tensor_bytes(h, f"{li}.dense.weight", d.weight)
        if d.bias is not None:
            MA._tensor_bytes(h, f"{li}.dense.bias", d.bias)
    return h.hexdigest()

def install(model, spec):
    L, H, d_k, d_v, dm, r = geometry(model)
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention implementation is {impl!r}; LlamaMLAAttention calls "
                           f"eager_attention_forward directly and only the eager path is correct")
    if d_k != d_v:
        raise ValueError(f"d_k={d_k} != d_v={d_v}; the MLA value path assumes one head width")
    per_layer = MA._mla_check_spec(spec, L, H, d_k, r)
    w, d_c, mode = spec["w"], spec["d_c"], spec["rotary_mode"]
    bias_set, scale = tuple(spec["bias_set"]), float(spec["mla_scale"])
    for li, lay in enumerate(kvlab.get_layers(model)):
        dC, sp, sh = per_layer[li]
        D = sum(dC)
        attn = get_attn(lay)
        if int(getattr(attn, "num_key_value_groups", 1)) != 1:
            raise RuntimeError(
                f"layer {li}: num_key_value_groups = {attn.num_key_value_groups} != 1. LLaMA's "
                f"eager_attention_forward would repeat_kv the MLA key, which is one cache head "
                f"shared by every query head already -- a GQA host needs its own path.")
        p = attn.q_proj
        dtype, device, rg = p.weight.dtype, p.weight.device, p.weight.requires_grad
        attn.dkv = MA._mla_linear(dm, d_c, "dkv" in bias_set, dtype, device, rg)
        attn.kr = MA._mla_linear(dm, w, True, dtype, device, rg)
        attn.q_content = MA._mla_linear(dm, D, True, dtype, device, rg)
        attn.q_rot = MA._mla_linear(dm, H * w if mode == "shared" else w, True, dtype, device, rg)
        attn.uk = nn.Parameter(torch.empty(D, d_c, dtype=dtype, device=device), requires_grad=rg)
        if "uk" in bias_set:
            attn.uk_bias = nn.Parameter(torch.empty(D, dtype=dtype, device=device), requires_grad=rg)
        else:
            attn.register_parameter("uk_bias", None)
        attn.uv = nn.Parameter(torch.empty(H * d_v, d_c, dtype=dtype, device=device), requires_grad=rg)
        if "uv" in bias_set:
            attn.uv_bias = nn.Parameter(torch.empty(H * d_v, dtype=dtype, device=device), requires_grad=rg)
        else:
            attn.register_parameter("uv_bias", None)
        attn.register_buffer("slot_plane", torch.tensor(sp, dtype=torch.long, device=device), persistent=True)
        attn.register_buffer("slot_head", torch.tensor(sh, dtype=torch.long, device=device), persistent=True)
        attn.register_buffer("dC", torch.tensor(dC, dtype=torch.long, device=device), persistent=True)
        if mode == "per_head":
            attn.register_buffer("pack_idx", torch.tensor(MA.pack_index(dC, sh, d_k),
                                                          dtype=torch.long, device=device),
                                 persistent=True)
        attn.register_buffer("mla_scale", torch.tensor(scale, dtype=dtype, device=device), persistent=True)
        attn.dense = attn.o_proj
        del attn.o_proj
        del attn.q_proj, attn.k_proj, attn.v_proj
        attn.head_size = d_v
        attn.n_heads, attn.w, attn.d_c, attn.rotary_mode = H, w, d_c, mode
        attn.dC0 = dC[0] if mode == "shared" else None
        attn.bias_set, attn.mla_scale_value = bias_set, scale
        attn.__class__ = LlamaMLAAttention

class _geometry_for_validate:

    def __enter__(self):
        self.old = MA.geometry
        if self.old is geometry:
            raise RuntimeError("dt_mla_arch.geometry is already swapped; nested use is a bug")
        MA.geometry = geometry
        return self

    def __exit__(self, *exc):
        MA.geometry = self.old
        return False

def validate_weights(weights, model, meta):
    with _geometry_for_validate():
        return MA.validate_weights(weights, model, meta)

def restructure(model, weights, meta):
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention implementation is {impl!r}; MLA installs on eager only")
    if teacher_attn_digest(model) != meta["teacher_attn_sha256"]:
        raise RuntimeError("teacher attention digest changed between build_weights and restructure: "
                           "the initialiser MUTATED the model, or a different model was passed")
    report = validate_weights(weights, model, meta)
    L, H, d_k, d_v, dm, r = geometry(model)
    dense0 = _dense_digest(model)
    w, d_c, mode = weights["w"], weights["d_c"], weights["rotary_mode"]
    install(model, {"w": w, "d_c": d_c, "rotary_mode": mode,
                    "bias_set": list(weights["bias_set"]), "mla_scale": float(weights["mla_scale"]),
                    "layers": [{k: [int(x) for x in lw[k]] for k in MA._INT_KEYS}
                               for lw in weights["layers"]]})
    worst = 0.0
    for li, lay in enumerate(kvlab.get_layers(model)):
        a, lw = get_attn(lay), weights["layers"][li]
        pairs = [(a.dkv.weight, lw["W_DKV"][:, :dm]), (a.kr.weight, lw["W_KR"][:, :dm]),
                 (a.kr.bias, lw["W_KR"][:, dm]), (a.q_content.weight, lw["W_UQ"][:, :dm]),
                 (a.q_content.bias, lw["W_UQ"][:, dm]), (a.q_rot.weight, lw["W_QR"][:, :dm]),
                 (a.q_rot.bias, lw["W_QR"][:, dm]), (a.uk, lw["W_UK"][:, :d_c]),
                 (a.uv, lw["W_UV"][:, :d_c])]
        if a.dkv.bias is not None:
            pairs.append((a.dkv.bias, lw["W_DKV"][:, dm]))
        if a.uk_bias is not None:
            pairs.append((a.uk_bias, lw["W_UK"][:, d_c]))
        if a.uv_bias is not None:
            pairs.append((a.uv_bias, lw["W_UV"][:, d_c]))
        with torch.no_grad():
            for prm, arr in pairs:
                src = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=prm.dtype, device=prm.device)
                if tuple(src.shape) != tuple(prm.shape):
                    raise RuntimeError(f"layer {li}: weight shape {tuple(src.shape)} != parameter "
                                       f"{tuple(prm.shape)}")
                prm.copy_(src)
                back = prm.detach().double().numpy()
                exp = torch.from_numpy(np.ascontiguousarray(arr)).to(prm.dtype).double().numpy()
                if not np.array_equal(back, exp):
                    raise RuntimeError(f"layer {li}: installed parameter != weights cast to {prm.dtype}")
                worst = max(worst, float(np.abs(back - arr).max()))
    if _dense_digest(model) != dense0:
        raise RuntimeError("restructure changed the output projection -- it must be left untouched")
    record = {"arm": weights["arm"], "version": weights["version"], "G": weights["G"],
              "w": w, "d_c": d_c, "rotary_mode": mode, "bias_set": list(weights["bias_set"]),
              "mla_scale": float(weights["mla_scale"]), "validate": report,
              "max_cast_error": worst, "cache_width": w + d_c,
              "dtype": str(next(model.parameters()).dtype)}
    model.mla_build_record = record
    return record

class llama_compat:

    def __init__(self, model):
        self.model = model
        self.layers = []

    def __enter__(self):
        L, H, d_k, d_v, dm, r = geometry(self.model)
        for lay in kvlab.get_layers(self.model):
            if hasattr(lay, "attention"):
                raise RuntimeError("layer already has `attention`; nested llama_compat is a bug")
            a = get_attn(lay)
            prev = a.__dict__.get("rotary_ndims", _NO_ATTR)
            if prev is not _NO_ATTR and int(prev) != int(r):
                raise RuntimeError(f"the host's rotary_ndims {prev} != r = {r} measured from inv_freq")
            object.__setattr__(lay, "attention", a)
            a.rotary_ndims = int(r)
            self.layers.append((lay, a, prev))
        return self

    def __exit__(self, *exc):
        for lay, a, prev in self.layers:
            object.__delattr__(lay, "attention")
            if prev is _NO_ATTR:
                del a.rotary_ndims
            else:
                a.rotary_ndims = prev
        self.layers = []
        return False

_NO_ATTR = object()

def _not_ported(name):
    def f(*a, **k):
        raise NotImplementedError(
            f"{name} is deliberately NOT ported to the LLaMA host. See the module docstring.")
    return f

STALE_LLAMA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

def skeleton(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MA.MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MA.MLA_SCHEMA!r}")
    G = meta.get("G")
    if not isinstance(G, int) or isinstance(G, bool):
        raise TypeError(f"meta G must be an int, got {G!r} ({type(G).__name__})")
    w, d_c = meta["w"], meta["d_c"]
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (w, d_c)):
        raise TypeError(f"meta w/d_c must be ints, got {w!r}/{d_c!r}")
    if w + d_c != G * (d_k + d_v):
        raise ValueError(f"cache identity BROKEN: w + d_c = {w} + {d_c} = {w + d_c} != "
                         f"B = G*(d_k+d_v) = {G * (d_k + d_v)}. Refusing to load a branch measured "
                         f"against a different budget.")
    spec = {k: meta[k] for k in ("w", "d_c", "rotary_mode", "bias_set", "mla_scale")}
    spec["layers"] = meta["mla_layers"]
    install(model, spec)
    return dict(L=L, H=H, G=G, w=w, d_c=d_c, rotary_mode=meta["rotary_mode"], d_k=d_k, dm=dm,
                cache_width=w + d_c)

def skeleton_budget(model, meta):
    if "B" not in meta:
        return skeleton(model, meta)
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MA.MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MA.MLA_SCHEMA!r}")
    G, B = meta.get("G"), meta["B"]
    if not isinstance(B, int) or isinstance(B, bool):
        raise TypeError(f"meta B must be an int, got {B!r} ({type(B).__name__})")
    if G is not None and (not isinstance(G, int) or isinstance(G, bool) or B != G * (d_k + d_v)):
        raise ValueError(f"meta G = {G!r} is not an int with B = G*(d_k+d_v) (B = {B})")
    w, d_c = meta["w"], meta["d_c"]
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (w, d_c)):
        raise TypeError(f"meta w/d_c must be ints, got {w!r}/{d_c!r}")
    if w + d_c != B:
        raise ValueError(f"cache identity BROKEN: w + d_c = {w} + {d_c} = {w + d_c} != the budget B = {B}. "
                         f"Refusing to load a branch measured against a different budget.")
    spec = {k: meta[k] for k in ("w", "d_c", "rotary_mode", "bias_set", "mla_scale")}
    spec["layers"] = meta["mla_layers"]
    install(model, spec)
    return dict(L=L, H=H, G=G, B=B, w=w, d_c=d_c, rotary_mode=meta["rotary_mode"], d_k=d_k, dm=dm,
                cache_width=w + d_c)

skeleton_mla = skeleton_budget

def restructure_mla(model, meta, weights):
    for k in MA.EXPECT_KEYS:
        if k not in meta:
            raise KeyError(f"restructure_mla: expectation meta lacks {k!r}")
    devs = {p.device.type for p in model.parameters()}
    if devs != {"cpu"}:
        raise RuntimeError(f"restructure_mla runs on CPU (the probe and fingerprints are CPU-side); "
                           f"model is on {sorted(devs)}")
    rec = restructure(model, weights, meta)
    mm = mla_meta(model)
    check_alloc(model, mm)
    record = {"arm": rec["arm"], "version": rec["version"], "G": rec["G"], "w": rec["w"], "d_c": rec["d_c"],
              "rotary_mode": rec["rotary_mode"], "bias_set": rec["bias_set"], "mla_scale": rec["mla_scale"],
              "provenance": dict(weights["provenance"]), "validate": rec["validate"],
              "max_cast_error": rec["max_cast_error"], "alloc_sha256": mm["alloc_sha256"],
              "attn_params_per_layer": mm["attn_params_per_layer"], "cache_width": rec["cache_width"],
              "dtype": rec["dtype"]}
    model.mla_build_record = record
    return record

def mla_meta(model):
    L, H, d_k, d_v, dm, r = geometry(model)
    attns = [get_attn(lay) for lay in kvlab.get_layers(model)]
    for li, a in enumerate(attns):
        if type(a).__name__ != "LlamaMLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not LlamaMLAAttention")
    a0 = attns[0]
    one = {(a.w, a.d_c, a.rotary_mode, a.bias_set, a.mla_scale_value) for a in attns}
    if len(one) != 1:
        raise AssertionError(f"layers disagree on (w, d_c, mode, bias_set, scale): "
                             f"{sorted(map(str, one))}")
    meta = {"mla_schema": MA.MLA_SCHEMA, "rotary_mode": a0.rotary_mode, "w": int(a0.w),
            "d_c": int(a0.d_c), "mla_scale": float(a0.mla_scale_value),
            "bias_set": list(a0.bias_set),
            "mla_layers": [{k: [int(x) for x in a.get_buffer(k).detach().cpu().tolist()]
                            for k in ("dC", "slot_plane", "slot_head")} for a in attns]}
    meta["alloc_sha256"] = MA._mla_alloc_sha(meta)
    counts = [MA.attn_param_count(H, d_k, dm, meta["w"], meta["d_c"], sum(lay["dC"]),
                                  meta["rotary_mode"], meta["bias_set"])
              for lay in meta["mla_layers"]]
    meta["attn_params_by_layer"] = counts
    meta["attn_params_per_layer"] = counts[0] if len(set(counts)) == 1 else None
    return meta

def mla_fingerprint(model):
    h = hashlib.sha256()
    for li, lay in enumerate(kvlab.get_layers(model)):
        a = get_attn(lay)
        if type(a).__name__ != "LlamaMLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not LlamaMLAAttention")
        h.update(f"{li}|{a.rotary_mode}|{a.w}|{a.d_c}|{a.dC0}|{a.mla_scale_value!r}|"
                 f"{','.join(a.bias_set)}|".encode("utf-8"))
        h.update("|".join(sorted(n for n, _ in a.named_parameters())).encode("utf-8"))
        for kind, names in (("p", MA._mla_param_names(a)), ("b", MA._mla_buffer_names(a))):
            for nm in names:
                t = a.get_parameter(nm) if kind == "p" else a.get_buffer(nm)
                arr = t.detach().cpu().contiguous().numpy()
                h.update(f"{kind}:{nm}|{tuple(arr.shape)}|{arr.dtype}|".encode("utf-8"))
                h.update(arr.tobytes())
    return h.hexdigest()[:16]

def check_alloc(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MA.MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MA.MLA_SCHEMA!r}")
    sha = MA._mla_alloc_sha(meta)
    if sha != meta.get("alloc_sha256"):
        raise RuntimeError(f"alloc_sha256 mismatch: the meta's allocation hashes to {sha[:16]}, the "
                           f"meta records {str(meta.get('alloc_sha256'))[:16]} -- lists were edited")
    layers = kvlab.get_layers(model)
    if len(meta["mla_layers"]) != len(layers):
        raise RuntimeError(f"meta has {len(meta['mla_layers'])} allocations, model has "
                           f"{len(layers)} layers")
    w, d_c, mode = int(meta["w"]), int(meta["d_c"]), meta["rotary_mode"]
    bias_set, scale = tuple(meta["bias_set"]), float(meta["mla_scale"])
    counts = meta.get("attn_params_by_layer")
    if counts is None or len(counts) != len(layers):
        raise RuntimeError("meta attn_params_by_layer is missing or has the wrong length")
    for li, lay in enumerate(layers):
        a = get_attn(lay)
        if type(a).__name__ != "LlamaMLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not LlamaMLAAttention")
        for stale in STALE_LLAMA_MODULES:
            if stale in a._modules:
                raise RuntimeError(f"layer {li} still carries {stale}; install() removes it, so this "
                                   f"layer was never converted or the skeleton ran on a live model")
        got = (a.w, a.d_c, a.rotary_mode, a.bias_set, a.n_heads, a.mla_scale_value)
        if got != (w, d_c, mode, bias_set, H, scale):
            raise RuntimeError(f"layer {li}: module attrs {got} != meta "
                               f"{(w, d_c, mode, bias_set, H, scale)}")
        want = meta["mla_layers"][li]
        for nm in ("slot_plane", "slot_head", "dC"):
            b = a.get_buffer(nm)
            if b.dtype != torch.long or b.dim() != 1 or nm in a._non_persistent_buffers_set:
                raise RuntimeError(f"layer {li}: buffer {nm} must be a persistent 1-D long tensor")
            have = [int(x) for x in b.detach().cpu().tolist()]
            ref = [int(x) for x in want[nm]]
            if have != ref:
                bad = next((i for i, (x, y) in enumerate(zip(have, ref)) if x != y),
                           min(len(have), len(ref)))
                raise RuntimeError(f"layer {li}: buffer {nm} disagrees with the sidecar meta at "
                                   f"index {bad} (buffer {have[bad:bad + 4]}, meta {ref[bad:bad + 4]}). "
                                   f"The state dict and the meta describe different allocations.")
        dC = [int(x) for x in want["dC"]]
        if mode == "shared":
            if "pack_idx" in a._buffers or a.dC0 != dC[0]:
                raise RuntimeError(f"layer {li}: shared mode carries no pack_idx and dC0 == dC[0]")
        else:
            pi = a.get_buffer("pack_idx")
            if a.dC0 is not None or pi.dtype != torch.long \
                    or "pack_idx" in a._non_persistent_buffers_set:
                raise RuntimeError(f"layer {li}: per_head needs a persistent long pack_idx, dC0 None")
            if pi.detach().cpu().tolist() != MA.pack_index(dC, want["slot_head"], d_k):
                raise RuntimeError(f"layer {li}: pack_idx is not the permutation dC/slot_head imply")
        s = a.get_buffer("mla_scale")
        if s.dim() != 0 or not s.is_floating_point() \
                or "mla_scale" in a._non_persistent_buffers_set:
            raise RuntimeError(f"layer {li}: mla_scale must be a persistent 0-dim float buffer")
        if float(s.item()) != float(torch.tensor(scale, dtype=s.dtype).item()):
            raise RuntimeError(f"layer {li}: mla_scale buffer {float(s.item())!r} != meta {scale!r} "
                               f"cast to {s.dtype}")
        names = sorted(n for n, _ in a.named_parameters())
        if names != sorted(MA._mla_param_names(a)):
            raise RuntimeError(f"layer {li}: parameters {names} != the set bias_set {bias_set} "
                               f"implies {sorted(MA._mla_param_names(a))}")
        D = sum(dC)
        shapes = {"dkv.weight": (d_c, dm), "dkv.bias": (d_c,), "kr.weight": (w, dm), "kr.bias": (w,),
                  "q_content.weight": (D, dm), "q_content.bias": (D,),
                  "q_rot.weight": (H * w if mode == "shared" else w, dm),
                  "q_rot.bias": (H * w if mode == "shared" else w,),
                  "uk": (D, d_c), "uk_bias": (D,), "uv": (H * d_v, d_c), "uv_bias": (H * d_v,)}
        for n, t in a.named_parameters():
            if n in shapes and tuple(t.shape) != shapes[n]:
                raise RuntimeError(f"layer {li}: {n} has shape {tuple(t.shape)}, "
                                   f"expected {shapes[n]}")
        n_count = MA.attn_param_count(H, d_k, dm, w, d_c, D, mode, bias_set)
        n_live = sum(t.numel() for n, t in a.named_parameters() if not n.startswith("dense."))
        if not (n_count == n_live == int(counts[li])):
            raise RuntimeError(f"layer {li}: attention parameter count formula {n_count}, live "
                               f"{n_live}, meta {counts[li]} disagree")
    return True

check_probe = MA.check_probe
MLA_PROBE_TOL = MA.MLA_PROBE_TOL

def mla_probe(model):
    v = min(int(MA.MLA_PROBE_VOCAB), int(model.config.vocab_size))
    if v == MA.MLA_PROBE_VOCAB:
        return MA.mla_probe(model)
    rng = np.random.default_rng(MA.MLA_PROBE_SEED)
    half = rng.integers(0, v, size=(2, 128))
    ids = np.concatenate([half, half], axis=1).astype(np.int64)
    pos = [int(x) for x in np.linspace(8, 254, 12).round()]
    voc = sorted(int(x) for x in rng.choice(v, size=32, replace=False))
    out = MA._mla_probe_eval(model, ids, pos, voc)
    return {"seed": MA.MLA_PROBE_SEED, "ids": ids.tolist(), "positions": pos, "vocab": voc,
            "tol": MA.MLA_PROBE_TOL, "dtype": str(next(model.parameters()).dtype), **out}
