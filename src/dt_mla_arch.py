import hashlib
import json
import math

import numpy as np
import torch
import torch.nn as nn
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import gam_core as gc
import kvlab

def get_layers(model):
    return kvlab.get_layers(model)

def geometry(model):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    r_inv = 2 * int(model.gpt_neox.rotary_emb.inv_freq.shape[0])
    if r != r_inv:
        raise RuntimeError(f"rotary width disagrees: module says {r}, 2*len(inv_freq) says {r_inv}")
    return L, H, d_k, d_v, dm, r

MLA_SCHEMA = "mla/v1"
MLA_BIAS_MEMBERS = ("dkv", "uk", "uv")
MLA_MODES = ("shared", "per_head")
MLA_PROBE_SEED = 20260929
MLA_PROBE_TOL = 5e-2
MLA_PROBE_VOCAB = 50000

class MLAAttention(GPTNeoXAttention):

    def _mla_qkv(self, hidden_states, position_embeddings, layer_past=None):
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
            raise RuntimeError(f"MLAAttention: unknown rotary_mode {self.rotary_mode!r}")
        qr, kr = apply_rotary_pos_emb(qr, kr, cos_s, sin_s)
        if layer_past is not None:
            kr, c = layer_past.update(kr, c, self.layer_idx)
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

    def forward(self, hidden_states, attention_mask=None, layer_past=None,
                position_embeddings=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        q, k, v = self._mla_qkv(hidden_states, position_embeddings, layer_past)
        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask, scaling=self.mla_scale,
            dropout=0.0 if not self.training else self.attention_dropout)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.dense(attn_output), attn_weights

def pack_index(dC, slot_head, d_h):
    dC = [int(x) for x in dC]
    sh = [int(x) for x in slot_head]
    H, D, w = len(dC), sum(dC), len(sh)
    out, off = [], 0
    for i in range(H):
        own = [D + s for s in range(w) if sh[s] == i]
        if dC[i] < 0 or dC[i] + len(own) != d_h:
            raise ValueError(f"pack_index: head {i} has dC={dC[i]} content rows and {len(own)} slots; "
                             f"they must sum to d_h={d_h} (no padding anywhere, §2.2)")
        out.extend(range(off, off + dC[i]))
        out.extend(own)
        off += dC[i]
    if sorted(out) != list(range(H * d_h)):
        raise ValueError(f"pack_index: not a permutation of range({H * d_h}); a slot_head entry lies "
                         f"outside [0, {H}) or D + w != H*d_h (D={D}, w={w})")
    return out

def attn_param_count(H, d_h, dm, w, d_c, D, rotary_mode, bias_set):
    n = dm * d_c + (d_c if "dkv" in bias_set else 0)
    n += (dm + 1) * w
    n += (dm + 1) * D
    n += (dm + 1) * (H * w if rotary_mode == "shared" else w)
    n += D * d_c + (D if "uk" in bias_set else 0)
    n += H * d_h * d_c + (H * d_h if "uv" in bias_set else 0)
    return int(n)

def _mla_linear(dm, out, bias, dtype, device, requires_grad):
    lin = nn.Linear(dm, out, bias=bool(bias)).to(device=device, dtype=dtype)
    for p in lin.parameters():
        p.requires_grad_(requires_grad)
    return lin

def _mla_param_names(attn):
    bs = attn.bias_set
    names = ["dkv.weight"] + (["dkv.bias"] if "dkv" in bs else [])
    names += ["kr.weight", "kr.bias", "q_content.weight", "q_content.bias", "q_rot.weight", "q_rot.bias"]
    names += ["uk"] + (["uk_bias"] if "uk" in bs else [])
    names += ["uv"] + (["uv_bias"] if "uv" in bs else [])
    names += ["dense.weight"] + (["dense.bias"] if attn.dense.bias is not None else [])
    return names

def _mla_buffer_names(attn):
    return (["slot_plane", "slot_head", "dC"] + (["pack_idx"] if attn.rotary_mode == "per_head" else [])
            + ["mla_scale"])

def _mla_alloc_sha(meta):
    body = {"mla_schema": meta["mla_schema"], "rotary_mode": meta["rotary_mode"],
            "w": int(meta["w"]), "d_c": int(meta["d_c"]), "bias_set": list(meta["bias_set"]),
            "mla_scale": repr(float(meta["mla_scale"])),
            "mla_layers": [{k: [int(x) for x in lay[k]] for k in ("dC", "slot_plane", "slot_head")}
                           for lay in meta["mla_layers"]]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def _mla_check_spec(spec, L, H, d_h, r):
    w, d_c, mode = spec["w"], spec["d_c"], spec["rotary_mode"]
    for nm, v in (("w", w), ("d_c", d_c)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"MLA {nm} must be an int, got {v!r} ({type(v).__name__})")
    if w < 2 or w % 2:
        raise ValueError(f"MLA rotary width w={w} must be even and >= 2 (w = 0 is outside every "
                         f"arm's method, D4)")
    if d_c < 1:
        raise ValueError(f"MLA latent width d_c={d_c} must be >= 1")
    if mode not in MLA_MODES:
        raise ValueError(f"rotary_mode {mode!r} not in {MLA_MODES}")
    bs = list(spec["bias_set"])
    if bs != sorted(set(bs)) or not set(bs) <= set(MLA_BIAS_MEMBERS):
        raise ValueError(f"bias_set {bs!r} must be a sorted, duplicate-free subset of {MLA_BIAS_MEMBERS}")
    sc = spec["mla_scale"]
    if not isinstance(sc, float) or not (0.0 < sc < math.inf):
        raise ValueError(f"mla_scale {sc!r} must be a finite positive float")
    layers = spec["layers"]
    if len(layers) != L:
        raise ValueError(f"allocation covers {len(layers)} layers, the model has {L}")
    half, out = w // 2, []
    for li, lay in enumerate(layers):
        sp = [int(x) for x in lay["slot_plane"]]
        sh = [int(x) for x in lay["slot_head"]]
        dC = [int(x) for x in lay["dC"]]
        if len(sp) != w or len(sh) != w or len(dC) != H:
            raise ValueError(f"layer {li}: slot_plane/slot_head must have w={w} entries and dC H={H} "
                             f"(got {len(sp)}/{len(sh)}/{len(dC)})")
        if any(p < 0 or p >= r // 2 for p in sp):
            raise ValueError(f"layer {li}: slot_plane entries must lie in [0, {r // 2})")
        if sp[:half] != sp[half:]:
            raise ValueError(f"layer {li}: slot s and slot s+w/2 must share one plane -- they are ONE "
                             f"complex coordinate (split-half layout, R1)")
        if any(x < 0 for x in dC):
            raise ValueError(f"layer {li}: negative dC")
        if mode == "shared":
            if any(x != -1 for x in sh):
                raise ValueError(f"layer {li}: slot_head must be -1 everywhere in shared mode")
            if len(set(dC)) != 1:
                raise ValueError(f"layer {li}: shared mode needs one content width for every head, "
                                 f"got {sorted(set(dC))}")
            for p in range(r // 2):
                if sp[:half].count(p) > H:
                    raise ValueError(f"layer {li}: plane {p} holds {sp[:half].count(p)} pairs > H={H}")
        else:
            if any(x < 0 or x >= H for x in sh):
                raise ValueError(f"layer {li}: per_head slot_head entries must lie in [0, {H})")
            if sh[:half] != sh[half:]:
                raise ValueError(f"layer {li}: slot s and slot s+w/2 must belong to one head")
            pack_index(dC, sh, d_h)
        out.append((dC, sp, sh))
    return out

def _install_mla(model, spec):
    L, H, d_k, d_v, dm, r = geometry(model)
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention implementation is {impl!r}; MLAAttention calls "
                           f"eager_attention_forward directly and only the eager path is correct")
    if d_k != d_v:
        raise ValueError(f"d_k={d_k} != d_v={d_v}; the MLA value path assumes one head width")
    per_layer = _mla_check_spec(spec, L, H, d_k, r)
    w, d_c, mode = spec["w"], spec["d_c"], spec["rotary_mode"]
    bias_set = tuple(spec["bias_set"])
    scale = float(spec["mla_scale"])
    for li, lay in enumerate(get_layers(model)):
        dC, sp, sh = per_layer[li]
        D = sum(dC)
        attn = lay.attention
        p = attn.query_key_value
        dtype, device, rg = p.weight.dtype, p.weight.device, p.weight.requires_grad
        attn.dkv = _mla_linear(dm, d_c, "dkv" in bias_set, dtype, device, rg)
        attn.kr = _mla_linear(dm, w, True, dtype, device, rg)
        attn.q_content = _mla_linear(dm, D, True, dtype, device, rg)
        attn.q_rot = _mla_linear(dm, H * w if mode == "shared" else w, True, dtype, device, rg)
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
            attn.register_buffer("pack_idx", torch.tensor(pack_index(dC, sh, d_k), dtype=torch.long,
                                                          device=device), persistent=True)
        attn.register_buffer("mla_scale", torch.tensor(scale, dtype=dtype, device=device), persistent=True)
        del attn.query_key_value
        attn.n_heads, attn.w, attn.d_c, attn.rotary_mode = H, w, d_c, mode
        attn.dC0 = dC[0] if mode == "shared" else None
        attn.bias_set, attn.mla_scale_value = bias_set, scale
        attn.__class__ = MLAAttention

def _skeleton_mla(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MLA_SCHEMA!r}")
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
    _install_mla(model, spec)
    return dict(L=L, H=H, G=G, w=w, d_c=d_c, rotary_mode=meta["rotary_mode"], d_k=d_k, dm=dm,
                cache_width=w + d_c)

def _fp_mla(model):
    h = hashlib.sha256()
    for li, lay in enumerate(get_layers(model)):
        a = lay.attention
        if type(a).__name__ != "MLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not MLAAttention")
        h.update(f"{li}|{a.rotary_mode}|{a.w}|{a.d_c}|{a.dC0}|{a.mla_scale_value!r}|"
                 f"{','.join(a.bias_set)}|".encode("utf-8"))
        h.update("|".join(sorted(n for n, _ in a.named_parameters())).encode("utf-8"))
        for kind, names in (("p", _mla_param_names(a)), ("b", _mla_buffer_names(a))):
            for nm in names:
                t = a.get_parameter(nm) if kind == "p" else a.get_buffer(nm)
                arr = t.detach().cpu().contiguous().numpy()
                h.update(f"{kind}:{nm}|{tuple(arr.shape)}|{arr.dtype}|".encode("utf-8"))
                h.update(arr.tobytes())
    return h.hexdigest()[:16]

def check_alloc(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MLA_SCHEMA!r}")
    sha = _mla_alloc_sha(meta)
    if sha != meta.get("alloc_sha256"):
        raise RuntimeError(f"alloc_sha256 mismatch: the meta's allocation hashes to {sha[:16]}, the "
                           f"meta records {str(meta.get('alloc_sha256'))[:16]} -- the lists were edited")
    layers = get_layers(model)
    if len(meta["mla_layers"]) != len(layers):
        raise RuntimeError(f"meta has {len(meta['mla_layers'])} allocations, model has {len(layers)} layers")
    w, d_c, mode = int(meta["w"]), int(meta["d_c"]), meta["rotary_mode"]
    bias_set, scale = tuple(meta["bias_set"]), float(meta["mla_scale"])
    counts = meta.get("attn_params_by_layer")
    if counts is None or len(counts) != len(layers):
        raise RuntimeError("meta attn_params_by_layer is missing or has the wrong length")
    for li, lay in enumerate(layers):
        a = lay.attention
        if type(a).__name__ != "MLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not MLAAttention")
        if "query_key_value" in a._modules:
            raise RuntimeError(f"layer {li} still carries the fused query_key_value")
        got = (a.w, a.d_c, a.rotary_mode, a.bias_set, a.n_heads, a.mla_scale_value)
        if got != (w, d_c, mode, bias_set, H, scale):
            raise RuntimeError(f"layer {li}: module attrs {got} != meta {(w, d_c, mode, bias_set, H, scale)}")
        want = meta["mla_layers"][li]
        for nm in ("slot_plane", "slot_head", "dC"):
            b = a.get_buffer(nm)
            if b.dtype != torch.long or b.dim() != 1 or nm in a._non_persistent_buffers_set:
                raise RuntimeError(f"layer {li}: buffer {nm} must be a persistent 1-D long tensor")
            have, ref = [int(x) for x in b.detach().cpu().tolist()], [int(x) for x in want[nm]]
            if have != ref:
                bad = next((i for i, (x, y) in enumerate(zip(have, ref)) if x != y), min(len(have), len(ref)))
                raise RuntimeError(f"layer {li}: buffer {nm} disagrees with the sidecar meta at index "
                                   f"{bad} (buffer {have[bad:bad + 4]}, meta {ref[bad:bad + 4]}). The "
                                   f"state dict and the meta describe different allocations.")
        dC = [int(x) for x in want["dC"]]
        if mode == "shared":
            if "pack_idx" in a._buffers or a.dC0 != dC[0]:
                raise RuntimeError(f"layer {li}: shared mode carries no pack_idx and dC0 == dC[0]")
        else:
            pi = a.get_buffer("pack_idx")
            if a.dC0 is not None or pi.dtype != torch.long or "pack_idx" in a._non_persistent_buffers_set:
                raise RuntimeError(f"layer {li}: per_head mode needs a persistent long pack_idx and dC0 None")
            if pi.detach().cpu().tolist() != pack_index(dC, want["slot_head"], d_k):
                raise RuntimeError(f"layer {li}: pack_idx is not the permutation dC/slot_head imply")
        s = a.get_buffer("mla_scale")
        if s.dim() != 0 or not s.is_floating_point() or "mla_scale" in a._non_persistent_buffers_set:
            raise RuntimeError(f"layer {li}: mla_scale must be a persistent 0-dim float buffer")
        if float(s.item()) != float(torch.tensor(scale, dtype=s.dtype).item()):
            raise RuntimeError(f"layer {li}: mla_scale buffer {float(s.item())!r} != meta {scale!r} "
                               f"cast to {s.dtype}")
        names = sorted(n for n, _ in a.named_parameters())
        if names != sorted(_mla_param_names(a)):
            raise RuntimeError(f"layer {li}: parameters {names} != the set bias_set {bias_set} implies "
                               f"{sorted(_mla_param_names(a))}")
        D = sum(dC)
        shapes = {"dkv.weight": (d_c, dm), "dkv.bias": (d_c,), "kr.weight": (w, dm), "kr.bias": (w,),
                  "q_content.weight": (D, dm), "q_content.bias": (D,),
                  "q_rot.weight": (H * w if mode == "shared" else w, dm),
                  "q_rot.bias": (H * w if mode == "shared" else w,),
                  "uk": (D, d_c), "uk_bias": (D,), "uv": (H * d_v, d_c), "uv_bias": (H * d_v,)}
        for n, t in a.named_parameters():
            if n in shapes and tuple(t.shape) != shapes[n]:
                raise RuntimeError(f"layer {li}: {n} has shape {tuple(t.shape)}, expected {shapes[n]}")
        n_count = attn_param_count(H, d_k, dm, w, d_c, D, mode, bias_set)
        n_live = sum(t.numel() for n, t in a.named_parameters() if not n.startswith("dense."))
        if not (n_count == n_live == int(counts[li])):
            raise RuntimeError(f"layer {li}: attention parameter count formula {n_count}, live "
                               f"{n_live}, meta {counts[li]} disagree")
    return True

def _mla_probe_inputs():
    rng = np.random.default_rng(MLA_PROBE_SEED)
    half = rng.integers(0, MLA_PROBE_VOCAB, size=(2, 128))
    ids = np.concatenate([half, half], axis=1).astype(np.int64)
    pos = [int(x) for x in np.linspace(8, 254, 12).round()]
    voc = sorted(int(x) for x in rng.choice(MLA_PROBE_VOCAB, size=32, replace=False))
    return ids, pos, voc

@torch.no_grad()
def _mla_probe_eval(model, ids, pos, voc):
    was = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=dev)
        logits = model(x).logits.double().cpu()
    finally:
        model.train(was)
    x = x.cpu()
    lp = torch.log_softmax(logits, dim=-1)
    nll = -lp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1)
    P = torch.tensor(pos, dtype=torch.long)
    V = torch.tensor(voc, dtype=torch.long)
    sel = logits[:, P][:, :, V]
    true = logits[:, P].gather(-1, x[:, P + 1][:, :, None]).squeeze(-1)
    return {"logits": sel.tolist(), "true_logit": true.tolist(), "nll": nll.tolist()}

def mla_probe(model):
    ids, pos, voc = _mla_probe_inputs()
    out = _mla_probe_eval(model, ids, pos, voc)
    return {"seed": MLA_PROBE_SEED, "ids": ids.tolist(), "positions": pos, "vocab": voc,
            "tol": MLA_PROBE_TOL, "dtype": str(next(model.parameters()).dtype), **out}

def check_probe(model, probe, verbose=True):
    for k in ("ids", "positions", "vocab", "logits", "true_logit", "nll", "tol"):
        if k not in probe:
            raise KeyError(f"mla probe is missing {k!r}")
    got = _mla_probe_eval(model, probe["ids"], probe["positions"], probe["vocab"])
    dev = {k: float(np.abs(np.asarray(got[k], dtype=np.float64)
                           - np.asarray(probe[k], dtype=np.float64)).max())
           for k in ("logits", "true_logit", "nll")}
    worst = max(dev.values())
    tol, stored = float(MLA_PROBE_TOL), float(probe["tol"])
    if verbose:
        note = "" if abs(stored - tol) <= 1e-12 else f", sidecar wrote {stored:.1e}"
        flag = "" if worst <= tol else "  <- above the reference; O(1) would be a fault, 1e-3..5e-2 is f32 noise"
        print(f"    mla probe (diagnostic, not a gate): max |dev| logits {dev['logits']:.2e}  "
              f"true {dev['true_logit']:.2e}  nll {dev['nll']:.2e}  (reference {tol:.1e}{note}){flag}")
    return dev

def _skeleton_mla_budget(model, meta):
    if "B" not in meta:
        return _skeleton_mla(model, meta)
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MLA_SCHEMA!r}")
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
    _install_mla(model, spec)
    return dict(L=L, H=H, G=G, B=B, w=w, d_c=d_c, rotary_mode=meta["rotary_mode"], d_k=d_k, dm=dm,
                cache_width=w + d_c)

skeleton_mla = _skeleton_mla_budget
mla_fingerprint = _fp_mla

MLA_ORACLE_VERSION = "oracle/1"
ORACLE_ARMS = {"shared": "mla_oracle", "per_head": "mla_oracle_per_head"}
ORACLE_BIAS_SET = ["uk", "uv"]
_LAYER_KEYS = ("slot_plane", "slot_head", "dC", "W_DKV", "W_KR", "W_UQ", "W_QR", "W_UK", "W_UV", "Psi")
_INT_KEYS = ("slot_plane", "slot_head", "dC")
_TOP_KEYS = ("schema", "arm", "version", "rotary_mode", "G", "B", "w", "d_c", "L", "H", "d_h", "dm", "r",
             "provenance", "bias_set", "mla_scale", "layers")
_PROV_KEYS = ("fisher_sha256", "sigma_sha256", "s2norm_sha256", "text")
EXPECT_KEYS = ("arm", "G", "w", "d_c", "rotary_mode", "bias_set", "mla_scale", "teacher_attn_sha256")

def _tensor_bytes(h, name, t):
    arr = t.detach().cpu().contiguous().numpy()
    h.update(f"{name}|{tuple(arr.shape)}|{arr.dtype}|".encode("utf-8"))
    h.update(arr.tobytes())

def teacher_attn_digest(model):
    if hasattr(get_layers(model)[0], "self_attn"):
        import dt_mla_arch_llama
        return dt_mla_arch_llama.teacher_attn_digest(model)
    h = hashlib.sha256()
    for li, lay in enumerate(get_layers(model)):
        a = lay.attention
        if "query_key_value" not in a._modules:
            raise RuntimeError(f"layer {li} has no fused query_key_value -- this model is already "
                               f"restructured; the teacher digest is defined on the teacher only")
        for sub in ("query_key_value", "dense"):
            m = getattr(a, sub)
            _tensor_bytes(h, f"{li}.{sub}.weight", m.weight)
            if m.bias is not None:
                _tensor_bytes(h, f"{li}.{sub}.bias", m.bias)
    return h.hexdigest()

def _dense_digest(model):
    h = hashlib.sha256()
    for li, lay in enumerate(get_layers(model)):
        d = lay.attention.dense
        _tensor_bytes(h, f"{li}.dense.weight", d.weight)
        if d.bias is not None:
            _tensor_bytes(h, f"{li}.dense.bias", d.bias)
    return h.hexdigest()

def oracle_weights(model, *, w, d_c, rotary_mode, G=None, seed=MLA_PROBE_SEED):
    L, H, d_k, d_v, dm, r = geometry(model)
    h = r // 2
    if rotary_mode not in MLA_MODES:
        raise ValueError(f"rotary_mode {rotary_mode!r} not in {MLA_MODES}")
    if not (isinstance(w, int) and 2 <= w <= H * r and w % 2 == 0):
        raise ValueError(f"oracle w={w!r} must be an even int in [2, H*r={H * r}]")
    if not (isinstance(d_c, int) and 1 <= d_c <= dm):
        raise ValueError(f"oracle d_c={d_c!r} must be an int in [1, dm={dm}]")
    B = w + d_c
    if G is not None and B != G * (d_k + d_v):
        raise ValueError(f"w + d_c = {B} != B = 128*G = {G * (d_k + d_v)} at G={G}")
    shared, full, half = rotary_mode == "shared", w == H * r, w // 2
    pairs = [(p, i) for p in range(h) for i in range(H)]
    if full:
        kept = pairs
    else:
        order = np.random.default_rng(seed).permutation(len(pairs))
        kept = sorted(pairs[k] for k in order[:half])
    kept_planes = [set() for _ in range(H)]
    for p, i in kept:
        kept_planes[i].add(p)
    slot_plane = np.array([p for p, _ in kept] * 2, dtype=np.int64)
    slot_head = (np.array([i for _, i in kept] * 2, dtype=np.int64) if not shared
                 else np.full(w, -1, dtype=np.int64))
    zero_kept_keys = shared and not full
    if zero_kept_keys:
        cdims = [list(range(d_k)) for _ in range(H)]
    else:
        cdims = [[j for j in range(d_k) if j >= r or (j % h) not in kept_planes[i]] for i in range(H)]
    dC = np.array([len(c) for c in cdims], dtype=np.int64)
    layers = []
    for lay in get_layers(model):
        Qt, Kt, Vt, _ = gc.read_layer(lay, H, d_k, d_v, dm)
        W_UQ = np.concatenate([Qt[i][cdims[i]] for i in range(H)], axis=0)
        Kc = []
        for i in range(H):
            rows = Kt[i][cdims[i]].copy()
            if zero_kept_keys:
                for p in kept_planes[i]:
                    rows[p] = 0.0
                    rows[p + h] = 0.0
            Kc.append(rows)
        Kc = np.concatenate(Kc, axis=0)
        V = Vt.reshape(H * d_v, dm + 1)
        if d_c == dm:
            W_DKV = np.concatenate([np.eye(dm), np.zeros((dm, 1))], axis=1)
            W_UK, W_UV = Kc.copy(), V.copy()
        else:
            _, _, Vh = np.linalg.svd(np.concatenate([Kc[:, :dm], V[:, :dm]], axis=0),
                                     full_matrices=False)
            P = Vh[:d_c]
            P = P * np.sign(P[np.arange(d_c), np.abs(P).argmax(axis=1)])[:, None]
            W_DKV = np.concatenate([P, np.zeros((d_c, 1))], axis=1)
            W_UK = np.concatenate([Kc[:, :dm] @ P.T, Kc[:, dm:]], axis=1)
            W_UV = np.concatenate([V[:, :dm] @ P.T, V[:, dm:]], axis=1)
        W_KR = np.zeros((w, dm + 1))
        W_QR = np.zeros((H * w if shared else w, dm + 1))
        Psi = np.zeros((H, r, w))
        for s, (p, i) in enumerate(kept):
            W_KR[s], W_KR[s + half] = Kt[i][p], Kt[i][p + h]
            base = i * w if shared else 0
            W_QR[base + s], W_QR[base + s + half] = Qt[i][p], Qt[i][p + h]
            Psi[i, p, s] = 1.0
            Psi[i, p + h, s + half] = 1.0
        layers.append({k: np.ascontiguousarray(v) for k, v in (
            ("slot_plane", slot_plane.copy()), ("slot_head", slot_head.copy()), ("dC", dC.copy()),
            ("W_DKV", W_DKV), ("W_KR", W_KR), ("W_UQ", W_UQ), ("W_QR", W_QR), ("W_UK", W_UK),
            ("W_UV", W_UV), ("Psi", Psi))})
    return {"schema": MLA_SCHEMA, "arm": ORACLE_ARMS[rotary_mode], "version": MLA_ORACLE_VERSION,
            "rotary_mode": rotary_mode, "G": G, "B": B, "w": w, "d_c": d_c, "L": L, "H": H, "d_h": d_k,
            "dm": dm, "r": r,
            "provenance": {"fisher_sha256": None, "sigma_sha256": None, "s2norm_sha256": None,
                           "text": f"data-free oracle fixture, kept-pair seed {seed}" + (
                               "" if full else f", {len(kept)} pairs")},
            "bias_set": list(ORACLE_BIAS_SET), "mla_scale": float(d_k) ** -0.5, "layers": layers}

def validate_weights(weights, model, meta):
    arm = weights.get("arm") if isinstance(weights, dict) else None

    def need(cond, msg):
        if not cond:
            raise ValueError(f"validate_weights[{arm!r}]: {msg}")

    need(isinstance(weights, dict), "weights must be a dict")
    for k in EXPECT_KEYS:
        need(k in meta, f"expectation meta lacks {k!r}")
    missing = [k for k in _TOP_KEYS if k not in weights]
    extra = [k for k in weights if k not in _TOP_KEYS and not str(k).startswith("_")]
    need(not missing and not extra, f"top-level keys: missing {missing}, unexpected {extra} "
                                    f"(diagnostic extras must start with '_')")
    need(weights["schema"] == MLA_SCHEMA, f"schema {weights['schema']!r} != {MLA_SCHEMA!r}")
    need(weights["arm"] == meta["arm"], f"arm {weights['arm']!r} != dispatching arm {meta['arm']!r}")
    need(isinstance(weights["version"], str) and weights["version"], "version must be a non-empty str")
    mode = weights["rotary_mode"]
    need(mode in MLA_MODES and mode == meta["rotary_mode"],
         f"rotary_mode {mode!r} (registry says {meta['rotary_mode']!r})")
    L, H, d_k, d_v, dm, r = geometry(model)
    for k in ("B", "w", "d_c", "L", "H", "d_h", "dm", "r"):
        need(isinstance(weights[k], int) and not isinstance(weights[k], bool), f"{k} must be an int")
    need((weights["L"], weights["H"], weights["d_h"], weights["dm"], weights["r"]) == (L, H, d_k, dm, r),
         f"geometry (L,H,d_h,dm,r) = {(weights['L'], weights['H'], weights['d_h'], weights['dm'], weights['r'])} "
         f"!= the model's {(L, H, d_k, dm, r)}")
    need(d_k == d_v, "d_k != d_v")
    G, B, w, d_c = weights["G"], weights["B"], weights["w"], weights["d_c"]
    need(G == meta["G"], f"G {G!r} != expected {meta['G']!r}")
    if G is None:
        need(meta["G"] is None, "G=None is for self-test widths only")
    else:
        need(isinstance(G, int) and not isinstance(G, bool), f"G must be an int or None, got {G!r}")
        need(B == G * (d_k + d_v), f"B={B} != 128*G = {G * (d_k + d_v)}")
    need(w >= 2 and w % 2 == 0, f"w={w} must be even and >= 2 (D4)")
    need(d_c >= 1 and w + d_c == B, f"w + d_c = {w + d_c} != B = {B}")
    need((w, d_c) == (meta["w"], meta["d_c"]), f"(w, d_c) = {(w, d_c)} != the width table's "
                                               f"{(meta['w'], meta['d_c'])} (_mla_widths is the only route)")
    prov = weights["provenance"]
    need(isinstance(prov, dict) and sorted(prov) == sorted(_PROV_KEYS), f"provenance keys must be {_PROV_KEYS}")
    need(isinstance(prov["text"], str), "provenance.text must be a str")
    for k in _PROV_KEYS[:3]:
        need(prov[k] is None or isinstance(prov[k], str), f"provenance.{k} must be a str or None")
    bs = weights["bias_set"]
    need(isinstance(bs, list) and bs == sorted(set(bs)) and set(bs) <= set(MLA_BIAS_MEMBERS),
         f"bias_set {bs!r} must be a sorted duplicate-free list within {MLA_BIAS_MEMBERS}")
    need(bs == sorted(meta["bias_set"]), f"bias_set {bs} != the registry's {sorted(meta['bias_set'])} (R1)")
    sc = weights["mla_scale"]
    need(isinstance(sc, float) and 0.0 < sc < math.inf, f"mla_scale {sc!r} must be a finite positive float")
    need(sc == meta["mla_scale"], f"mla_scale {sc!r} != the registry's {meta['mla_scale']!r} (D1)")
    lays = weights["layers"]
    need(isinstance(lays, list) and len(lays) == L, f"layers must be a list of L={L} dicts")
    half, h = w // 2, r // 2
    f = float(d_k) ** -0.5 / sc
    worst = {"psi_offblock": 0.0, "psi_complex_linear": 0.0, "psi_per_head": 0.0, "qr_consistency": 0.0}
    for li, lw in enumerate(lays):
        need(isinstance(lw, dict) and sorted(lw) == sorted(_LAYER_KEYS),
             f"layer {li}: keys {sorted(lw) if isinstance(lw, dict) else lw!r} != {sorted(_LAYER_KEYS)}")
        for k in _LAYER_KEYS:
            a = lw[k]
            need(isinstance(a, np.ndarray) and a.flags["C_CONTIGUOUS"], f"layer {li}: {k} must be a "
                                                                          f"C-contiguous numpy array")
            need(a.dtype == (np.int64 if k in _INT_KEYS else np.float64),
                 f"layer {li}: {k} dtype {a.dtype} (indices int64, floats float64)")
            need(np.all(np.isfinite(a)), f"layer {li}: {k} is not finite")
        need(lw["slot_plane"].shape == (w,) and lw["slot_head"].shape == (w,) and lw["dC"].shape == (H,),
             f"layer {li}: index shapes")
        D = int(lw["dC"].sum())
        shapes = {"W_DKV": (d_c, dm + 1), "W_KR": (w, dm + 1), "W_UQ": (D, dm + 1),
                  "W_QR": ((H * w if mode == "shared" else w), dm + 1), "W_UK": (D, d_c + 1),
                  "W_UV": (H * d_v, d_c + 1), "Psi": (H, r, w)}
        for k, shp in shapes.items():
            need(lw[k].shape == shp, f"layer {li}: {k} shape {lw[k].shape} != {shp}")
    try:
        _mla_check_spec({"w": w, "d_c": d_c, "rotary_mode": mode, "bias_set": bs, "mla_scale": sc,
                         "layers": [{k: lw[k].tolist() for k in _INT_KEYS} for lw in lays]},
                        L, H, d_k, r)
    except (ValueError, TypeError) as e:
        need(False, str(e))
    for li, lw in enumerate(lays):
        sp, sh = lw["slot_plane"], lw["slot_head"]
        for member, key in (("dkv", "W_DKV"), ("uk", "W_UK"), ("uv", "W_UV")):
            if member not in bs:
                need(np.all(lw[key][:, -1] == 0.0), f"layer {li}: {key}'s bias column must be exactly 0 "
                                                    f"because {member!r} is not in bias_set (R1)")
        Psi = lw["Psi"]
        mx = float(np.abs(Psi).max())
        need(mx > 0.0, f"layer {li}: Psi is identically zero")
        off = (np.arange(r)[:, None] % h) != sp[None, :]
        v_off = float(np.abs(Psi[:, off]).max()) if off.any() else 0.0
        v_cl = max(float(np.abs(Psi[:, :h, :half] - Psi[:, h:, half:]).max()),
                   float(np.abs(Psi[:, h:, :half] + Psi[:, :h, half:]).max()))
        worst["psi_offblock"] = max(worst["psi_offblock"], v_off / mx)
        worst["psi_complex_linear"] = max(worst["psi_complex_linear"], v_cl / mx)
        need(v_off <= 1e-12 * mx, f"layer {li}: Psi is not block-sparse by plane ({v_off:.3e} vs max {mx:.3e})")
        need(v_cl <= 1e-12 * mx, f"layer {li}: Psi is not complex-linear ({v_cl:.3e} vs max {mx:.3e}) -- "
                                 f"a reflection-type or conjugated-by-row block")
        Qt, _, _, _ = gc.read_layer(get_layers(model)[li], H, d_k, d_v, dm)
        WQR = lw["W_QR"]
        if mode == "shared":
            for i in range(H):
                blk = WQR[i * w:(i + 1) * w]
                dev = float(np.linalg.norm(blk - f * (Psi[i].T @ Qt[i][:r])))
                nrm = float(np.linalg.norm(blk))
                worst["qr_consistency"] = max(worst["qr_consistency"], dev / nrm if nrm else dev)
                need(dev <= 1e-10 * nrm, f"layer {li} head {i}: W_QR_i != (d_h^-1/2/mla_scale) Psi^T Qt[:r] "
                                         f"(dev {dev:.3e}, norm {nrm:.3e})")
        else:
            notown = sh[None, :] != np.arange(H)[:, None]
            v_ph = float(np.abs(Psi.transpose(0, 2, 1)[notown]).max()) if notown.any() else 0.0
            worst["psi_per_head"] = max(worst["psi_per_head"], v_ph / mx)
            need(v_ph <= 1e-12 * mx, f"layer {li}: Psi[i,:,s] must vanish for slots of other heads")
            ref = f * sum(Psi[i].T @ Qt[i][:r] for i in range(H))
            dev, nrm = float(np.linalg.norm(WQR - ref)), float(np.linalg.norm(WQR))
            worst["qr_consistency"] = max(worst["qr_consistency"], dev / nrm if nrm else dev)
            need(dev <= 1e-10 * nrm, f"layer {li}: W_QR != (d_h^-1/2/mla_scale) sum_i Psi_i^T Qt_i[:r] "
                                     f"(dev {dev:.3e}, norm {nrm:.3e})")
    return worst

def restructure_mla(model, meta, weights):
    for k in EXPECT_KEYS:
        if k not in meta:
            raise KeyError(f"restructure_mla: expectation meta lacks {k!r}")
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention implementation is {impl!r}; MLA installs on eager only")
    devs = {p.device.type for p in model.parameters()}
    if devs != {"cpu"}:
        raise RuntimeError(f"restructure_mla runs on CPU (the probe and fingerprints are CPU-side); "
                           f"model is on {sorted(devs)}")
    if teacher_attn_digest(model) != meta["teacher_attn_sha256"]:
        raise RuntimeError("teacher attention digest changed between build_weights and restructure_mla: "
                           "the initialiser MUTATED the model, or a different model was passed")
    report = validate_weights(weights, model, meta)
    L, H, d_k, d_v, dm, r = geometry(model)
    dense0 = _dense_digest(model)
    w, d_c, mode = weights["w"], weights["d_c"], weights["rotary_mode"]
    spec = {"w": w, "d_c": d_c, "rotary_mode": mode, "bias_set": list(weights["bias_set"]),
            "mla_scale": float(weights["mla_scale"]),
            "layers": [{k: [int(x) for x in lw[k]] for k in _INT_KEYS} for lw in weights["layers"]]}
    _install_mla(model, spec)
    worst_cast = 0.0
    for li, lay in enumerate(get_layers(model)):
        a, lw = lay.attention, weights["layers"][li]
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
                worst_cast = max(worst_cast, float(np.abs(back - arr).max()))
    if _dense_digest(model) != dense0:
        raise RuntimeError("restructure_mla changed attention.dense -- it must be left untouched")
    mm = mla_meta(model)
    check_alloc(model, mm)
    record = {"arm": weights["arm"], "version": weights["version"], "G": weights["G"], "w": w, "d_c": d_c,
              "rotary_mode": mode, "bias_set": list(weights["bias_set"]),
              "mla_scale": float(weights["mla_scale"]), "provenance": dict(weights["provenance"]),
              "validate": report, "max_cast_error": worst_cast, "alloc_sha256": mm["alloc_sha256"],
              "attn_params_per_layer": mm["attn_params_per_layer"], "cache_width": w + d_c,
              "dtype": str(next(model.parameters()).dtype)}
    model.mla_build_record = record
    return record

def mla_meta(model):
    L, H, d_k, d_v, dm, r = geometry(model)
    attns = [lay.attention for lay in get_layers(model)]
    for li, a in enumerate(attns):
        if type(a).__name__ != "MLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not MLAAttention")
    a0 = attns[0]
    one = {(a.w, a.d_c, a.rotary_mode, a.bias_set, a.mla_scale_value) for a in attns}
    if len(one) != 1:
        raise AssertionError(f"layers disagree on (w, d_c, mode, bias_set, scale): {sorted(map(str, one))}")
    meta = {"mla_schema": MLA_SCHEMA, "rotary_mode": a0.rotary_mode, "w": int(a0.w), "d_c": int(a0.d_c),
            "mla_scale": float(a0.mla_scale_value), "bias_set": list(a0.bias_set),
            "mla_layers": [{k: [int(x) for x in a.get_buffer(k).detach().cpu().tolist()]
                            for k in ("dC", "slot_plane", "slot_head")} for a in attns]}
    meta["alloc_sha256"] = _mla_alloc_sha(meta)
    counts = [attn_param_count(H, d_k, dm, meta["w"], meta["d_c"], sum(lay["dC"]), meta["rotary_mode"],
                               meta["bias_set"]) for lay in meta["mla_layers"]]
    meta["attn_params_by_layer"] = counts
    meta["attn_params_per_layer"] = counts[0] if len(set(counts)) == 1 else None
    return meta
