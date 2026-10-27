import numpy as np

import kvlab

def is_llama(model):
    try:
        a = kvlab.get_layers(model)[0].self_attn
    except (AttributeError, IndexError):
        return False
    return all(hasattr(a, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))

def geometry(model):
    L, H, d_k, dm = kvlab.geometry(model)
    c = model.config
    hd = getattr(c, "head_dim", None)
    if hd is not None and hd != d_k:
        raise RuntimeError("config.head_dim = %d but hidden_size/num_attention_heads = %d; "
                           "kvlab.geometry's d_model//H is wrong for this checkpoint" % (hd, d_k))
    kv = getattr(c, "num_key_value_heads", H)
    if kv != H:
        raise RuntimeError("%s is GQA (%d key/value heads of %d query heads). Every solve here "
                           "assumes one key head per query head; a GQA path is separate work."
                           % (getattr(c, "_name_or_path", "model"), kv, H))
    r = 2 * int(_rotary_emb(model).inv_freq.shape[0])
    if not 0 < r <= d_k or r % 2:
        raise RuntimeError("2*len(inv_freq) = %d is not an even rotary width in (0, %d]" % (r, d_k))
    pct = getattr(c, "rotary_pct", None) or getattr(c, "partial_rotary_factor", None) or 1.0
    if int(round(d_k * pct)) != r:
        raise RuntimeError("inv_freq says r = %d but the config's rotary fraction %.4f of d_k = %d "
                           "says %d" % (r, pct, d_k, int(round(d_k * pct))))
    return L, H, d_k, d_k, dm, r

def _rotary_emb(model):
    for get in (lambda m: m.model.rotary_emb,
                lambda m: kvlab.get_layers(m)[0].self_attn.rotary_emb):
        try:
            e = get(model)
            if e is not None and hasattr(e, "inv_freq"):
                return e
        except (AttributeError, IndexError):
            continue
    raise RuntimeError("could not locate a rotary_emb with inv_freq on %s" % type(model).__name__)

def inv_freq(model):
    return _rotary_emb(model).inv_freq.detach().double().numpy().copy()

def split_qkv(layer, H, d_k, dm):
    a = layer.self_attn
    out, bs = [], []
    for name, d in (("q_proj", d_k), ("k_proj", d_k), ("v_proj", _vdim(a, H))):
        lin = getattr(a, name)
        W = lin.weight.detach().double().numpy()
        if W.shape != (H * d, dm):
            raise RuntimeError("%s.weight is %s, expected %s" % (name, W.shape, (H * d, dm)))
        out.append(np.ascontiguousarray(W.reshape(H, d, dm)))
        if lin.bias is None:
            bs.append(np.zeros((H, d)))
        else:
            b = lin.bias.detach().double().numpy()
            if b.shape != (H * d,):
                raise RuntimeError("%s.bias is %s, expected %s" % (name, b.shape, (H * d,)))
            bs.append(np.ascontiguousarray(b.reshape(H, d)))
    Wq, Wk, Wv = out
    return (Wq, Wk, Wv, bs[0], bs[1], bs[2])

def _write_linear(lin, W, b, name):
    import torch
    if tuple(lin.weight.shape) != tuple(W.shape):
        raise RuntimeError("%s.weight is %s, the write is %s" % (name, tuple(lin.weight.shape), W.shape))
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(W)).to(lin.weight.dtype))
        if lin.bias is not None:
            lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)).to(lin.bias.dtype))
        elif np.any(b != 0.0):
            lin.bias = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(b)).to(
                dtype=lin.weight.dtype, device=lin.weight.device), requires_grad=lin.weight.requires_grad)

def merge_qkv(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm):
    a = layer.self_attn
    for name, W, b in (("q_proj", Wq, bq), ("k_proj", Wk, bk), ("v_proj", Wv, bv)):
        H_, d, dm_ = W.shape
        if H_ != H or dm_ != dm or b.shape != (H, d):
            raise RuntimeError("%s: write shapes %s / %s for H=%d dm=%d" % (name, W.shape, b.shape, H, dm))
        _write_linear(getattr(a, name), W.reshape(H * d, dm), b.reshape(H * d), name)

def merge_wo(layer, Wo, b=None):
    import torch
    W = np.concatenate([Wo[i] for i in range(Wo.shape[0])], axis=1)
    p = layer.self_attn.o_proj
    with torch.no_grad():
        p.weight.copy_(torch.from_numpy(W).to(p.weight.dtype))
        if b is not None and p.bias is not None:
            p.bias.copy_(torch.from_numpy(b).to(p.bias.dtype))

def pool_layer(layer, G, H, d_k, dm, which="kv"):
    assert which in ("kv", "k", "v"), f"which={which!r}"
    Wq, Wk, Wv, bq, bk, bv = split_qkv(layer, H, d_k, dm)
    Wk, Wv, bk, bv = Wk.copy(), Wv.copy(), bk.copy(), bv.copy()
    for idx in kvlab.groups_of(H, G):
        if which in ("kv", "k"):
            Wk[idx] = Wk[idx].mean(axis=0, keepdims=True)
            bk[idx] = bk[idx].mean(axis=0, keepdims=True)
        if which in ("kv", "v"):
            Wv[idx] = Wv[idx].mean(axis=0, keepdims=True)
            bv[idx] = bv[idx].mean(axis=0, keepdims=True)
    merge_qkv(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm)

def _vdim(a, H):
    return a.v_proj.weight.shape[0] // H

def split_wo(layer, H, d_v, dm):
    lin = layer.self_attn.o_proj
    W = lin.weight.detach().double().numpy()
    if W.shape != (dm, H * d_v):
        raise RuntimeError("o_proj.weight is %s, expected %s" % (W.shape, (dm, H * d_v)))
    b = None if lin.bias is None else lin.bias.detach().double().numpy()
    return np.stack([W[:, i * d_v:(i + 1) * d_v] for i in range(H)]), b

def ln_gain(layer):
    ln = layer.input_layernorm
    if getattr(ln, "weight", None) is None:
        if type(ln).__name__ != "OlmoLayerNorm":
            raise RuntimeError(f"input_layernorm {type(ln).__name__} has no weight and is not OlmoLayerNorm")
        return np.ones(int(ln.normalized_shape[0]), dtype=np.float64)
    return ln.weight.detach().double().numpy().copy()
