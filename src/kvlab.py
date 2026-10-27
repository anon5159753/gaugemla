import glob
import os

import numpy as np
import torch

D_HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(D_HERE, "_text")

DTYPE = torch.float32

def load_model(model_name, dtype=DTYPE):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager"
    )
    impl = getattr(model.config, "_attn_implementation", "unknown")
    assert impl == "eager", f"attention fell back to {impl!r}; every instrument needs eager"
    if getattr(model.config, "model_type", None) == "llama":
        tok.add_bos_token = False
    if getattr(model.config, "model_type", None) == "olmo":
        tok.add_bos_token = False
        tok.add_eos_token = False
    _probe = "The cat sat on the mat. 12 + 7 = 19"
    if tok(_probe).input_ids != tok(_probe, add_special_tokens=False).input_ids:
        raise RuntimeError(f"{model_name}: the tokenizer still adds special tokens to a plain encode "
                           f"({tok(_probe).input_ids[:3]}... vs {tok(_probe, add_special_tokens=False).input_ids[:3]}...)")
    if getattr(model.config, "clip_qkv", None) is not None:
        raise NotImplementedError(f"clip_qkv = {model.config.clip_qkv}: the key is not linear in x")
    _expand_gqa(model)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok

def _expand_gqa(model):
    import torch
    c = model.config
    H = int(c.num_attention_heads)
    Hkv = int(getattr(c, "num_key_value_heads", None) or H)
    if Hkv == H:
        return 1
    if getattr(c, "model_type", None) not in ("llama",) or not hasattr(get_layers(model)[0], "self_attn"):
        raise NotImplementedError(f"GQA expansion is written for the LLaMA layout; {c.model_type} is not")
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of the {Hkv} KV heads")
    n_rep = H // Hkv
    d = int(getattr(c, "head_dim", None) or c.hidden_size // H)
    with torch.no_grad():
        for lay in get_layers(model):
            a = lay.self_attn
            for name in ("k_proj", "v_proj"):
                lin = getattr(a, name)
                if lin.weight.shape[0] != Hkv * d:
                    raise RuntimeError(f"{name} has {lin.weight.shape[0]} rows, expected Hkv*d = {Hkv * d}")
                new = torch.nn.Linear(lin.in_features, H * d, bias=lin.bias is not None).to(
                    dtype=lin.weight.dtype, device=lin.weight.device)
                new.weight.copy_(lin.weight.view(Hkv, d, -1).repeat_interleave(n_rep, dim=0).reshape(H * d, -1))
                if lin.bias is not None:
                    new.bias.copy_(lin.bias.view(Hkv, d).repeat_interleave(n_rep, dim=0).reshape(H * d))
                setattr(a, name, new)
            a.num_key_value_groups = 1
            if hasattr(a, "num_key_value_heads"):
                a.num_key_value_heads = H
    c.num_key_value_heads = H
    object.__setattr__(model, "_gqa_kv_heads", Hkv)
    return n_rep

def get_layers(model):
    for path in (
        lambda m: m.gpt_neox.layers,
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.model.decoder.layers,
    ):
        try:
            layers = path(model)
            if len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        "add its attribute path to get_layers()"
    )

def detach_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(detach_tree(o) for o in obj)
    if isinstance(obj, list):
        return [detach_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: detach_tree(v) for k, v in obj.items()}
    return obj

def geometry(model):
    c = model.config
    d_model = c.hidden_size
    H = c.num_attention_heads
    return len(get_layers(model)), H, d_model // H, d_model

def load_batch(tok, seq_len, batch_idx=0, chars_per_doc=30000):
    paths = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    if not paths:
        raise RuntimeError(f"no calibration text found under {TEXT_DIR}")
    for off in range(len(paths)):
        p = paths[(batch_idx + off) % len(paths)]
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read(chars_per_doc)
        ids = tok(txt, return_tensors=None, add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            return torch.tensor(ids[:seq_len], dtype=torch.long).unsqueeze(0), os.path.basename(p)
    raise RuntimeError(f"no document long enough for seq_len={seq_len}")

def load_batch_tiled(tok, seq_len, text):
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < seq_len:
        ids = torch.cat([ids, ids], dim=1)
    return ids[:, :seq_len], "TILED-CALIB"

def norm_flat(t):
    return float(t.float().norm())

def norm_pertoken(t):
    return float(t.float().norm(dim=-1).mean())

NORMS = {"flat": norm_flat, "pertoken": norm_pertoken}

def split_qkv(layer, H, d_k, d_model, wrong=False):
    W = layer.attention.query_key_value.weight.detach().double().numpy()
    b = layer.attention.query_key_value.bias.detach().double().numpy()
    if wrong:
        Wq, Wk, Wv = W[:d_model], W[d_model:2 * d_model], W[2 * d_model:]
        bq, bk, bv = b[:d_model], b[d_model:2 * d_model], b[2 * d_model:]
        return (Wq.reshape(H, d_k, d_model), Wk.reshape(H, d_k, d_model),
                Wv.reshape(H, d_k, d_model), bq.reshape(H, d_k),
                bk.reshape(H, d_k), bv.reshape(H, d_k))
    Wr = W.reshape(H, 3 * d_k, d_model)
    br = b.reshape(H, 3 * d_k)
    return (Wr[:, 0:d_k, :], Wr[:, d_k:2 * d_k, :], Wr[:, 2 * d_k:3 * d_k, :],
            br[:, 0:d_k], br[:, d_k:2 * d_k], br[:, 2 * d_k:3 * d_k])

def merge_qkv(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, d_model):
    Wr = np.concatenate([Wq, Wk, Wv], axis=1)
    br = np.concatenate([bq, bk, bv], axis=1)
    W = Wr.reshape(3 * d_model, d_model)
    b = br.reshape(3 * d_model)
    p = layer.attention.query_key_value
    with torch.no_grad():
        p.weight.copy_(torch.from_numpy(W).to(p.weight.dtype))
        p.bias.copy_(torch.from_numpy(b).to(p.bias.dtype))

def groups_of(H, G):
    assert H % G == 0, f"H={H} not divisible by G={G}"
    per = H // G
    return [list(range(j * per, (j + 1) * per)) for j in range(G)]

def pool_layer(layer, G, H, d_k, d_model, which="kv"):
    assert which in ("kv", "k", "v"), f"which={which!r}"
    Wq, Wk, Wv, bq, bk, bv = split_qkv(layer, H, d_k, d_model)
    Wk, Wv, bk, bv = Wk.copy(), Wv.copy(), bk.copy(), bv.copy()
    for idx in groups_of(H, G):
        if which in ("kv", "k"):
            Wk[idx] = Wk[idx].mean(axis=0, keepdims=True)
            bk[idx] = bk[idx].mean(axis=0, keepdims=True)
        if which in ("kv", "v"):
            Wv[idx] = Wv[idx].mean(axis=0, keepdims=True)
            bv[idx] = bv[idx].mean(axis=0, keepdims=True)
    merge_qkv(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, d_model)

def wse_weights(layer, G, H, d_k, d_model):
    _, Wk, Wv, _, _, _ = split_qkv(layer, H, d_k, d_model)
    tot_k = tot_v = 0.0
    for idx in groups_of(H, G):
        Dk = Wk[idx] - Wk[idx].mean(axis=0, keepdims=True)
        Dv = Wv[idx] - Wv[idx].mean(axis=0, keepdims=True)
        tot_k += float((Dk ** 2).mean(axis=(1, 2)).sum())
        tot_v += float((Dv ** 2).mean(axis=(1, 2)).sum())
    return {"wse_w": tot_k + tot_v, "wse_w_K": tot_k, "wse_w_V": tot_v}

def wse_activations(Kact, Vact, G, H):
    tot_k = tot_v = 0.0
    for idx in groups_of(H, G):
        Dk = Kact[idx] - Kact[idx].mean(axis=0, keepdims=True)
        Dv = Vact[idx] - Vact[idx].mean(axis=0, keepdims=True)
        tot_k += float((Dk ** 2).mean(axis=(1, 2)).sum())
        tot_v += float((Dv ** 2).mean(axis=(1, 2)).sum())
    return {"wse_a": tot_k + tot_v, "wse_a_K": tot_k, "wse_a_V": tot_v}

def gauge_rescale(layer, c, H, d_k, d_model):
    Wq, Wk, Wv, bq, bk, bv = split_qkv(layer, H, d_k, d_model)
    Wq, Wk, bq, bk = Wq.copy(), Wk.copy(), bq.copy(), bk.copy()
    c = np.asarray(c, dtype=np.float64)
    Wk *= c[:, None, None]
    bk *= c[:, None]
    Wq /= c[:, None, None]
    bq /= c[:, None]
    merge_qkv(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, d_model)

def tv_kl(logits_a, logits_b):
    la = torch.log_softmax(logits_a.float(), dim=-1)
    lb = torch.log_softmax(logits_b.float(), dim=-1)
    pa = la.exp()
    tv = 0.5 * (pa - lb.exp()).abs().sum(-1)
    kl = (pa * (la - lb)).sum(-1)
    return float(tv.mean()), float(kl.mean())

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    def rank(v):
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            sums = np.zeros(len(cnt))
            np.add.at(sums, inv, r)
            r = (sums / cnt)[inv]
        return r
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")
