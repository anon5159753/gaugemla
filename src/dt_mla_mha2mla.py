import hashlib
import math
import numbers
import os

import numpy as np
import torch

import gam_core as gc
import kvlab

ARM = "mla_mha2mla_s2norm"
SCHEMA = "mla/v1"
VERSION = "mha2mla-init/1 (2026-09-29)"
ROTARY_MODE = "per_head"
BIAS_SET = ["uk", "uv"]
MLA_SCALE = 0.125

def mla_scale_of(w, d_h):
    return float(d_h ** -0.5)

PIN ="JT-Ushio/MHA2MLA@7a56785ac2a814ec594b33d472a174693a55a952"
REL_GAP_WARN = 1e-3
ATTN_INPUT_TOL = 1e-6
QKV_LAYOUT_TOL = 1e-4
FIT_SHAPE = (64, 512)
S2NORM_KEYS_REQUIRED = ("q", "k", "n_tok")
S2NORM_KEYS_OPTIONAL = ("score",)
S2NORM_CACHE_SCHEMA = "s2norm-cache/v1"
S2NORM_FIT_VERSION = "s2norm-fit/1: input_layernorm hook, f64 qk recompute with bias, hypot(p, p+r/2), " \
                     "token mean then doc mean"

class AttnInputMismatch(RuntimeError):
    pass

def _geom(model):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if d_v != d_k:
        raise ValueError(f"d_v={d_v} != d_k={d_k}")
    if r <= 0 or r % 2 or r > d_k:
        raise ValueError(f"rotary width r={r} must be even and in (0, d_k={d_k}]")
    return L, H, d_k, dm, r

def _as_int(name, v, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(v).__name__} {v!r}")
    return int(v)

def _attn_digest(model):
    h = hashlib.sha256()
    for li, layer in enumerate(kvlab.get_layers(model)):
        for name, p in sorted(layer.attention.named_parameters()):
            t = p.detach().cpu().contiguous()
            h.update(f"{li}.{name}.{t.dtype}.{tuple(t.shape)}".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16
                     else t.float().numpy().tobytes())
    return h.hexdigest()

def _check_ids(model, ids, expect_shape):
    if isinstance(ids, np.ndarray):
        ids = torch.from_numpy(ids)
    if not torch.is_tensor(ids):
        raise TypeError(f"ids must be a torch tensor or numpy array, got {type(ids).__name__}")
    if ids.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"ids must be integer token ids, got {ids.dtype}")
    ids = ids.to(torch.long)
    if ids.ndim != 2:
        raise ValueError(f"ids must be (n_doc, seq_len), got {tuple(ids.shape)}")
    if expect_shape is not None and tuple(ids.shape) != tuple(expect_shape):
        raise ValueError(f"ids shape {tuple(ids.shape)} != expected fit split {tuple(expect_shape)}")
    n_vocab = model.get_input_embeddings().num_embeddings
    lo, hi = int(ids.min()), int(ids.max())
    if lo < 0 or hi >= n_vocab:
        raise ValueError(f"token ids span [{lo}, {hi}], outside the vocabulary [0, {n_vocab})")
    return ids

def verify_attn_input(model, ids, *, source="input_layernorm", tol=ATTN_INPUT_TOL):
    layers = kvlab.get_layers(model)
    got_src, got_attn = {}, {}

    def _src(li):
        def fn(_m, _inp, out):
            got_src[li] = out.detach().clone()
        return fn

    def _pre(li):
        def fn(_m, args, kwargs):
            if len(args) > 0:
                x = args[0]
            elif "hidden_states" in kwargs:
                x = kwargs["hidden_states"]
            else:
                raise RuntimeError(f"layer {li}: attention called without hidden_states")
            got_attn[li] = x.detach().clone()
        return fn

    handles = []
    try:
        for li, layer in enumerate(layers):
            handles.append(getattr(layer, source).register_forward_hook(_src(li)))
            handles.append(layer.attention.register_forward_pre_hook(_pre(li), with_kwargs=True))
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if len(got_src) != len(layers) or len(got_attn) != len(layers):
        raise RuntimeError(f"hooks fired on {len(got_src)}/{len(got_attn)} of {len(layers)} layers")
    out = []
    for li in range(len(layers)):
        a, b = got_src[li].double(), got_attn[li].double()
        if a.shape != b.shape:
            raise AttnInputMismatch(f"layer {li}: {source} output {tuple(a.shape)} vs attention "
                                    f"input {tuple(b.shape)}")
        err, scale = float((a - b).abs().max()), float(b.abs().max())
        if not err <= tol * scale:
            raise AttnInputMismatch(f"layer {li}: {source} output is not the attention input: "
                                    f"max|diff| = {err:.3e} > {tol:g} * {scale:.3e}")
        out.append((err, scale))
    return out

def fit_s2norm(model, ids, *, batch_size=8, expect_shape=FIT_SHAPE):
    ids = _check_ids(model, ids, expect_shape)
    if model.training:
        raise RuntimeError("fit_s2norm needs the model in eval mode; it will not change the mode itself")
    batch_size = _as_int("batch_size", batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size={batch_size}")
    L, H, d_k, dm, r = _geom(model)
    h = r // 2
    layers = kvlab.get_layers(model)
    before = _attn_digest(model)

    pos = verify_attn_input(model, ids[:1], source="input_layernorm")
    try:
        verify_attn_input(model, ids[:1], source="post_attention_layernorm")
    except AttnInputMismatch as e:
        neg_msg = str(e)
    else:
        raise RuntimeError("negative control failed: post_attention_layernorm passed verify_attn_input, "
                           "so the check cannot tell the attention input from another tensor")

    WR = []
    for layer in layers:
        Qt, Kt, _, _ = gc.read_layer(layer, H, d_k, d_k, dm)
        WR.append(torch.from_numpy(np.ascontiguousarray(np.concatenate(
            [Qt[:, :r, :].reshape(H * r, dm + 1), Kt[:, :r, :].reshape(H * r, dm + 1)], axis=0))))

    qsum = np.zeros((L, H, h), dtype=np.float64)
    ksum = np.zeros((L, H, h), dtype=np.float64)
    fired_ln = np.zeros(L, dtype=np.int64)
    fired_qkv = np.zeros(L, dtype=np.int64)
    layout_rel = np.zeros(L, dtype=np.float64)
    cur = {}

    def _ln(li):
        def fn(_m, _inp, out):
            x = out.detach().to(torch.float64)
            xa = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
            y = xa @ WR[li].T
            q = y[..., :H * r].reshape(*x.shape[:-1], H, r)
            k = y[..., H * r:].reshape(*x.shape[:-1], H, r)
            qn = torch.hypot(q[..., :h], q[..., h:])
            kn = torch.hypot(k[..., :h], k[..., h:])
            qsum[li] += qn.mean(dim=1).sum(dim=0).numpy()
            ksum[li] += kn.mean(dim=1).sum(dim=0).numpy()
            cur[li] = (q, k)
            fired_ln[li] += 1
        return fn

    def _qkv(li):
        def fn(_m, _inp, out):
            if li not in cur:
                raise RuntimeError(f"layer {li}: query_key_value ran before input_layernorm's hook")
            q64, k64 = cur.pop(li)
            o = out.detach().reshape(*out.shape[:-1], H, 3 * d_k).to(torch.float64)
            q32, k32 = o[..., :r], o[..., d_k:d_k + r]
            err = max(float((q32 - q64).abs().max()), float((k32 - k64).abs().max()))
            scale = max(float(q64.abs().max()), float(k64.abs().max()))
            layout_rel[li] = max(layout_rel[li], err / scale)
            fired_qkv[li] += 1
        return fn

    handles = []
    n_batches = 0
    try:
        for li, layer in enumerate(layers):
            handles.append(layer.input_layernorm.register_forward_hook(_ln(li)))
            handles.append(layer.attention.query_key_value.register_forward_hook(_qkv(li)))
        with torch.no_grad():
            for s in range(0, ids.shape[0], batch_size):
                model(ids[s:s + batch_size], use_cache=False)
                n_batches += 1
                bad = np.flatnonzero(layout_rel > QKV_LAYOUT_TOL)
                if bad.size:
                    raise RuntimeError(f"f64 recompute disagrees with the model's own qkv output at layers "
                                       f"{bad.tolist()} (max rel {layout_rel.max():.3e}): layout error")
    finally:
        for hd in handles:
            hd.remove()
    if not (np.all(fired_ln == n_batches) and np.all(fired_qkv == n_batches)) or cur:
        raise RuntimeError(f"hook counts ln={fired_ln.tolist()} qkv={fired_qkv.tolist()} for "
                           f"{n_batches} batches; {len(cur)} unmatched")
    if _attn_digest(model) != before:
        raise RuntimeError("fit_s2norm modified the model's attention parameters")

    n_doc = int(ids.shape[0])
    qn = np.ascontiguousarray(qsum / n_doc)
    kn = np.ascontiguousarray(ksum / n_doc)
    if not (np.all(np.isfinite(qn)) and np.all(np.isfinite(kn))):
        raise RuntimeError("non-finite S2 norms")
    print(f"[s2norm] {n_doc} docs x {ids.shape[1]} tokens; attention-input check max err "
          f"{max(e for e, _ in pos):.1e} (tol {ATTN_INPUT_TOL:g}*scale); post_attention_layernorm "
          f"negative control raised ({neg_msg.split(':')[0]}); qkv layout max rel {layout_rel.max():.2e}")
    return {"q": qn, "k": kn, "score": qn * kn, "n_tok": int(n_doc * ids.shape[1])}

def _check_s2norm(s2norm, L, H, h):
    if not isinstance(s2norm, dict):
        raise TypeError(f"s2norm must be a dict, got {type(s2norm).__name__}")
    keys = set(s2norm)
    missing = [k for k in S2NORM_KEYS_REQUIRED if k not in keys]
    extra = keys - set(S2NORM_KEYS_REQUIRED) - set(S2NORM_KEYS_OPTIONAL)
    if missing or extra:
        raise KeyError(f"s2norm keys: missing {missing}, unexpected {sorted(extra)}")
    out = {}
    for k in ("q", "k"):
        a = s2norm[k]
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != (L, H, h):
            raise ValueError(f"s2norm[{k!r}] must be float64 ndarray {(L, H, h)}, got "
                             f"{type(a).__name__} {getattr(a, 'dtype', None)} {getattr(a, 'shape', None)}")
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"s2norm[{k!r}] has non-finite or negative entries")
        out[k] = np.ascontiguousarray(a)
    n_tok = _as_int("s2norm['n_tok']", s2norm["n_tok"])
    if n_tok <= 0:
        raise ValueError(f"s2norm['n_tok'] = {n_tok}")
    if "score" in s2norm:
        sc = np.asarray(s2norm["score"])
        if sc.shape != (L, H, h) or not np.array_equal(sc, out["q"] * out["k"]):
            raise ValueError("s2norm['score'] is not q * k")
    return out["q"], out["k"], n_tok

def s2norm_digest(s2norm):
    q, k = np.ascontiguousarray(s2norm["q"], dtype=np.float64), np.ascontiguousarray(s2norm["k"], dtype=np.float64)
    h = hashlib.sha256(b"s2norm/v1|")
    h.update(f"{q.shape}|{int(s2norm['n_tok'])}|".encode())
    h.update(q.tobytes())
    h.update(k.tobytes())
    return h.hexdigest()

def ids_sha256(ids):
    a = ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids)
    if a.dtype.kind not in "iu":
        raise TypeError(f"ids must be integer token ids, got {a.dtype}")
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()

def model_param_digest(model):
    h = hashlib.sha256(b"model-params/v1|")
    for name, p in model.named_parameters():
        t = p.detach().cpu().contiguous()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16
                 else t.float().numpy().tobytes())
    return h.hexdigest()

def model_tag(model):
    name = str(getattr(model.config, "_name_or_path", "") or "")
    tag = name.replace("\\", "/").rstrip("/").split("/")[-1]
    if not tag:
        raise ValueError("model.config._name_or_path is empty; pass the cache path explicitly")
    return tag

def s2norm_cache_name(model, ids):
    return f"s2norm_{model_tag(model)}_fit_{ids_sha256(ids)[:16]}.npz"

def _doc_hash_set(x, name):
    if x is None:
        return None
    s = sorted(str(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x))
    if len(s) != len(set(s)) or not all(len(v) == 64 for v in s):
        raise ValueError(f"{name} must be distinct sha256 hex digests")
    return s

def save_s2norm_cache(path, s2norm, model, ids, *, fit_doc_sha256=None, eval_doc_sha256=None):
    import dt_mla_arch
    L, H, d_k, dm, r = _geom(model)
    qn, kn, n_tok = _check_s2norm(s2norm, L, H, r // 2)
    ids_t = _check_ids(model, ids, None)
    if n_tok != int(ids_t.numel()):
        raise ValueError(f"s2norm n_tok={n_tok} but the ids hold {int(ids_t.numel())} tokens")
    fit_h = _doc_hash_set(fit_doc_sha256, "fit_doc_sha256")
    ev_h = _doc_hash_set(eval_doc_sha256, "eval_doc_sha256")
    if fit_h is not None and ev_h is not None and set(fit_h) & set(ev_h):
        raise ValueError(f"fit and eval documents overlap ({len(set(fit_h) & set(ev_h))} shared): leakage")
    rec = {
        "schema": S2NORM_CACHE_SCHEMA, "fit_version": S2NORM_FIT_VERSION, "model_tag": model_tag(model),
        "ids_sha256": ids_sha256(ids_t), "ids_shape": list(ids_t.shape),
        "teacher_attn_sha256": dt_mla_arch.teacher_attn_digest(model),
        "model_param_sha256": model_param_digest(model),
        "s2norm_sha256": s2norm_digest({"q": qn, "k": kn, "n_tok": n_tok}),
    }
    arrays = {"q": qn, "k": kn, "score": qn * kn, "n_tok": np.array(n_tok, dtype=np.int64),
              "ids_shape": np.array(rec["ids_shape"], dtype=np.int64)}
    arrays.update({k: np.array(v) for k, v in rec.items() if k != "ids_shape"})
    if fit_h is not None:
        arrays["fit_doc_sha256"] = np.array(fit_h)
    if ev_h is not None:
        arrays["eval_doc_sha256"] = np.array(ev_h)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.abspath(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    rec["n_tok"] = n_tok
    rec["fit_doc_sha256_n"] = None if fit_h is None else len(fit_h)
    rec["eval_doc_sha256_n"] = None if ev_h is None else len(ev_h)
    print(f"[s2norm cache] wrote {path}: ids {rec['ids_sha256'][:16]} {rec['ids_shape']}, teacher attn "
          f"{rec['teacher_attn_sha256'][:16]}, params {rec['model_param_sha256'][:16]}, s2norm "
          f"{rec['s2norm_sha256'][:16]}")
    return rec

def load_s2norm_cache(path, model, ids, *, fit_doc_sha256=None, eval_doc_sha256=None):
    import dt_mla_arch
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no S2 cache at {path}")
    L, H, d_k, dm, r = _geom(model)
    ids_t = _check_ids(model, ids, None)
    with np.load(path, allow_pickle=False) as z:
        got = {k: z[k] for k in z.files}

    def s(k):
        if k not in got:
            raise ValueError(f"S2 cache {path}: missing {k!r}")
        return str(got[k].item())

    bad = []
    if s("schema") != S2NORM_CACHE_SCHEMA:
        bad.append(f"schema {s('schema')!r} != {S2NORM_CACHE_SCHEMA!r}")
    if s("fit_version") != S2NORM_FIT_VERSION:
        bad.append(f"fit code tag {s('fit_version')!r} != this module's {S2NORM_FIT_VERSION!r}")
    if s("ids_sha256") != ids_sha256(ids_t) or list(got["ids_shape"].tolist()) != list(ids_t.shape):
        bad.append(f"fit ids differ: cache {s('ids_sha256')[:16]} {got['ids_shape'].tolist()}, given "
                   f"{ids_sha256(ids_t)[:16]} {list(ids_t.shape)}")
    if s("teacher_attn_sha256") != dt_mla_arch.teacher_attn_digest(model):
        bad.append("teacher_attn_digest differs: a different (or already converted) model")
    if s("model_param_sha256") != model_param_digest(model):
        bad.append("full-parameter digest differs: a different model (S2 depends on every layer below)")
    if bad:
        raise ValueError(f"S2 cache {path} REFUSED: " + "; ".join(bad))
    out = {"q": np.ascontiguousarray(got["q"]), "k": np.ascontiguousarray(got["k"]),
           "n_tok": int(got["n_tok"].item())}
    if "score" in got:
        out["score"] = np.ascontiguousarray(got["score"])
    qn, kn, n_tok = _check_s2norm(out, L, H, r // 2)
    if s2norm_digest(out) != s("s2norm_sha256"):
        raise ValueError(f"S2 cache {path} REFUSED: its arrays hash to {s2norm_digest(out)[:16]}, the "
                         f"record says {s('s2norm_sha256')[:16]} (edited or corrupted)")
    if n_tok != int(ids_t.numel()):
        raise ValueError(f"S2 cache {path} REFUSED: n_tok {n_tok} != ids.numel() {int(ids_t.numel())}")
    cf = got["fit_doc_sha256"].tolist() if "fit_doc_sha256" in got else None
    ce = got["eval_doc_sha256"].tolist() if "eval_doc_sha256" in got else None
    if cf is not None and ce is not None and set(cf) & set(ce):
        raise ValueError(f"S2 cache {path} REFUSED: its fit and eval document sets overlap")
    for nm, given, cached in (("fit_doc_sha256", fit_doc_sha256, cf), ("eval_doc_sha256", eval_doc_sha256, ce)):
        if given is None:
            continue
        if cached is None:
            raise ValueError(f"S2 cache {path} REFUSED: {nm} was passed but the cache records none")
        if _doc_hash_set(given, nm) != sorted(cached):
            raise ValueError(f"S2 cache {path} REFUSED: {nm} differs from the cached set")
    print(f"[s2norm cache] loaded {os.path.basename(path)}: ids {s('ids_sha256')[:16]}, s2norm "
          f"{s('s2norm_sha256')[:16]}, n_tok {n_tok}; model and code tag match")
    return {"q": qn, "k": kn, "score": qn * kn, "n_tok": n_tok}

def select_pairs(score, w):
    score = np.asarray(score)
    if score.dtype != np.float64 or score.ndim != 2:
        raise ValueError(f"score must be float64 (H, planes), got {score.dtype} {score.shape}")
    if not np.all(np.isfinite(score)):
        raise ValueError("non-finite score")
    H, h = score.shape
    w = _as_int("w", w)
    if w % 2 or w < 2 or w > 2 * H * h:
        raise ValueError(f"w={w} must be even and in [2, {2 * H * h}] (D4: w = 0 is outside the method)")
    m = w // 2
    head = np.repeat(np.arange(H, dtype=np.int64), h)
    plane = np.tile(np.arange(h, dtype=np.int64), H)
    s = score.reshape(-1)
    order = np.lexsort((plane, head, -s))
    kept = order[:m]
    last_kept = float(s[order[m - 1]])
    first_dropped = float(s[order[m]]) if m < H * h else None
    if first_dropped is None:
        rel_gap = math.inf
    elif last_kept == 0.0:
        rel_gap = 0.0
    else:
        rel_gap = (last_kept - first_dropped) / abs(last_kept)
    hh, pp = head[kept], plane[kept]
    srt = np.lexsort((pp, hh))
    hh, pp = np.ascontiguousarray(hh[srt]), np.ascontiguousarray(pp[srt])
    k_i = np.bincount(hh, minlength=H).astype(np.int64)
    for i in range(H):
        own = np.lexsort((np.arange(h), -score[i]))[:k_i[i]]
        if set(own.tolist()) != set(pp[hh == i].tolist()):
            raise AssertionError(f"head {i}: kept planes {sorted(pp[hh == i].tolist())} are not a prefix "
                                 f"of its own ranking {own.tolist()}")
    return hh, pp, {"rel_gap": rel_gap, "last_kept": last_kept, "first_dropped": first_dropped,
                    "k_i": k_i}

def _svd_f32_diag(stack, Vh64, S64, d_c):
    _, _, V32 = torch.linalg.svd(torch.from_numpy(stack).to(torch.float32), full_matrices=False)
    V32 = V32[:d_c].double().numpy()
    V64 = Vh64[:d_c]
    R = V32.T - V64.T @ (V64 @ V32.T)
    sin_max = float(np.linalg.norm(R, 2)) if R.size else 0.0
    e64 = float((S64[:d_c] ** 2).sum())
    e32 = float(((stack @ V32.T) ** 2).sum())
    return {"f32_max_principal_angle_deg": math.degrees(math.asin(min(1.0, sin_max))),
            "f32_energy_rel_diff": (e64 - e32) / e64}

def _build_layer(Qt, Kt, Vt, score, *, w, d_c, r, f32_diag=True):
    H, d_k, n1 = Qt.shape
    dm = n1 - 1
    h = r // 2
    if Kt.shape != Qt.shape or Vt.shape != Qt.shape:
        raise ValueError(f"shapes Qt {Qt.shape} Kt {Kt.shape} Vt {Vt.shape}")
    for nm, a in (("Qt", Qt), ("Kt", Kt), ("Vt", Vt)):
        if a.dtype != np.float64:
            raise ValueError(f"{nm} must be float64, got {a.dtype}")
    hh, pp, sel = select_pairs(score, w)
    m = w // 2

    WQ = Qt.reshape(H * d_k, n1)
    WK = Kt.reshape(H * d_k, n1)
    WV = Vt.reshape(H * d_k, n1)
    rot_a = d_k * hh + pp
    rot_b = rot_a + h
    rows_rot = np.concatenate([rot_a, rot_b])
    notS = np.setdiff1d(np.arange(H * d_k, dtype=np.int64), rows_rot)
    D = H * d_k - w
    if notS.shape != (D,):
        raise AssertionError(f"complement has {notS.size} rows, expected {D}")

    W_KR = WK[rows_rot]
    W_QR = WQ[rows_rot]
    W_UQ = WQ[notS]
    k_i = sel["k_i"]
    dC = (d_k - 2 * k_i).astype(np.int64)
    if int(dC.sum()) != D or np.any(dC < 0):
        raise AssertionError(f"dC {dC.tolist()} does not sum to D={D}")

    stack = np.ascontiguousarray(np.concatenate([WK[notS, :dm], WV[:, :dm]], axis=0))
    if not 1 <= d_c <= min(stack.shape):
        raise ValueError(f"d_c={d_c} must be in [1, {min(stack.shape)}] (the stack is {stack.shape})")
    U, S, Vh = np.linalg.svd(stack, full_matrices=False)
    up = U[:, :d_c] * S[:d_c]
    W_DKV = np.concatenate([Vh[:d_c], np.zeros((d_c, 1))], axis=1)
    W_UK = np.concatenate([up[:D], WK[notS, dm:]], axis=1)
    W_UV = np.concatenate([up[D:], WV[:, dm:]], axis=1)

    Psi = np.zeros((H, r, w), dtype=np.float64)
    s_idx = np.arange(m)
    Psi[hh, pp, s_idx] = 1.0
    Psi[hh, pp + h, s_idx + m] = 1.0

    lay = {
        "slot_plane": np.concatenate([pp, pp]).astype(np.int64),
        "slot_head": np.concatenate([hh, hh]).astype(np.int64),
        "dC": dC,
        "W_DKV": W_DKV, "W_KR": W_KR, "W_UQ": W_UQ, "W_QR": W_QR,
        "W_UK": W_UK, "W_UV": W_UV, "Psi": Psi,
    }
    lay = {k: np.ascontiguousarray(v) for k, v in lay.items()}

    tail = float((S[d_c:] ** 2).sum())
    diag = {
        "kept": [[int(a), int(b)] for a, b in zip(hh, pp)],
        "k_i": k_i.tolist(),
        "rel_gap": sel["rel_gap"], "last_kept": sel["last_kept"], "first_dropped": sel["first_dropped"],
        "sv_cut": float(S[d_c - 1]), "sv_next": float(S[d_c]) if d_c < S.size else None,
        "sv_cut_ratio": float(S[d_c - 1] / S[d_c]) if d_c < S.size and S[d_c] > 0 else None,
        "energy_kept_frac": float(1.0 - tail / float((S ** 2).sum())),
    }
    if f32_diag:
        diag.update(_svd_f32_diag(stack, Vh, S, d_c))
    return lay, diag

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    if fisher is not None or sigma is not None:
        raise ValueError("the MHA2MLA initialiser uses neither fisher nor sigma; the dispatch must not "
                         "pass them")
    if s2norm is None:
        raise ValueError("the MHA2MLA initialiser needs s2norm (fit_s2norm on the fit split)")
    L, H, d_k, dm, r = _geom(model)
    h = r // 2
    w, d_c, G = _as_int("w", w), _as_int("d_c", d_c), _as_int("G", G, allow_none=True)
    B = w + d_c
    if w % 2 or w < 2 or w > H * r:
        raise ValueError(f"w={w} must be even and in [2, {H * r}]")
    if d_c < 1:
        raise ValueError(f"d_c={d_c}")
    if G is not None and B != 2 * d_k * G:
        raise ValueError(f"w + d_c = {B} != 2 d_h G = {2 * d_k * G}")
    scale = mla_scale_of(w, d_k)
    scale_txt = "1/8" if d_k == 64 else f"{d_k}^-1/2"
    qn, kn, n_tok = _check_s2norm(s2norm, L, H, h)
    digest = s2norm_digest({"q": qn, "k": kn, "n_tok": n_tok})

    before = _attn_digest(model)
    layers_out, diags = [], []
    for l, layer in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, _ = gc.read_layer(layer, H, d_k, d_k, dm)
        lay, dg = _build_layer(Qt, Kt, Vt, qn[l] * kn[l], w=w, d_c=d_c, r=r)
        if dg["rel_gap"] < REL_GAP_WARN:
            print(f"[mha2mla] WARNING layer {l}: selection cut is a near-tie, rel_gap = {dg['rel_gap']:.2e} "
                  f"< {REL_GAP_WARN:g} (last kept {dg['last_kept']:.6g}, first dropped "
                  f"{dg['first_dropped']:.6g})")
        layers_out.append(lay)
        diags.append(dg)
    if _attn_digest(model) != before:
        raise RuntimeError("build_weights modified the model's attention parameters")

    kept_planes = np.concatenate([lay["slot_plane"][: w // 2] for lay in layers_out])
    k_all = np.array([dg["k_i"] for dg in diags])
    plane_hist = np.bincount(kept_planes, minlength=h).tolist()
    ki_hist = np.bincount(k_all.reshape(-1), minlength=h + 1).tolist()
    gaps = np.array([dg["rel_gap"] for dg in diags])
    meta = {
        "pin": PIN,
        "plane_hist": plane_hist,
        "k_i_hist": ki_hist,
        "head_layers_without_rope": int((k_all == 0).sum()),
        "min_rel_gap": float(gaps.min()), "min_rel_gap_layer": int(gaps.argmin()),
        "layers": diags,
    }
    if "f32_max_principal_angle_deg" in diags[0]:
        ang = np.array([dg["f32_max_principal_angle_deg"] for dg in diags])
        meta["f32_max_principal_angle_deg"] = float(ang.max())
        meta["f32_max_principal_angle_layer"] = int(ang.argmax())
    print(f"[mha2mla] w={w} d_c={d_c} G={G}: kept pairs per plane {plane_hist}; per-head k_i histogram "
          f"{ki_hist}; head-layers without RoPE {meta['head_layers_without_rope']}/{L * H}; min rel_gap "
          f"{meta['min_rel_gap']:.2e} (layer {meta['min_rel_gap_layer']}); f32-vs-f64 SVD max principal "
          f"angle {meta.get('f32_max_principal_angle_deg', float('nan')):.3g} deg (layer "
          f"{meta.get('f32_max_principal_angle_layer')})")

    return {
        "schema": SCHEMA, "arm": ARM, "version": VERSION, "rotary_mode": ROTARY_MODE,
        "G": G, "B": int(B), "w": int(w), "d_c": int(d_c),
        "L": int(L), "H": int(H), "d_h": int(d_k), "dm": int(dm), "r": int(r),
        "provenance": {
            "fisher_sha256": None, "sigma_sha256": None, "s2norm_sha256": digest,
            "text": (f"MHA2MLA ({PIN}), adapted: S2 partial RoPE, global top w/2 (head, plane) by "
                     f"E|q_ip| E|k_ip| (input_layernorm-hooked, f64, n_tok={n_tok}); kept q/k rows copied "
                     f"with bias; joint SVD in f64 of the bias-free [W_K[notS]; W_V]; W_DKV = Vh[:d_c] "
                     f"(no bias); up = U S; biases b_K[notS], b_V on the up-projections; scale {scale_txt}."),
        },
        "bias_set": list(BIAS_SET),
        "mla_scale": scale,
        "layers": layers_out,
        "_meta": meta,
    }
