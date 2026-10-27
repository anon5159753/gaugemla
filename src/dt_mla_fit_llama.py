import hashlib
import os

import numpy as np
import torch

import gam_core as gc
import gam_core_llama
import kvlab

PQ = os.path.expanduser(os.path.join(
    "~", ".cache", "huggingface", "hub", "datasets--NeelNanda--pile-10k", "snapshots",
    "127bfedcd5047750df5ccf3a12979a47bfa0bafa", "data",
    "train-00000-of-00001-4746b8785c874cc7.parquet"))
CACHE_ROOT = os.path.expanduser(os.path.join("~", ".cache", "kvtying"))
CACHE = os.path.join(CACHE_ROOT, "llama160m")
SEED, SEQ = 20260912, 512
FIT_VERSION = "fit_llama/1 (2026-10-01)"

def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def corpus(tok, n_fit, n_eval, seed=SEED, seq=SEQ):
    import pyarrow.parquet as pq

    texts = pq.read_table(PQ, columns=["text"]).column("text").to_pylist()
    rng = np.random.default_rng(seed)
    take, seen = [], set()
    for j in rng.permutation(len(texts)):
        t = tok(texts[int(j)], return_tensors="pt").input_ids[0]
        if t.shape[0] < seq + 1:
            continue
        ids = t[:seq + 1]
        h = _sha(ids.numpy())
        if h in seen:
            continue
        seen.add(h)
        take.append((int(j), ids, h))
        if len(take) == n_fit + n_eval:
            break
    if len(take) < n_fit + n_eval:
        raise RuntimeError(f"only {len(take)} documents of >= {seq + 1} tokens under this tokenizer; "
                           f"{n_fit + n_eval} were asked for")
    fit, ev = take[:n_fit], take[n_fit:]
    fh, eh = [x[2] for x in fit], [x[2] for x in ev]
    if set(fh) & set(eh):
        raise RuntimeError("fit and eval documents overlap: leakage")
    meta = {"seed": int(seed), "seq": int(seq), "n_fit": n_fit, "n_eval": n_eval,
            "fit_doc_sha256": fh, "eval_doc_sha256": eh,
            "fit_rows": [x[0] for x in fit], "eval_rows": [x[0] for x in ev],
            "fit_version": FIT_VERSION}
    return torch.stack([x[1] for x in fit]), torch.stack([x[1] for x in ev]), meta

def fit_sigma(model, ids, log=None):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    SIG = [np.zeros((dm + 1, dm + 1)) for _ in range(L)]
    ntok = [0]
    fired = np.zeros(L, dtype=np.int64)

    def hook(li):
        def fn(_m, _i, out):
            x = out.detach().to(torch.float64).reshape(-1, dm)
            xa = torch.cat([x, x.new_ones(x.shape[0], 1)], dim=1)
            SIG[li] += (xa.T @ xa).numpy()
            fired[li] += 1
            if li == 0:
                ntok[0] += x.shape[0]
        return fn

    hs = [lay.input_layernorm.register_forward_hook(hook(li))
          for li, lay in enumerate(kvlab.get_layers(model))]
    model.eval()
    try:
        with torch.no_grad():
            for d in range(ids.shape[0]):
                model(ids[d:d + 1, :SEQ], use_cache=False)
                if log and (d + 1) % 8 == 0:
                    log("    sigma %d/%d docs" % (d + 1, ids.shape[0]))
    finally:
        for h in hs:
            h.remove()
    if not np.all(fired == ids.shape[0]):
        raise RuntimeError(f"sigma hooks fired {fired.tolist()} times for {ids.shape[0]} documents")
    return [np.ascontiguousarray(S / ntok[0]) for S in SIG]

def fit_fisher(model, ids, log=None):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    GK = np.zeros((L, H, d_k, d_k))
    GV = np.zeros((L, H, d_v, d_v))
    got = {}

    def hook(li, which, d):
        def fn(_m, _gi, go):
            got[(li, which)] = go[0].detach().reshape(go[0].shape[0], go[0].shape[1], H, d).double()
        return fn

    hs = []
    for li, lay in enumerate(kvlab.get_layers(model)):
        a = gam_core_llama_attn(lay)
        hs.append(a.k_proj.register_full_backward_hook(hook(li, "k", d_k)))
        hs.append(a.v_proj.register_full_backward_hook(hook(li, "v", d_v)))
    hs.append(model.get_input_embeddings().register_forward_hook(
        lambda _m, _i, o: o.requires_grad_(True)))
    model.eval()
    ntok = 0
    try:
        for d in range(ids.shape[0]):
            chunk = ids[d:d + 1, :SEQ]
            model.zero_grad(set_to_none=True)
            model(chunk, labels=chunk).loss.backward()
            if len(got) != 2 * L:
                raise RuntimeError(f"{len(got)} of {2 * L} k/v backward hooks fired")
            for li in range(L):
                gk, gv = got[(li, "k")].numpy(), got[(li, "v")].numpy()
                GK[li] += np.einsum("bthi,bthj->hij", gk, gk)
                GV[li] += np.einsum("bthi,bthj->hij", gv, gv)
            ntok += chunk.shape[1]
            got.clear()
            if log and (d + 1) % 8 == 0:
                log("    fisher %d/%d docs" % (d + 1, ids.shape[0]))
    finally:
        for h in hs:
            h.remove()
        model.zero_grad(set_to_none=True)
    return GK / ntok, GV / ntok

def gam_core_llama_attn(layer):
    return layer.self_attn

QKV_LAYOUT_TOL = 1e-4

def fit_s2norm(model, ids, batch_size=4, tol=QKV_LAYOUT_TOL, log=None, _permute_heads=False):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    h = r // 2
    layers = kvlab.get_layers(model)
    qsum = np.zeros((L, H, h))
    ksum = np.zeros((L, H, h))
    WQ = np.zeros((L, H * r, dm))
    WK = np.zeros((L, H * r, dm))
    rows = np.array([i * d_k + j for i in range(H) for j in range(r)])
    BQ = np.zeros((L, H * r))
    BK = np.zeros((L, H * r))
    for li, lay in enumerate(layers):
        a = gam_core_llama_attn(lay)
        WQ[li] = a.q_proj.weight.detach().double().numpy()[rows]
        WK[li] = a.k_proj.weight.detach().double().numpy()[rows]
        if a.q_proj.bias is not None:
            BQ[li] = a.q_proj.bias.detach().double().numpy()[rows]
        if a.k_proj.bias is not None:
            BK[li] = a.k_proj.bias.detach().double().numpy()[rows]
    if _permute_heads:
        WQ = WQ.reshape(L, H, r, dm)[:, np.r_[1:H, 0]].reshape(L, H * r, dm).copy()
        WK = WK.reshape(L, H, r, dm)[:, np.r_[1:H, 0]].reshape(L, H * r, dm).copy()
        BQ = BQ.reshape(L, H, r)[:, np.r_[1:H, 0]].reshape(L, H * r).copy()
        BK = BK.reshape(L, H, r)[:, np.r_[1:H, 0]].reshape(L, H * r).copy()
    WQt = [torch.from_numpy(WQ[li]) for li in range(L)]
    WKt = [torch.from_numpy(WK[li]) for li in range(L)]
    BQt = [torch.from_numpy(BQ[li]) for li in range(L)]
    BKt = [torch.from_numpy(BK[li]) for li in range(L)]
    cur, rel = {}, np.zeros((L, 2))
    fired = np.zeros((L, 3), dtype=np.int64)

    def _ln(li):
        def fn(_m, _i, out):
            x = out.detach().to(torch.float64)
            q = (x @ WQt[li].T + BQt[li]).reshape(*x.shape[:-1], H, r)
            k = (x @ WKt[li].T + BKt[li]).reshape(*x.shape[:-1], H, r)
            qsum[li] += torch.hypot(q[..., :h], q[..., h:]).mean(dim=1).sum(dim=0).numpy()
            ksum[li] += torch.hypot(k[..., :h], k[..., h:]).mean(dim=1).sum(dim=0).numpy()
            cur[li] = (q, k)
            fired[li, 0] += 1
        return fn

    def _proj(li, j, name):
        def fn(_m, _i, out):
            if li not in cur:
                raise RuntimeError(f"layer {li}: {name} ran before input_layernorm's hook")
            ref = cur[li][j]
            o = out.detach().reshape(*out.shape[:-1], H, d_k).to(torch.float64)[..., :r]
            sc = max(float(ref.abs().max()), 1e-300)
            rel[li, j] = max(rel[li, j], float((o - ref).abs().max()) / sc)
            fired[li, j + 1] += 1
            if j == 1:
                cur.pop(li)
        return fn

    hs = []
    for li, lay in enumerate(layers):
        a = gam_core_llama_attn(lay)
        hs.append(lay.input_layernorm.register_forward_hook(_ln(li)))
        hs.append(a.q_proj.register_forward_hook(_proj(li, 0, "q_proj")))
        hs.append(a.k_proj.register_forward_hook(_proj(li, 1, "k_proj")))
    model.eval()
    nb = 0
    try:
        with torch.no_grad():
            for s in range(0, ids.shape[0], batch_size):
                model(ids[s:s + batch_size, :SEQ], use_cache=False)
                nb += 1
                bad = np.flatnonzero(rel.max(axis=1) > tol)
                if bad.size:
                    raise RuntimeError(f"the f64 recompute disagrees with the model's own q/k output "
                                       f"at layers {bad.tolist()} (max rel {rel.max():.3e})")
                if log:
                    log("    s2norm batch %d" % nb)
    finally:
        for hd in hs:
            hd.remove()
    if not (np.all(fired[:, 0] == nb) and np.all(fired[:, 1] == nb) and np.all(fired[:, 2] == nb)) or cur:
        raise RuntimeError(f"hook counts {fired.tolist()} for {nb} batches, {len(cur)} unpaired")
    n = float(ids.shape[0])
    qn, kn = qsum / n, ksum / n
    return {"q": qn, "k": kn, "score": qn * kn, "n_tok": int(ids.shape[0] * SEQ),
            "layout_rel_max": float(rel.max())}

def _tag(meta):
    return hashlib.sha256(("|".join(meta["fit_doc_sha256"]) + FIT_VERSION).encode()).hexdigest()[:16]

def _cached(path, fn, log):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["fit_version"]) != FIT_VERSION:
            raise RuntimeError(f"{path} was written by {z['fit_version']!s}, not {FIT_VERSION}")
        log(f"  reusing {path}")
        return {k: z[k] for k in z.files if k != "fit_version"}
    out = fn()
    np.savez(path, fit_version=np.array(FIT_VERSION), **out)
    log(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return out

def load_or_fit(model, fit_ids, meta, log=print, cache_dir=None, cache_root=None):
    root = CACHE_ROOT if cache_root is None else cache_root
    cache = CACHE if (cache_dir is None and cache_root is None) else os.path.join(root, cache_dir or "")
    if cache_root is not None and not os.path.isdir(cache_root):
        raise ValueError(f"cache_root {cache_root!r} does not exist. It is passed explicitly, so it " + "is not created here: on Colab a missing Drive path means the mount did not complete.")
    os.makedirs(cache, exist_ok=True)
    tag = _tag(meta)
    sp = os.path.join(cache, f"sigma_{tag}.npz")
    fp = os.path.join(cache, f"fisher_{tag}.npz")
    s2p = os.path.join(cache, f"s2norm_{tag}.npz")

    def _sig():
        log("  fitting Sigma ...")
        return {"SIG": np.stack(fit_sigma(model, fit_ids, log=log))}

    def _fis():
        log("  fitting the Fisher ...")
        GK, GV = fit_fisher(model, fit_ids, log=log)
        return {"GK": GK, "GV": GV}

    def _s2():
        log("  fitting S2 (MHA2MLA) ...")
        s = fit_s2norm(model, fit_ids, log=None)
        log(f"    S2 layout cross-check: max relative deviation {s['layout_rel_max']:.3e} "
            f"(bar {QKV_LAYOUT_TOL:g})")
        return {"q": s["q"], "k": s["k"], "n_tok": np.array(s["n_tok"]),
                "layout_rel_max": np.array(s["layout_rel_max"])}

    SIG = _cached(sp, _sig, log)["SIG"]
    fis = _cached(fp, _fis, log)
    s2 = _cached(s2p, _s2, log)
    return ({"GK": fis["GK"], "GV": fis["GV"],
             "SIGMA": [np.ascontiguousarray(s) for s in SIG],
             "s2norm": {"q": s2["q"], "k": s2["k"], "score": s2["q"] * s2["k"],
                        "n_tok": int(s2["n_tok"])},
             "s2norm_layout_rel_max": float(s2["layout_rel_max"])},
            [sp, fp, s2p])
