import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

SCHEMA = "regauge-kv-gqa/v1"
FLOOR_SCHEMA = "regauge/v1"
POOL_ARM = "pool"
COUNT_UP = tuple(range(1, 17))
GAUGES = {
    "identity": dict(cycle=(1,), planes="one", nonrot="one", values="one",
                     text="Omega_i = I, Gamma_i = I: the control, built and measured by the same kit under the same conditions"),
    "kvs": dict(cycle=COUNT_UP, planes="one", nonrot="one", values="one",
                text="Omega_i = c_i I (alpha_p = c_i I_2, eta = c_i I_48), Gamma_i = c_i I: c_i = i for heads i = 1 .. 16, the "
                     "same in every layer"),
    "kvd": dict(cycle=COUNT_UP, planes="one", nonrot="index", values="index",
                text="Omega_i = c_i diag(I_16, 1, 2, ..., 48) (alpha_p = c_i I_2 for every plane, eta = c_i diag(1..48)), "
                     "Gamma_i = c_i diag(1, 2, ..., 64): c_i = i for heads i = 1 .. 16, the diagonals the same in every head "
                     "and layer"),
}
MATERIAL = 0.10
CONDITIONS = {
    "identity": dict(gauge="identity", fisher="measured"),
    "kvs": dict(gauge="kvs", fisher="measured"),
    "kvd": dict(gauge="kvd", fisher="measured"),
}
CONDITION_ORDER = tuple(CONDITIONS)
PRIMARY = tuple(c for c, s in CONDITIONS.items() if s["fisher"] == "measured")
SECONDARY = tuple(c for c, s in CONDITIONS.items() if s["fisher"] != "measured")
assert CONDITION_ORDER[0] == "identity" and set(PRIMARY) == set(CONDITIONS) and SECONDARY == ()

FISHER_N_DOC = 64
FISHER_SEQ_LEN = 512
FISHER_SEED_EVAL = 20260912
FISHER_DOC_OFFSET = 2000
JITTER_SEEDS = (11, 12, 13)
GATE_DELTAS = (1, 2, 5, 17, 128, 511)

REFERENCE_STAMP = "grid-p410m_rg-pythia-2.8b-float32-50M-lr2e-05-s42-gidentity"
REFERENCE_FROM = (f"p410m_rg GPU run 255278 (NVIDIA RTX PRO 6000 Blackwell Server Edition, TF32 off): grid/regauge_test/"
                  f"p410m_rg/results/runs/{REFERENCE_STAMP}/t0_only/kd_{REFERENCE_STAMP}_pool_G<G>_t0only.jsonl")
IDENTITY_POOL_REFERENCE = {
    128: dict(G=1, kl_teacher=4.882061958312988, kl_original=4.586638912558556, ckpt_fingerprint="bd9ea72a56ab2512"),
    256: dict(G=2, kl_teacher=4.747744411230087, kl_original=4.4522433169186115, ckpt_fingerprint="a24285126e10f65c"),
    512: dict(G=4, kl_teacher=4.941674537956715, kl_original=4.641765762120485, ckpt_fingerprint="4bbfea42afc99afe"),
}
IDENTITY_TOL = 1e-3
IDENTITY_CHECK_NAME = "identity_check.json"
REFERENCE_KL_TEACHER = 0.337366871885024
EVAL_DOCS_SHA = "239226f87c30f945cddb60ac7dc0d6d2b63ff44dde1d4a3c36c5db12c75045cd"

def describe(cond):
    s = CONDITIONS[cond]
    g = GAUGES[s["gauge"]]
    return f"{g['text']}; the GQA mean pool built from the regauged checkpoint (closed-form from its weights; nothing fitted)"

def builds(cond, arm):
    if CONDITIONS[cond]["fisher"] != "measured":
        raise RuntimeError(f"{cond}: this kit has primary conditions only")
    if arm != POOL_ARM:
        raise RuntimeError(f"{arm!r} is not this kit's arm ({POOL_ARM!r}): the registry entry and regauge_gauges.py disagree")
    return True

def cells_for(cond, cells_all):
    out = [c for c in cells_all if builds(cond, c[0])]
    if not out:
        raise RuntimeError(f"{cond}: no cell of {cells_all} is built in this condition")
    return out

def arms_for(cond, arms_all):
    out = tuple(a for a in arms_all if builds(cond, a))
    if not out:
        raise RuntimeError(f"{cond}: no arm of {arms_all} is built in this condition")
    return out

def check_registry(regauge_tuple):
    if tuple(regauge_tuple) != CONDITION_ORDER:
        raise RuntimeError(f"the registry names the conditions {tuple(regauge_tuple)}, this module {CONDITION_ORDER}: "
                           f"one of them changed without the other")

def check_constants(n_doc, seq_len, seed_eval, doc_offset):
    mine = (FISHER_N_DOC, FISHER_SEQ_LEN, FISHER_SEED_EVAL, FISHER_DOC_OFFSET)
    if (n_doc, seq_len, seed_eval, doc_offset) != mine:
        raise RuntimeError(f"the init's corpus constants {(n_doc, seq_len, seed_eval, doc_offset)} differ from "
                           f"regauge_gauges' {mine}: the local gate and the run's gate would read different passages")

def head_scalar(gauge, i, Hkv, H):
    cyc = GAUGES[gauge]["cycle"]
    return float(cyc[(i // (H // Hkv)) % len(cyc)])

def _profile(kind, n):
    if kind == "one":
        return np.ones(n)
    if kind == "index":
        return np.arange(1, n + 1, dtype=np.float64)
    raise ValueError(f"unknown profile {kind!r}")

def key_ramps(gauge, d_k, r):
    s = GAUGES[gauge]
    return _profile(s["planes"], r // 2), _profile(s["nonrot"], d_k - r)

def value_ramp(gauge, d_v):
    return _profile(GAUGES[gauge]["values"], d_v)

def omegas(gauge, L, H, d_k, r, Hkv=None, d_v=None):
    import gam_core as gc
    Hkv = H if Hkv is None else Hkv
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of Hkv = {Hkv}")
    if r % 2 or r <= 0:
        raise ValueError(f"the rotary width must be even and positive, got r = {r}")
    u, e = key_ramps(gauge, d_k, r)
    Om = np.zeros((H, d_k, d_k), dtype=np.float64)
    Oi = np.zeros((H, d_k, d_k), dtype=np.float64)
    for i in range(H):
        c = head_scalar(gauge, i, Hkv, H)
        Om[i], Oi[i] = gc.omega_commutant((c * u).astype(np.complex128), np.diag(c * e) if d_k > r else None, d_k, r)
    return (np.broadcast_to(Om, (L, H, d_k, d_k)).copy(), np.broadcast_to(Oi, (L, H, d_k, d_k)).copy())

def gammas(gauge, L, H, d_v, Hkv=None, d_k=None, r=None):
    Hkv = H if Hkv is None else Hkv
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of Hkv = {Hkv}")
    g = value_ramp(gauge, d_v)
    Gm = np.zeros((H, d_v, d_v), dtype=np.float64)
    Gi = np.zeros((H, d_v, d_v), dtype=np.float64)
    for i in range(H):
        c = head_scalar(gauge, i, Hkv, H)
        Gm[i] = np.diag(c * g)
        Gi[i] = np.diag(1.0 / (c * g))
    return (np.broadcast_to(Gm, (L, H, d_v, d_v)).copy(), np.broadcast_to(Gi, (L, H, d_v, d_v)).copy())

def spec(gauge, H, d_k, r, Hkv=None, d_v=None):
    Hkv = H if Hkv is None else Hkv
    d_v = d_k if d_v is None else d_v
    s = GAUGES[gauge]
    u, e = key_ramps(gauge, d_k, r)
    g = value_ramp(gauge, d_v)
    return dict(gauge=gauge, cycle=list(s["cycle"]), profiles=dict(planes=s["planes"], nonrot=s["nonrot"], values=s["values"]),
                text=s["text"], H=H, Hkv=Hkv, d_k=d_k, d_v=d_v, r=r, planes=r // 2,
                nonrotary_rows=d_k - r, c_per_head=[head_scalar(gauge, i, Hkv, H) for i in range(H)],
                plane_diag_P=u.tolist(), nonrotary_diag_N=e.tolist(), value_diag_V=g.tolist(),
                keys="Omega_i = c_i diag(P_1 I_2, ..., P_h I_2, N_1 .. N_{d_k-r}) (SPLIT-HALF: plane p = coordinates p-1, p-1+h)",
                values="Gamma_i = c_i diag(V_1 .. V_{d_v}); W_O -> W_O Gamma_i^-1")

def omega_digest(Om):
    return hashlib.sha256(np.ascontiguousarray(Om).tobytes()).hexdigest()[:16]

def _is_pow2(x):
    m, e = np.frexp(np.asarray(x, dtype=np.float64))
    return bool(np.all((m == 0.5) | (x == 0)))

def algebra_check(gauge, model):
    import gam_core as gc
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    Hkv = getattr(model, "_gqa_kv_heads", H)
    Om, Oi = omegas(gauge, L, H, d_k, r, Hkv, d_v)
    Gm, Gi = gammas(gauge, L, H, d_v, Hkv, d_k, r)
    theta = gc.inv_freq(model)
    if len(theta) != r // 2:
        raise RuntimeError(f"{len(theta)} rotary frequencies for r = {r}")
    worst = 0.0
    for d in GATE_DELTAS:
        R = gc.rot(theta, d, d_k, r)
        nR = np.linalg.norm(R, 2)
        for i in range(H):
            O = Om[0, i]
            worst = max(worst, float(np.linalg.norm(O @ R - R @ O, 2) / (np.linalg.norm(O, 2) * nR)))
    inv = max(float(np.abs(Om[0, i] @ Oi[0, i] - np.eye(d_k)).max()) for i in range(H))
    ginv = max(float(np.abs(Gm[0, i] @ Gi[0, i] - np.eye(d_v)).max()) for i in range(H))
    nz = Om[0][Om[0] != 0]
    gz = Gm[0][Gm[0] != 0]
    out = dict(gauge=gauge, commutator_defect=worst, inverse_defect=inv, gamma_inverse_defect=ginv,
               omega_digest=omega_digest(Om), gamma_digest=omega_digest(Gm),
               entries_all_powers_of_two=_is_pow2(nz) and _is_pow2(gz), min_entry=float(min(np.abs(nz).min(), np.abs(gz).min())),
               max_entry=float(max(np.abs(nz).max(), np.abs(gz).max())),
               condition_number=float(max(np.linalg.cond(Om[0, i]) for i in range(H))),
               gamma_condition_number=float(max(np.linalg.cond(Gm[0, i]) for i in range(H))))
    if worst > 1e-12 or inv > 1e-12 or ginv > 1e-12:
        raise RuntimeError(f"{gauge} is not in the commutant on this model, or an inverse is wrong: commutator defect "
                           f"{worst:.3e}, Omega inverse defect {inv:.3e}, Gamma inverse defect {ginv:.3e}")
    return out

def attn_fingerprint(model):
    import kvlab
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        projs = ((lay.self_attn.q_proj, lay.self_attn.k_proj, lay.self_attn.v_proj, lay.self_attn.o_proj)
                 if hasattr(lay, "self_attn") else (lay.attention.query_key_value, lay.attention.dense))
        for p in projs:
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def apply(model, gauge, log=print):
    import gam_core as gc
    import kvlab
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    Hkv = getattr(model, "_gqa_kv_heads", H)
    if Hkv != H:
        raise RuntimeError(f"Hkv = {Hkv} != H = {H}: the value gauge on a GQA source is not built in this kit (Pythia only)")
    Om, Oi = omegas(gauge, L, H, d_k, r, Hkv, d_v)
    Gm, Gi = gammas(gauge, L, H, d_v, Hkv, d_k, r)
    fp0 = attn_fingerprint(model)
    for li, lay in enumerate(kvlab.get_layers(model) if gauge != "identity" else ()):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        Qh, Kh, Vh, Oh = gc.apply_gauge(Qt, Kt, Vt, Wo, Om[li], Oi[li], Gm[li], Gi[li])
        if np.array_equal(Vh, Vt) or np.array_equal(Oh, Wo):
            raise RuntimeError(f"layer {li}: the value gauge left the values or the output map unchanged")
        gc.write_layer(lay, Qh, Kh, Vh, Oh, H, d_k, d_v, dm)
    fp1 = attn_fingerprint(model)
    if (gauge == "identity") != (fp0 == fp1):
        raise RuntimeError(f"{gauge}: attention fingerprint {fp0} -> {fp1}; the identity must leave it and every other "
                           f"gauge must move it")
    if log:
        log(f"@@REGAUGE@@ applied {gauge} to every layer's keys, queries, values and output map (biases included): "
            f"attention {fp0} -> {fp1}")
    return fp0, fp1

def jitter(model, seed, eps):
    import gam_core as gc
    import kvlab
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    rng = np.random.default_rng(seed)
    for lay in kvlab.get_layers(model):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        j = lambda A: A * (1.0 + eps * rng.normal(size=A.shape))
        gc.write_layer(lay, j(Qt), j(Kt), j(Vt), j(Wo), H, d_k, d_v, dm)

def _summ(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return dict(min=float(xs.min()), median=float(np.median(xs)), max=float(xs.max()))

def pool_moved(gauge, model, Gs=(1, 2, 4), log=print, strict=True):
    import gam_core as gc
    import kvlab
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    Hkv = getattr(model, "_gqa_kv_heads", H)
    Om, Oi = omegas(gauge, L, H, d_k, r, Hkv, d_v)
    Gm, Gi = gammas(gauge, L, H, d_v, Hkv, d_k, r)
    sides = ("keys", "values")
    per = {s: {G: dict(pool_relF=[], damage=[], damage_regauged=[], damage_rel_change=[]) for G in Gs} for s in sides}
    for li, lay in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        Kg = np.einsum("hij,hjn->hin", Om[li], Kt)
        Qg = np.einsum("hji,hjn->hin", Oi[li], Qt)
        Vg = np.einsum("hij,hjn->hin", Gm[li], Vt)
        Og = np.einsum("hmj,hji->hmi", Wo, Gi[li])
        reader = {"keys": (np.einsum("hin,hjn->hij", Qt, Qt), np.einsum("hin,hjn->hij", Qg, Qg)),
                  "values": (np.einsum("hmi,hmj->hij", Wo, Wo), np.einsum("hmi,hmj->hij", Og, Og))}
        maps = {"keys": (Kt, Kg), "values": (Vt, Vg)}
        for s in sides:
            A, Ag = maps[s]
            R0, Rg = reader[s]
            for G in Gs:
                num = den = D = Dg = 0.0
                for idx in kvlab.groups_of(H, G):
                    T, Tg = A[idx].mean(0), Ag[idx].mean(0)
                    num += float(((Tg - T) ** 2).sum())
                    den += float((T ** 2).sum())
                    for i in idx:
                        Bm, Bg = T - A[i], Tg - Ag[i]
                        D += float((R0[i] * (Bm @ Bm.T)).sum())
                        Dg += float((Rg[i] * (Bg @ Bg.T)).sum())
                per[s][G]["pool_relF"].append(np.sqrt(num / den))
                per[s][G]["damage"].append(D)
                per[s][G]["damage_regauged"].append(Dg)
                per[s][G]["damage_rel_change"].append(abs(Dg - D) / D)
    out = dict(gauge=gauge, threshold=MATERIAL, by_side={})
    material, exact_zero = True, True
    for s in sides:
        out["by_side"][s] = {}
        for G in Gs:
            p = per[s][G]
            st = dict(G=G, groups=G, heads_per_group=H // G, pool_relF=_summ(p["pool_relF"]),
                      damage_rel_change=_summ(p["damage_rel_change"]),
                      damage_per_layer=[float(x) for x in p["damage"]],
                      damage_regauged_per_layer=[float(x) for x in p["damage_regauged"]],
                      pool_relF_per_layer=[float(x) for x in p["pool_relF"]],
                      damage_rel_change_per_layer=[float(x) for x in p["damage_rel_change"]])
            st["material"] = st["pool_relF"]["median"] >= MATERIAL and st["damage_rel_change"]["median"] >= MATERIAL
            material &= st["material"]
            exact_zero &= max(p["pool_relF"]) == 0.0 and max(p["damage_rel_change"]) == 0.0
            out["by_side"][s][G] = st
            if log:
                verdict = ("exactly 0 (the identity)" if gauge == "identity" and st["pool_relF"]["max"] == 0.0
                           and st["damage_rel_change"]["max"] == 0.0 else "MATERIAL" if st["material"] else "WEAK")
                log(f"@@REGAUGE@@   pool moved [{gauge}] {s:<6} G={G} ({H // G} heads/group): pool rel. Frobenius change "
                    f"min/med/max {st['pool_relF']['min']:.3f} / {st['pool_relF']['median']:.3f} / {st['pool_relF']['max']:.3f}; "
                    f"pooled damage rel. change {st['damage_rel_change']['min']:.3f} / {st['damage_rel_change']['median']:.3f} / "
                    f"{st['damage_rel_change']['max']:.3f}  -> {verdict} (bar {MATERIAL:.0%} on the medians)")
    out["material"], out["exact_zero"] = bool(material), bool(exact_zero)
    out["refuses"] = False
    if gauge == "identity":
        if not exact_zero:
            raise RuntimeError("the identity gauge moved a pool or its damage; the pool_moved computation is wrong")
    elif log:
        log(f"@@REGAUGE@@   pool moved [{gauge}] is a REPORT, never a refusal (the authors: \"Keep c_i = i, report only\"; "
            f"p410m_rgkv's gauges, unchanged; the pool's KL is the measurement): {'every' if material else 'NOT every'} side "
            f"and G clears the {MATERIAL:.0%} median")
    return out

def _rowspace(W, rtol=1e-10):
    U, s, Vt = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
    k = int((s > s[0] * rtol).sum()) if s.size else 0
    return Vt[:k].T

def subspace_distance(W1, W2):
    B1, B2 = _rowspace(W1), _rowspace(W2)
    if B1.shape != B2.shape:
        raise ValueError(f"row spaces of different rank or dimension: {B1.shape} vs {B2.shape}")
    s = np.clip(np.linalg.svd(B1.T @ B2, compute_uv=False), 0.0, 1.0)
    th = np.degrees(np.arccos(s))
    return dict(k=int(B1.shape[1]), mean_deg=float(th.mean()), max_deg=float(th.max()),
                chordal=float(np.sqrt(np.mean(1.0 - s ** 2))))

LATENT_KEYS = {
    "gqa": {"key": "k_proj.weight", "value": "v_proj.weight"},
}

def _layer_index(key):
    import re
    m = re.search(r"\.layers\.(\d+)\.", key)
    if not m:
        raise ValueError(f"no layer index in {key!r}")
    return int(m.group(1))

def subspace_vs_identity(ckpt_dir, identity_dir, log=None):
    import dt_checkpoint
    meta = json.load(open(os.path.join(ckpt_dir, "kv_tying_meta.json"), encoding="utf-8"))
    arch = meta["arch"]
    if arch not in LATENT_KEYS:
        raise RuntimeError(f"{ckpt_dir}: arch {arch!r} has no latent this statistic knows")
    if not os.path.exists(os.path.join(identity_dir, "kv_tying_meta.json")):
        raise RuntimeError(f"{identity_dir} is not a built checkpoint: the identity condition must be built first")
    s1 = dt_checkpoint._load_raw_state_dict(ckpt_dir)
    s2 = dt_checkpoint._load_raw_state_dict(identity_dir)
    out = {}
    for name, suffix in LATENT_KEYS[arch].items():
        keys = sorted((k for k in s1 if k.endswith("." + suffix)), key=_layer_index)
        if not keys:
            raise RuntimeError(f"{ckpt_dir}: no parameter ends with {suffix!r}")
        per = []
        for k in keys:
            if k not in s2:
                raise RuntimeError(f"{identity_dir} lacks {k}")
            per.append(subspace_distance(s1[k].detach().double().numpy(), s2[k].detach().double().numpy()))
        out[name] = dict(k=per[0]["k"], chordal=_summ([p["chordal"] for p in per]), mean_deg=_summ([p["mean_deg"] for p in per]),
                         max_deg=_summ([p["max_deg"] for p in per]), per_layer=per)
        if log:
            log(f"@@REGAUGE@@   subspace vs identity [{os.path.basename(ckpt_dir)}] {name} (rank {per[0]['k']}): chordal "
                f"min/med/max {out[name]['chordal']['min']:.4f} / {out[name]['chordal']['median']:.4f} / {out[name]['chordal']['max']:.4f}; "
                f"mean angle med {out[name]['mean_deg']['median']:.2f} deg")
    return out

def pair_pass(a, b, ids, bs=2, log=print, label=""):
    import torch
    n = ids.shape[0]
    dmax, scale, tv, kl = 0.0, 0.0, [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, bs):
            chunk = ids[i:i + bs]
            la = a(chunk).logits.double()
            lb = b(chunk).logits.double()
            dmax = max(dmax, float((la - lb).abs().max()))
            scale = max(scale, float(la.abs().max()))
            pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
            tv.extend((0.5 * (pa.exp() - pb.exp()).abs().sum(-1)).mean(dim=1).tolist())
            kl.extend((pa.exp() * (pa - pb)).sum(-1).mean(dim=1).tolist())
            del la, lb, pa, pb
            if log and ((i + bs) % 16 == 0 or i + bs >= n):
                log(f"    {label:<12} {min(i + bs, n):>3}/{n} docs  max|dlogit| {dmax:.3e}  mean TV {np.mean(tv):.3e}  "
                    f"({time.time() - t0:.0f} s)")
    return dict(n_doc=int(n), seq_len=int(ids.shape[1]), max_abs_dlogit=dmax, logit_scale=scale,
                rel_max_dlogit=dmax / max(scale, 1e-30), mean_tv=float(np.mean(tv)), max_tv=float(np.max(tv)),
                mean_kl=float(np.mean(kl)), max_kl=float(np.max(kl)), secs=round(time.time() - t0, 1))

def _ids_sha(ids):
    return hashlib.sha256(np.ascontiguousarray(ids.cpu().numpy().astype("int64")).tobytes()).hexdigest()[:16]

def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def floor(orig, ids, rg_root, model_name, log=print, seeds=JITTER_SEEDS):
    import torch
    path = os.path.join(rg_root, "gate_floor.json")
    key = dict(schema=FLOOR_SCHEMA, model=model_name, ids_sha=_ids_sha(ids), n_doc=int(ids.shape[0]),
               seq_len=int(ids.shape[1]), seeds=list(seeds))
    if os.path.exists(path):
        rec = json.load(open(path, encoding="utf-8"))
        same = all(rec.get(k) == v for k, v in key.items())
        if not same:
            raise RuntimeError(f"{path} was measured for {{k: rec.get(k) for k in key}}, this process has {key}: delete "
                               f"it only if that is understood")
        log(f"@@REGAUGE@@ gate floor reused <- {path} (measured by {rec.get('measured_by')}, bar {rec['bar']:.3e} TV)")
        return rec
    eps = float(np.finfo(np.float32).eps)
    log(f"@@REGAUGE@@ measuring the gate floor on {ids.shape[0]} x {ids.shape[1]} tokens (CPU): the original run twice, "
        f"then {len(seeds)} relative-eps ({eps:.2e}) jitters of its attention weights ...")
    rerun = pair_pass(orig, orig, ids, log=log, label="rerun")
    jit = []
    for s in seeds:
        m = copy.deepcopy(orig)
        jitter(m, s, eps)
        jit.append(dict(seed=s, **pair_pass(orig, m, ids, log=log, label=f"jitter s{s}")))
        del m
    bar = 3.0 * max(j["mean_tv"] for j in jit)
    rec = dict(key, eps=eps, rerun=rerun, jitter=jit, bar=bar, bar_rule="3 x the worst jitter seed's mean TV",
               measured_by=os.environ.get("JOB_ID") or f"local pid {os.getpid()}", torch=torch.__version__,
               threads=torch.get_num_threads(), written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_json(path, rec)
    log(f"@@REGAUGE@@ floor: rerun max|dlogit| {rerun['max_abs_dlogit']:.3e} mean TV {rerun['mean_tv']:.3e}; jitter mean TV "
        f"{min(j['mean_tv'] for j in jit):.3e} .. {max(j['mean_tv'] for j in jit):.3e}, max|dlogit| "
        f"{max(j['max_abs_dlogit'] for j in jit):.3e}; bar {bar:.3e} TV -> {path}")
    return rec

def gate(cond, orig, ids, cond_root, rg_root, model_name, log=print, reuse=True):
    import gam_core as gc
    import torch
    g = CONDITIONS[cond]["gauge"]
    L, H, d_k, d_v, dm, r = gc.geometry(orig)
    Hkv = getattr(orig, "_gqa_kv_heads", H)
    alg = algebra_check(g, orig)
    path = os.path.join(cond_root, "gate.json")
    key = dict(schema=SCHEMA, condition=cond, gauge=g, model=model_name, omega_digest=alg["omega_digest"],
               gamma_digest=alg["gamma_digest"], ids_sha=_ids_sha(ids), n_doc=int(ids.shape[0]), seq_len=int(ids.shape[1]))
    fp0 = attn_fingerprint(orig)
    moved = pool_moved(g, orig, log=log)
    if reuse and os.path.exists(path):
        rec = json.load(open(path, encoding="utf-8"))
        if all(rec.get(k) == v for k, v in key.items()) and rec.get("verdict") == "PASS" and rec.get("fp_original") == fp0:
            log(f"@@REGAUGE@@ gate reused <- {path}: {cond} PASSED on {rec.get('written')} (mean TV "
                f"{rec['regauged']['mean_tv']:.3e} <= bar {rec['bar']:.3e}, max|dlogit| {rec['regauged']['max_abs_dlogit']:.3e})")
            rec["reused"] = True
            return rec
        log(f"@@REGAUGE@@ {path} exists but is not a PASS for this condition, gauge, model and passages: measuring again")
    fl = floor(orig, ids, rg_root, model_name, log=log)
    log(f"@@REGAUGE@@ gate for {cond}: {describe(cond)}")
    log(f"@@REGAUGE@@   algebra: commutator defect {alg['commutator_defect']:.1e}, inverse defects Omega {alg['inverse_defect']:.1e} / "
        f"Gamma {alg['gamma_inverse_defect']:.1e}, entries "
        f"{'all powers of two (exact in float32)' if alg['entries_all_powers_of_two'] else 'not all powers of two'}, "
        f"|entries| in [{alg['min_entry']:.4g}, {alg['max_entry']:.4g}], per-head kappa Omega {alg['condition_number']:.3g} / "
        f"Gamma {alg['gamma_condition_number']:.3g}")
    rg = copy.deepcopy(orig)
    _, fp1 = apply(rg, g, log=log)
    stats = pair_pass(orig, rg, ids, log=log, label=cond)
    del rg
    verdict = "PASS" if stats["mean_tv"] <= fl["bar"] else "FAIL"
    rec = dict(key, spec=spec(g, H, d_k, r, Hkv, d_v), algebra=alg, pool_moved=moved, fp_original=fp0, fp_regauged=fp1,
               regauged=stats, floor_rerun=fl["rerun"], floor_jitter_worst_mean_tv=max(j["mean_tv"] for j in fl["jitter"]),
               floor_jitter_max_abs_dlogit=max(j["max_abs_dlogit"] for j in fl["jitter"]), bar=fl["bar"],
               bar_rule=fl["bar_rule"], x_bar=stats["mean_tv"] / max(fl["bar"], 1e-300), verdict=verdict,
               bitwise=stats["max_abs_dlogit"] == 0.0, torch=torch.__version__, threads=torch.get_num_threads(),
               measured_by=os.environ.get("JOB_ID") or f"local pid {os.getpid()}",
               written=time.strftime("%Y-%m-%dT%H:%M:%S"), reused=False)
    _write_json(path, rec)
    log(f"@@REGAUGE@@ GATE {verdict}: {cond}: max|dlogit| {stats['max_abs_dlogit']:.3e} (logit scale {stats['logit_scale']:.3g}; "
        f"jitter's {rec['floor_jitter_max_abs_dlogit']:.3e}), mean TV {stats['mean_tv']:.3e} = {rec['x_bar']:.3g} x the bar "
        f"{fl['bar']:.3e}, mean KL {stats['mean_kl']:.3e}; rerun floor max|dlogit| {fl['rerun']['max_abs_dlogit']:.3e}, "
        f"mean TV {fl['rerun']['mean_tv']:.3e}"
        + ("; the regauged model's logits are BITWISE the original's" if rec["bitwise"] else "") + f" -> {path}")
    if verdict != "PASS":
        raise RuntimeError(f"GATE FAILED for {cond}: the regauged checkpoint does not compute the original's function "
                           f"(mean TV {stats['mean_tv']:.3e} > bar {fl['bar']:.3e}). Nothing is built from it. Report "
                           f"{path} to the authors.")
    return rec

def install_loader(kvlab_mod, gauge, model_name, expected_fp, log=print):
    if getattr(kvlab_mod.load_model, "_regauge", None) is not None:
        raise RuntimeError(f"the regauge loader is already installed ({kvlab_mod.load_model._regauge}); install it once")
    orig_load = kvlab_mod.load_model

    def load_model(name, *a, **kw):
        if name != model_name:
            raise RuntimeError(f"the regauge loader was asked for {name!r}; this kit regauges {model_name!r} only")
        model, tok = orig_load(name, *a, **kw)
        fp0, fp1 = apply(model, gauge, log=None)
        if fp1 != expected_fp:
            raise RuntimeError(f"the regauged {name} has attention fingerprint {fp1}, the gate's copy had {expected_fp}")
        log(f"@@REGAUGE@@ kvlab.load_model({name!r}) -> regauged under {gauge} (attention {fp0} -> {fp1}); every "
            f"statistic and build below reads this checkpoint")
        return model, tok

    load_model._regauge = (gauge, model_name, expected_fp)
    load_model._original = orig_load
    kvlab_mod.load_model = load_model
    log(f"@@REGAUGE@@ kvlab.load_model patched: {model_name} is served regauged under {gauge} from here on")

def gate_and_install(cond, orig, ids, cond_root, rg_root, kvlab_mod, model_name, log=print):
    rec = gate(cond, orig, ids, cond_root, rg_root, model_name, log=log)
    g = CONDITIONS[cond]["gauge"]
    _, fp1 = apply(orig, g, log=log)
    if fp1 != rec["fp_regauged"]:
        raise RuntimeError(f"regauging the fit model gave attention {fp1}, the gate's copy {rec['fp_regauged']}")
    install_loader(kvlab_mod, g, model_name, rec["fp_regauged"], log=log)
    return rec

def _check_path(rg_root):
    return os.path.join(rg_root, "identity", IDENTITY_CHECK_NAME)

def identity_check(rows, eval_docs_sha, rg_root, stamp, reference_line=None, log=print):
    problems, per = [], {}
    if eval_docs_sha != EVAL_DOCS_SHA:
        problems.append(f"the held-out passages' sha256 {eval_docs_sha[:16]} is not the reference rows' {EVAL_DOCS_SHA[:16]}")
    for B, ref in sorted(IDENTITY_POOL_REFERENCE.items()):
        r = rows.get(B)
        if r is None:
            problems.append(f"B = {B}: no identity pool row")
            continue
        if r.get("regauge") != "identity" or r.get("arm") != POOL_ARM or r.get("G") != ref["G"]:
            problems.append(f"B = {B}: the row is {r.get('regauge')!r} / {r.get('arm')!r} / G = {r.get('G')!r}, not "
                            f"identity / {POOL_ARM} / G = {ref['G']}")
            continue
        if r.get("tf32") is not False:
            problems.append(f"B = {B}: tf32 = {r.get('tf32')!r}; the reference was measured with TF32 off")
        ko, kt = float(r["kl_orig_mean"]), float(r["kl_mean"])
        d_o, d_t = ko - ref["kl_original"], kt - ref["kl_teacher"]
        fp = r.get("ckpt_fingerprint")
        ok = abs(d_o) <= IDENTITY_TOL and abs(d_t) <= IDENTITY_TOL
        per[str(B)] = dict(G=ref["G"], kl_original=ko, reference_kl_original=ref["kl_original"], minus_reference_kl_original=d_o,
                           kl_teacher=kt, reference_kl_teacher=ref["kl_teacher"], minus_reference_kl_teacher=d_t,
                           ckpt_fingerprint=fp, reference_ckpt_fingerprint=ref["ckpt_fingerprint"],
                           fingerprint_equal=fp == ref["ckpt_fingerprint"], within_tolerance=ok)
        log(f"@@REGAUGE@@ identity check B={B} (G={ref['G']}): KL_o {ko!r} - reference {ref['kl_original']!r} = {d_o:+.3e}; "
            f"KL_t {kt!r} - reference {ref['kl_teacher']!r} = {d_t:+.3e}; checkpoint {fp} "
            f"{'=' if fp == ref['ckpt_fingerprint'] else '!='} the reference's {ref['ckpt_fingerprint']} (reported, not gated) "
            f"-> {'within' if ok else 'OUTSIDE'} {IDENTITY_TOL:g}")
        if not ok:
            problems.append(f"B = {B}: KL_o {ko:.6f} vs {ref['kl_original']:.6f} ({d_o:+.3e}), KL_t {kt:.6f} vs "
                            f"{ref['kl_teacher']:.6f} ({d_t:+.3e}): beyond {IDENTITY_TOL:g}")
    rl = None
    if reference_line is not None:
        rl = dict(kl_teacher_vs_original=float(reference_line), reference=REFERENCE_KL_TEACHER,
                  minus_reference=float(reference_line) - REFERENCE_KL_TEACHER)
        log(f"@@REGAUGE@@ identity check: this run's reference line KL(teacher || original) {float(reference_line)!r} - the "
            f"key-only run's {REFERENCE_KL_TEACHER!r} = {rl['minus_reference']:+.3e} (reported, not gated)")
    rec = dict(schema=SCHEMA, kind="identity instrument check", stamp=stamp, verdict="FAIL" if problems else "PASS",
               tolerance=IDENTITY_TOL, reference_from=REFERENCE_FROM, eval_docs_sha=eval_docs_sha,
               reference_eval_docs_sha=EVAL_DOCS_SHA, per_budget=per, reference_line=rl, problems=problems,
               measured_by=os.environ.get("JOB_ID") or f"pid {os.getpid()}", written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_json(_check_path(rg_root), rec)
    log(f"@@REGAUGE@@ IDENTITY CHECK {rec['verdict']}: the identity's pool rows against the key-only kit's "
        f"(|delta| <= {IDENTITY_TOL:g} on KL_o and KL_t) -> {_check_path(rg_root)}")
    if problems:
        raise RuntimeError("the identity's pool rows do not reproduce the key-only kit's (a different model or instrument, "
                           "not rounding); kvs and kvd are not measured. Keep the log and identity_check.json:\n  "
                           + "\n  ".join(problems))
    return rec

def require_identity_check(cond, rg_root, log=print):
    if cond == "identity":
        return None
    p = _check_path(rg_root)
    if not os.path.exists(p):
        raise RuntimeError(f"{cond}: {p} does not exist. The identity condition's run process writes it after its three "
                           f"pool rows and must PASS before {cond} is measured (run identity first)")
    rec = json.load(open(p, encoding="utf-8"))
    if rec.get("schema") != SCHEMA or rec.get("verdict") != "PASS":
        raise RuntimeError(f"{cond}: {p} says schema {rec.get('schema')!r}, verdict {rec.get('verdict')!r}: only a PASSED "
                           f"identity check of this kit lets {cond} be measured")
    log(f"@@REGAUGE@@ {cond}: identity's instrument check PASSED ({rec.get('written')}, stamp {rec.get('stamp')}) <- {p}")
    return rec

def record_builds(cond, ckpt_root, cells, gate_rec, identity_root, log=print):
    g = CONDITIONS[cond]["gauge"]
    pm = {s: {str(G): dict(pool_relF=st["pool_relF"], damage_rel_change=st["damage_rel_change"], material=st["material"])
              for G, st in d.items()} for s, d in gate_rec["pool_moved"]["by_side"].items()}
    base = dict(schema=SCHEMA, condition=cond, gauge=g, fisher=CONDITIONS[cond]["fisher"], text=describe(cond),
                spec=gate_rec["spec"], omega_digest=gate_rec["omega_digest"], gamma_digest=gate_rec["gamma_digest"],
                fp_original=gate_rec["fp_original"], fp_regauged=gate_rec["fp_regauged"],
                gate=dict(verdict=gate_rec["verdict"], mean_tv=gate_rec["regauged"]["mean_tv"],
                          max_abs_dlogit=gate_rec["regauged"]["max_abs_dlogit"], bar=gate_rec["bar"],
                          bitwise=gate_rec["bitwise"], written=gate_rec["written"]),
                pool_moved=pm)
    n = 0
    for arm, G, B in cells:
        d = os.path.join(ckpt_root, f"{arm}_B{B}_t0")
        if not os.path.exists(os.path.join(d, "kv_tying_meta.json")):
            raise RuntimeError(f"{d}: no sidecar, so no checkpoint to record the gauge beside")
        p = os.path.join(d, "regauge.json")
        if os.path.exists(p):
            old = json.load(open(p, encoding="utf-8"))
            if any(old.get(k) != v for k, v in base.items()) or "subspace" not in old:
                raise RuntimeError(f"{p} records a different condition, gate or format than this process: "
                                   f"{old.get('condition')} / {old.get('omega_digest')} vs {cond} / {base['omega_digest']}")
            continue
        sub = subspace_vs_identity(d, os.path.join(identity_root, f"{arm}_B{B}_t0"), log=log)
        _write_json(p, dict(base, subspace=sub))
        n += 1
    log(f"@@REGAUGE@@ regauge.json written beside {n} of {len(cells)} checkpoints under {ckpt_root} (the rest already had it)")

def _parquet_path():
    import glob
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    c = sorted(glob.glob(os.path.join(hub, "datasets--NeelNanda--pile-10k", "snapshots", "*", "data", "*.parquet")))
    if len(c) != 1:
        raise RuntimeError(f"expected exactly one pile-10k parquet under {hub}, found {c}")
    return c[0]

def pile_split_parquet(tok, n_doc, skip, texts):
    import torch
    rng = np.random.default_rng(FISHER_SEED_EVAL)
    ids, pick, taken = [], [], 0
    for j in rng.permutation(len(texts)):
        if j < FISHER_DOC_OFFSET:
            continue
        e = tok(texts[int(j)])["input_ids"]
        if len(e) < FISHER_SEQ_LEN + 1:
            continue
        taken += 1
        if taken <= skip:
            continue
        ids.append(e[:FISHER_SEQ_LEN])
        pick.append(int(j))
        if len(ids) == n_doc:
            break
    if len(ids) != n_doc:
        raise RuntimeError(f"only {len(ids)} documents were long enough; need {n_doc}")
    h = {hashlib.sha256(texts[int(j)].encode()).hexdigest() for j in pick}
    return torch.tensor(ids, dtype=torch.long), pick, h

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="the local gate: one condition's regauged Pythia-410m against the original")
    ap.add_argument("--condition", default="kvd", choices=list(CONDITIONS))
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--ndocs", type=int, default=FISHER_N_DOC, help="calibration passages (the first n of the fit split)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True, help="a directory: gate_floor.json and <condition>/gate.json go there")
    ap.add_argument("--check-split", default=None,
                    help="an existing Fisher cache (.npz) whose eval_sha256 / fit_sha256 the parquet split must reproduce")
    ap.add_argument("--algebra-only", action="store_true", help="the commutant and inverse checks for every gauge, no passes")
    ap.add_argument("--pool-only", action="store_true", help="the pool-moved check for every gauge (writes pool_moved.json in --out), no passes")
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    ofe = os.path.normpath(os.path.join(here, "..", "..", "..", "src"))
    if not os.path.isdir(ofe):
        raise SystemExit(f"{ofe} is not the pipeline directory")
    sys.path.insert(0, ofe)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.set_num_threads(a.threads)
    import kvlab
    log = lambda *x: print(*x, flush=True)
    log(f"regauge_gauges (p410m_rgkv_gqa) local gate: condition {a.condition}, {a.model}, torch {torch.__version__}, "
        f"{torch.get_num_threads()} threads")
    model, tok = kvlab.load_model(a.model)
    if a.algebra_only or a.pool_only:
        for g in GAUGES:
            r = algebra_check(g, model)
            log(f"  {g:<9} commutator {r['commutator_defect']:.1e} inverse Omega {r['inverse_defect']:.1e} Gamma "
                f"{r['gamma_inverse_defect']:.1e} pow2 {r['entries_all_powers_of_two']} kappa Omega {r['condition_number']:.3g} "
                f"Gamma {r['gamma_condition_number']:.3g} |entries| [{r['min_entry']:.4g}, {r['max_entry']:.4g}] digests "
                f"{r['omega_digest']} / {r['gamma_digest']}")
        if a.pool_only:
            allm = {}
            for g in GAUGES:
                allm[g] = pool_moved(g, model, log=log, strict=False)
            os.makedirs(a.out, exist_ok=True)
            _write_json(os.path.join(a.out, "pool_moved.json"), allm)
            weak = [g for g, m in allm.items() if g != "identity" and not m["material"]]
            log(f"-> {os.path.join(a.out, 'pool_moved.json')}; weak gauges: {weak or 'none'}")
            return 1 if weak else 0
        return 0
    import pyarrow.parquet as pq
    pqp = _parquet_path()
    texts = pq.read_table(pqp, columns=["text"]).column("text").to_pylist()
    log(f"pile-10k: {len(texts)} rows from {pqp}")
    eids, epick, ehash = pile_split_parquet(tok, FISHER_N_DOC, 0, texts)
    fids, fpick, fhash = pile_split_parquet(tok, FISHER_N_DOC, FISHER_N_DOC, texts)
    if ehash & fhash or set(epick) & set(fpick):
        raise RuntimeError("the fit split overlaps the eval split")
    log(f"fit split: {FISHER_N_DOC} docs, picks {fpick[:4]} ...; eval picks {epick[:4]} ...")
    if a.check_split:
        z = np.load(a.check_split, allow_pickle=False)
        ok_e = set(z["eval_sha256"].tolist()) == ehash
        ok_f = set(z["fit_sha256"].tolist()) == fhash if "fit_sha256" in z.files else None
        log(f"split cross-check against {os.path.basename(a.check_split)}: eval set {'MATCH' if ok_e else 'DIFFERS'}, "
            f"fit set {'MATCH' if ok_f else ('DIFFERS' if ok_f is False else 'not recorded')}")
        if not ok_e or ok_f is False:
            raise RuntimeError("the parquet split does not reproduce the recorded split; the local gate would read other passages")
    ids = fids[:a.ndocs]
    rec = gate(a.condition, model, ids, os.path.join(a.out, a.condition), a.out, a.model, log=log, reuse=False)
    log(json.dumps({k: rec[k] for k in ("condition", "gauge", "verdict", "bitwise", "bar", "x_bar", "fp_original",
                                        "fp_regauged")}, indent=1))
    log(json.dumps(rec["regauged"], indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
