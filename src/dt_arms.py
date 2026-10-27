import os

import kvlab
import gam_core as gc
import gam_solve as gs
import gam_apply as ga

GAUGE_DIR = os.environ.get("KVTYING_GAUGE_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauges"))
GROUPS = [1, 2, 4, 8]
CASE = "B"
T_EVAL = 512

RAND_SEED = 0
RAND_KAPPA = 2.0

OPT_VARIANT = {
    "optimal":    "full",
    "optnoquery": "noquery",
    "optgeom":    "geom",
    "optdelta0":  "delta0",
    "optkonly":   "konly",
    "optvonly":   "vonly",
    "optident":   "identity",
    "optidgauge": "identgauge",
    "optoptgauge":"optgauge",
}
OPT_MATCH_VARIANT = {
    "optmatch":         "full",
    "optmatch_noquery": "noquery",
    "optmatch_geom":    "geom",
    "optmatch_delta0":  "delta0",
    "optmatch_konly":   "konly",
    "optmatch_vonly":   "vonly",
}
OPT_MATCH = "optmatch"

ASYM_G = [1, 2, 4, 8, 16]
OPT_ASYM = {f"optasym_k{g}": g for g in ASYM_G}

POOL_ASYM = {f"poolasym_k{g}": g for g in ASYM_G}
G3B_ASYM = {f"g3basym_k{g}": g for g in ASYM_G}
PAIR_JSON = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "deep_dive", "scripts",
                                         "pair_and_identity_{short}.json"))

def matched_pairs(model_name):
    import json
    p = PAIR_JSON.format(short=model_name.split("/")[-1])
    with open(p) as f:
        rows = json.load(f)["layers"]
    return [[list(pair) for pair in row["match_g2_pairs"]] for row in rows]

def arm_names():
    out = ["mha"]
    for fam in ("pool", "refit", "gauge3A", "gauge3B", "randcomm", "norope",
                "omegaonly", "gammaonly"):
        out += [f"{fam}_G{g}" for g in GROUPS]
    for fam in OPT_VARIANT:
        out += [f"{fam}_G{g}" for g in GROUPS]
    out += [f"{fam}_G8" for fam in OPT_MATCH_VARIANT]
    for fams in (OPT_ASYM, POOL_ASYM, G3B_ASYM):
        out += [f"{fam}_G{g}" for fam in fams for g in ASYM_G
                if not (fams[fam] == 16 and g == 16)]
    return out

def family(name):
    return "mha" if name == "mha" else name.rsplit("_G", 1)[0]

def n_groups(name):
    return None if name == "mha" else int(name.rsplit("_G", 1)[1])

def npz_path(model_name, arm, G):
    short = model_name.split("/")[-1]
    return os.path.join(GAUGE_DIR, f"gauges_{short}_G{G}_{arm}_{CASE}.npz")

def available(model_name):
    ok, missing = [], []
    for a in arm_names():
        fam, G = family(a), n_groups(a)
        if fam in OPT_MATCH_VARIANT:
            p = PAIR_JSON.format(short=model_name.split("/")[-1])
            (ok if os.path.exists(p) else missing).append(a)
        elif fam in G3B_ASYM:
            need = [g for g in (G3B_ASYM[fam], G) if g != 16]
            (ok if all(os.path.exists(npz_path(model_name, "arm3B", g)) for g in need)
             else missing).append(a)
        elif (fam in ("mha", "pool", "refit", "randcomm", "norope")
                or fam in OPT_VARIANT or fam in OPT_ASYM or fam in POOL_ASYM):
            ok.append(a)
        else:
            arm = "arm3A" if fam == "gauge3A" else "arm3B"
            (ok if os.path.exists(npz_path(model_name, arm, G)) else missing).append(a)
    return ok, missing

def build(model, model_name, arm):
    fam, G = family(arm), n_groups(arm)
    if fam == "mha":
        return "untouched checkpoint"

    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)

    if fam == "pool":
        for lay in layers:
            gc.pool_layer(lay, G, H, d_k, dm, which="kv")
        return f"kvlab.pool_layer G={G}"

    if fam == "refit":
        groups = kvlab.groups_of(H, G)
        theta = gc.inv_freq(model)
        for lay in layers:
            Qc, Kc, Vc, Wo, sh, shipped = gc.layer_maps(lay, H, d_k, d_v, dm, CASE)
            a = gs.solve_grouped("arm2b", Qc, Kc, Vc, Wo, groups, d_k, d_v, r,
                                 theta=theta, T=T_EVAL)
            Kb, Vb, Qh, Oh = gc.serve_maps(a, sh, shipped, CASE, groups)
            Qt, Kt, Vt, _ = shipped
            gc.write_layer(lay, Qh, gc.tie(Kt, Kb, groups), gc.tie(Vt, Vb, groups), Oh,
                           H, d_k, d_v, dm)
        return f"arm2b solve_grouped G={G} Sigma={CASE}"

    if fam == "norope":
        import dt_norope
        return dt_norope.build(model, G)

    if fam == "randcomm":
        import dt_norope
        import numpy as np
        groups = kvlab.groups_of(H, G)
        rng = np.random.default_rng(RAND_SEED)
        for lay in layers:
            Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
            Qh, Kh = np.zeros_like(Qt), np.zeros_like(Kt)
            Vh, Oh = np.zeros_like(Vt), np.zeros_like(Wo)
            for i in range(H):
                Om, Oi = dt_norope.random_omega(rng, d_k, r, RAND_KAPPA)
                Gm, Gi = dt_norope.random_gamma(rng, d_v, RAND_KAPPA)
                Kh[i] = Om @ Kt[i]
                Qh[i] = Oi.T @ Qt[i]
                Vh[i] = Gm @ Vt[i]
                Oh[i] = Wo[i] @ Gi
            Kb = np.stack([Kh[list(g)].mean(axis=0) for g in groups])
            Vb = np.stack([Vh[list(g)].mean(axis=0) for g in groups])
            gc.write_layer(lay, Qh, gc.tie(Kh, Kb, groups), gc.tie(Vh, Vb, groups), Oh,
                           H, d_k, d_v, dm)
        return f"random commutant gauge (seed={RAND_SEED}, kappa={RAND_KAPPA}) then pool, G={G}"

    if fam in ("omegaonly", "gammaonly"):
        import numpy as np
        p = npz_path(model_name, "arm3B", G)
        z = np.load(p, allow_pickle=False)
        meta = ga.load_meta(p)
        groups = meta["groups"]
        for l, lay in enumerate(layers):
            Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
            if fam == "omegaonly":
                Oi = z[f"Omega_inv_{l}"]
                Qh = np.stack([Oi[i].T @ Qt[i] for i in range(H)])
                Kb = z[f"Kbar_{l}"].astype(np.float64)
                Oh = Wo
                Vb = np.stack([Vt[list(g)].mean(axis=0) for g in groups])
            else:
                Gi = z[f"Gamma_inv_{l}"]
                Qh = Qt
                Kb = np.stack([Kt[list(g)].mean(axis=0) for g in groups])
                Oh = np.stack([Wo[i] @ Gi[i] for i in range(H)])
                Vb = z[f"Vbar_{l}"].astype(np.float64)
            gc.write_layer(lay, Qh, gc.tie(Kt, Kb, groups), gc.tie(Vt, Vb, groups), Oh,
                           H, d_k, d_v, dm)
        side = "Omega only (QK gauged, OV plain-pooled)" if fam == "omegaonly" else \
               "Gamma only (OV gauged, QK plain-pooled)"
        return f"arm3B ablation -- {side}, G={G}"

    if fam in OPT_ASYM:
        import dt_optimal
        return dt_optimal.build(model, OPT_ASYM[fam], variant="full", G_v=G)

    if fam in POOL_ASYM:
        GK = POOL_ASYM[fam]
        for lay in layers:
            if GK != 16:
                gc.pool_layer(lay, GK, H, d_k, dm, which="k")
            if G != 16:
                gc.pool_layer(lay, G, H, d_k, dm, which="v")
        return f"kvlab.pool_layer G_K={GK} G_V={G}"

    if fam in G3B_ASYM:
        import numpy as np
        GK = G3B_ASYM[fam]
        zK = mK = zV = mV = None
        if GK != 16:
            pK = npz_path(model_name, "arm3B", GK)
            zK, mK = np.load(pK, allow_pickle=False), ga.load_meta(pK)
        if G != 16:
            pV = npz_path(model_name, "arm3B", G)
            zV, mV = np.load(pV, allow_pickle=False), ga.load_meta(pV)
        for l, lay in enumerate(layers):
            Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
            if zK is None:
                Qh, Kh = Qt, Kt
            else:
                Oi = zK[f"Omega_inv_{l}"]
                Qh = np.stack([Oi[i].T @ Qt[i] for i in range(H)])
                Kh = gc.tie(Kt, zK[f"Kbar_{l}"].astype(np.float64), mK["groups"])
            if zV is None:
                Oh, Vh = Wo, Vt
            else:
                Gi = zV[f"Gamma_inv_{l}"]
                Oh = np.stack([Wo[i] @ Gi[i] for i in range(H)])
                Vh = gc.tie(Vt, zV[f"Vbar_{l}"].astype(np.float64), mV["groups"])
            gc.write_layer(lay, Qh, Kh, Vh, Oh, H, d_k, d_v, dm)
        return f"arm3B gauges, G_K={GK} G_V={G}"

    if fam in OPT_VARIANT or fam in OPT_MATCH_VARIANT:
        import dt_optimal
        if fam in OPT_MATCH_VARIANT:
            if G != 8:
                raise ValueError(f"{fam} exists only at G=8 (|g|=2); no partition search has been "
                                 f"run at G={G}, and at G=1 the partition is unique so the "
                                 f"corresponding opt* arm already is the matched construction")
            return dt_optimal.build(model, G, variant=OPT_MATCH_VARIANT[fam],
                                    groups_by_layer=matched_pairs(model_name))
        return dt_optimal.build(model, G, variant=OPT_VARIANT[fam])

    arm_key = "arm3A" if fam == "gauge3A" else "arm3B"
    p = npz_path(model_name, arm_key, G)
    m = ga.apply_saved(model, p)
    return f"apply_saved {os.path.basename(p)} (G={m['G']}, H={m['H']})"

def fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    for lay in kvlab.get_layers(model):
        projs = ((lay.self_attn.q_proj, lay.self_attn.k_proj, lay.self_attn.v_proj, lay.self_attn.o_proj)
                 if hasattr(lay, "self_attn") else (lay.attention.query_key_value, lay.attention.dense))
        for p in projs:
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
