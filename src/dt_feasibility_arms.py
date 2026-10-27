import time

import numpy as np

import dt_arms
import dt_calib
import dt_decoupled
import dt_decoupled_arch
import dt_gqa_arch
import dt_groupsearch
import dt_optimal
import dt_optimal_arch
import dt_procrustes
import gam_core as gc
import kvlab
import hashlib
import importlib
import json

import torch

import dt_mla_arch
import dt_arch_llama
import dt_mla_arch_llama

def _arch_of(model):
    if dt_arch_llama.is_llama_layout(model):
        return (dt_arch_llama.restructure_gqa, dt_arch_llama.restructure_optimal,
                dt_arch_llama.restructure_decoupled)
    return dt_gqa_arch.restructure, dt_optimal_arch.restructure_optimal, dt_decoupled_arch.restructure_decoupled

ARM_NAMES =("mha", "pool", "gauge3B", "optimal", "optimal_search",
             "procrustes", "procrustes_search", "procrustes_simk", "optwide",
             "decoupled_optwide",
             "decoupled_optwide_median",
             "decoupled_optwide_flexible",
             "decoupled_optwide_fisher",
             "optwide_fisher",
             "decoupled_optwide_fisher_sigma",
             "decoupled_optwide_fisher_sigma_dstar",
             "mla_ours_fisher_sigma",
             "mla_transmla_sigma",
             "mla_mha2mla_s2norm",
             "mla_care_sigma",
             "mla_ours_fisher_sigma_eig",
             "mla_w64_ours_fisher_sigma",
             "mla_w64_transmla_sigma",
             "mla_w64_mha2mla_s2norm",
             "mla_w64_care_sigma",
             "decoupled_optwide_fisher_sigma_dkr64",
             "mla_w64_ours_unrope_energy", "mla_w64_ours_unrope_cost",
             "mla_ours_unrope_energy", "mla_ours_unrope_cost",
             "mla_w16_transmla_sigma", "mla_w16_mha2mla_s2norm",
             "mla_w16_ours_unrope_energy", "mla_w16_ours_unrope_cost",
             "decoupled_optwide_fisher_sigma_dkr16",
             "mla_w64_ours_unrope_tmcopy", "mla_w64_ours_unrope_tmdamped",
             "mla_w64_ours_unrope_tmcopy_fi",
             "mla_w64_ours_unrope_cost_fi",
             "mla_w16_ours_unrope_cost_fi", "mla_w64_ours_unrope_cost_si", "mla_w16_ours_unrope_cost_si",
             "mla_w64_ours_unrope_cost_ii", "mla_w16_ours_unrope_cost_ii")

MLA_ARMS = {
    "mla_ours_fisher_sigma": ("dt_mla_ours", "shared", frozenset({"fisher", "sigma"}), ("dkv",),
                              "d_h^-1/2"),
    "mla_transmla_sigma": ("dt_mla_transmla", "shared", frozenset({"sigma"}), ("dkv",),
                           "(d_h+w)^-1/2"),
    "mla_mha2mla_s2norm": ("dt_mla_mha2mla", "per_head", frozenset({"s2norm"}), ("uk", "uv"),
                           "d_h^-1/2"),
    "mla_care_sigma": ("dt_mla_care", "shared", frozenset({"sigma"}), ("dkv",),
                       "(d_h+w)^-1/2"),
    "mla_ours_fisher_sigma_eig": ("dt_mla_ours_eig", "shared", frozenset({"fisher", "sigma"}), ("dkv",),
                                  "d_h^-1/2"),
    "mla_w64_ours_fisher_sigma": ("dt_mla_w64_ours", "shared", frozenset({"fisher", "sigma"}), ("dkv",),
                                  "d_h^-1/2"),
    "mla_w64_transmla_sigma": ("dt_mla_w64_transmla", "shared", frozenset({"sigma"}), ("dkv",),
                               "(d_h+w)^-1/2"),
    "mla_w64_mha2mla_s2norm": ("dt_mla_w64_mha2mla", "per_head", frozenset({"s2norm"}), ("uk", "uv"),
                               "d_h^-1/2"),
    "mla_w64_care_sigma": ("dt_mla_w64_care", "shared", frozenset({"sigma"}), ("dkv",),
                           "(d_h+w)^-1/2"),
    "mla_w64_ours_unrope_energy": ("dt_mla_w64_ours_unrope_energy", "shared", frozenset({"fisher", "sigma"}),
                                   ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_cost": ("dt_mla_w64_ours_unrope_cost", "shared", frozenset({"fisher", "sigma"}),
                                 ("dkv",), "d_h^-1/2"),
    "mla_ours_unrope_energy": ("dt_mla_ours_unrope_energy", "shared", frozenset({"fisher", "sigma"}),
                               ("dkv",), "d_h^-1/2"),
    "mla_ours_unrope_cost": ("dt_mla_ours_unrope_cost", "shared", frozenset({"fisher", "sigma"}),
                             ("dkv",), "d_h^-1/2"),
    "mla_w16_transmla_sigma": ("dt_mla_w16_transmla", "shared", frozenset({"sigma"}), ("dkv",),
                               "(d_h+w)^-1/2"),
    "mla_w16_mha2mla_s2norm": ("dt_mla_w16_mha2mla", "per_head", frozenset({"s2norm"}), ("uk", "uv"),
                               "d_h^-1/2"),
    "mla_w16_ours_unrope_energy": ("dt_mla_w16_ours_unrope_energy", "shared", frozenset({"fisher", "sigma"}),
                                   ("dkv",), "d_h^-1/2"),
    "mla_w16_ours_unrope_cost": ("dt_mla_w16_ours_unrope_cost", "shared", frozenset({"fisher", "sigma"}),
                                 ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_tmcopy": ("dt_mla_w64_ours_unrope_tmcopy", "shared", frozenset({"fisher", "sigma"}),
                                   ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_tmdamped": ("dt_mla_w64_ours_unrope_tmdamped", "shared", frozenset({"fisher", "sigma"}),
                                     ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_tmcopy_fi": ("dt_mla_w64_ours_unrope_tmcopy_fi", "shared", frozenset({"sigma"}),
                                      ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_cost_fi": ("dt_mla_w64_ours_unrope_cost_fi", "shared", frozenset({"sigma"}),
                                    ("dkv",), "d_h^-1/2"),
    "mla_w16_ours_unrope_cost_fi": ("dt_mla_w16_ours_unrope_cost_fi", "shared", frozenset({"sigma"}),
                                    ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_cost_si": ("dt_mla_w64_ours_unrope_cost_si", "shared", frozenset({"fisher"}),
                                    ("dkv",), "d_h^-1/2"),
    "mla_w16_ours_unrope_cost_si": ("dt_mla_w16_ours_unrope_cost_si", "shared", frozenset({"fisher"}),
                                    ("dkv",), "d_h^-1/2"),
    "mla_w64_ours_unrope_cost_ii": ("dt_mla_w64_ours_unrope_cost_ii", "shared", frozenset(), ("dkv",), "d_h^-1/2"),
    "mla_w16_ours_unrope_cost_ii": ("dt_mla_w16_ours_unrope_cost_ii", "shared", frozenset(), ("dkv",), "d_h^-1/2"),
}
MLA_WIDTH_RULE = {"mla_ours_fisher_sigma_eig": "eigen",
                  "mla_w64_ours_fisher_sigma": "fixed:64", "mla_w64_transmla_sigma": "fixed:64",
                  "mla_w64_mha2mla_s2norm": "fixed:64", "mla_w64_care_sigma": "fixed:64",
                  "mla_w64_ours_unrope_energy": "fixed:64", "mla_w64_ours_unrope_cost": "fixed:64",
                  "mla_ours_unrope_energy": "unrope_argmin", "mla_ours_unrope_cost": "unrope_argmin",
                  "mla_w16_transmla_sigma": "fixed:16", "mla_w16_mha2mla_s2norm": "fixed:16",
                  "mla_w16_ours_unrope_energy": "fixed:16", "mla_w16_ours_unrope_cost": "fixed:16",
                  "mla_w64_ours_unrope_tmcopy": "fixed:64", "mla_w64_ours_unrope_tmdamped": "fixed:64",
                  "mla_w64_ours_unrope_tmcopy_fi": "fixed:64",
                  "mla_w64_ours_unrope_cost_fi": "fixed:64",
                  "mla_w16_ours_unrope_cost_fi": "fixed:16",
                  "mla_w64_ours_unrope_cost_si": "fixed:64", "mla_w16_ours_unrope_cost_si": "fixed:16",
                  "mla_w64_ours_unrope_cost_ii": "fixed:64", "mla_w16_ours_unrope_cost_ii": "fixed:16"}
_MLA_WIDTH_RULES = ("argmin", "eigen", "fixed:64", "unrope_argmin", "fixed:16")
_MLA_FIXED_W = {"fixed:64": 64, "fixed:16": 16}
MLA_WIDTH_SCHEMA = "mla-width/v3"
_MLA_INPUTS = ("fisher", "sigma", "s2norm")
_MLA_SCALE_RULES = ("d_h^-1/2", "(d_h+w)^-1/2")

def _search_weights_invariant_by_layer(model, G, n_restarts=8, seed=0, kv_weight=0.5,
                                        max_evals_per_restart=4000):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    theta = gc.inv_freq(model)
    wD = dt_optimal._offset_weights(dt_optimal.T_EVAL)
    out = []
    t_start = time.time()
    for li, lay in enumerate(layers):
        t0 = time.time()
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        AK = np.stack([Qt[i] @ Qt[i].T for i in range(H)])
        AK = np.stack([dt_optimal._rope_avg(AK[i], r, theta, wD) for i in range(H)])
        AV = np.stack([Wo[i].T @ Wo[i] for i in range(H)])

        group_score_fn = dt_groupsearch.group_scorer_weights_invariant(Kt, AK, Vt, AV, d_k, d_v,
                                                                        kv_weight)
        out.append(dt_groupsearch.search(H, G, group_score_fn, direction="min",
                                          n_restarts=n_restarts, seed=seed,
                                          max_evals_per_restart=max_evals_per_restart,
                                          label=f"weights-invariant search layer {li + 1}/{L}"))
        print(f"  [weights-invariant search] layer {li + 1}/{L} done ({time.time() - t0:.1f}s, "
              f"{time.time() - t_start:.1f}s total)")
    return out

def _equal_partitions(items, size):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    import itertools
    for others in itertools.combinations(rest, size - 1):
        block = [first, *others]
        remaining = [x for x in rest if x not in others]
        for tail in _equal_partitions(remaining, size):
            yield [block] + tail

def _coarsen_kv_groups(H, Hkv, G, group_score_fn):
    if Hkv % G:
        raise ValueError(f"G = {G} does not coarsen the source's {Hkv} KV groups (D-G2)")
    n_rep = H // Hkv
    units = [list(range(j * n_rep, (j + 1) * n_rep)) for j in range(Hkv)]
    best = None
    for part in _equal_partitions(list(range(Hkv)), Hkv // G):
        groups = [sorted(h for u in blk for h in units[u]) for blk in part]
        s = sum(group_score_fn(g) for g in groups)
        if best is None or s > best[0]:
            best = (s, groups)
    return best[1]

def _search_simk_by_layer(model, calib_batch, G, n_restarts=8, seed=0, max_evals_per_restart=4000):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    x_by_layer = dt_calib.capture_calibration_x(model, calib_batch)
    out = []
    t_start = time.time()
    for li, lay in enumerate(layers):
        t0 = time.time()
        Kact, _ = dt_procrustes.layer_activations(lay, H, d_k, d_v, dm, x_by_layer.pop(li))
        S = dt_procrustes.simk_matrix(Kact, H)

        group_score_fn = dt_groupsearch.group_scorer_simk(S)
        Hkv = getattr(model, "_gqa_kv_heads", None)
        if Hkv is not None:
            out.append(_coarsen_kv_groups(H, Hkv, G, group_score_fn))
        else:
            out.append(dt_groupsearch.search(H, G, group_score_fn, direction="max",
                                              n_restarts=n_restarts, seed=seed,
                                              max_evals_per_restart=max_evals_per_restart,
                                              label=f"SimK search layer {li + 1}/{L}"))
        print(f"  [SimK search] layer {li + 1}/{L} done ({time.time() - t0:.1f}s, "
              f"{time.time() - t_start:.1f}s total)")
    return out

DKR_FIXED = 64
DKR_FIXED_16 = 16
_BUDGET_ARMS = ("mha", "optwide", "optwide_fisher", "decoupled_optwide", "decoupled_optwide_median",
                "decoupled_optwide_flexible", "decoupled_optwide_fisher", "decoupled_optwide_fisher_sigma",
                "decoupled_optwide_fisher_sigma_dstar", "decoupled_optwide_fisher_sigma_dkr64",
                "decoupled_optwide_fisher_sigma_dkr16")
_GUESSED_WIDTH_ARMS = frozenset({"mha", "pool", "gauge3B", "optimal", "optimal_search", "procrustes",
                                 "procrustes_search", "procrustes_simk", "optwide", "decoupled_optwide",
                                 "decoupled_optwide_median", "decoupled_optwide_flexible",
                                 "decoupled_optwide_fisher", "optwide_fisher"})

def _budget_keyed(dstar):
    key = (dstar or {}).get("budget_key", "G")
    if key not in ("G", "B"):
        raise ValueError(f"dstar['budget_key'] = {key!r}, expected 'G' or 'B'")
    return key == "B"

def _width_from(dstar, arm, G, B=None):
    if arm in MLA_ARMS:
        raise ValueError(
            f"{arm} is an MLA arm: its width comes ONLY from _mla_widths(arm, G, dstar, fisher, sigma) "
            f"(dstar['mla'], MLA_ARMS_BUILD_PLAN.md §3), never from _width_from.")
    if dstar is None:
        raise ValueError(
            f"{arm} at G={G} needs a rotary width and no `dstar` table was passed. Run the d* cell "
            f"(it prints the per-layer table and checks it against elegant_formulation.tex) and pass "
            f"its DSTAR through init_all_branches(dstar=...).")
    if arm == "decoupled_optwide_fisher_sigma":
        return dt_decoupled.DKR_DEFAULT
    if arm == "decoupled_optwide_fisher_sigma_dkr64":
        return DKR_FIXED
    if arm == "decoupled_optwide_fisher_sigma_dkr16":
        return DKR_FIXED_16

    if arm == "decoupled_optwide_fisher_sigma_dstar":
        key = "fisher_sigma"
    elif arm in _GUESSED_WIDTH_ARMS:
        key = "guessed"
    else:
        raise ValueError(f"unknown arm {arm!r} for _width_from: it is neither an MLA arm, nor one of the "
                         f"three Sigma arms, nor in _GUESSED_WIDTH_ARMS {sorted(_GUESSED_WIDTH_ARMS)}.")
    table = dstar.get(key)
    if not table:
        raise ValueError(
            f"{arm} needs dstar[{key!r}], which is absent. Run the d* cell -- and if {key!r} is "
            f"'fisher_sigma', note that cell must run AFTER the Sigma cell, since the whitened "
            f"spectra it crosses cannot be computed without Sigma.")
    if _budget_keyed(dstar):
        if B is None:
            raise ValueError(f"{arm}: dstar[{key!r}] is keyed by the budget B and no B was passed (G={G}).")
        row = table.get(B, table.get(str(B)))
        if row is None:
            raise ValueError(f"{arm}: dstar[{key!r}] has no entry for B={B} (has {sorted(table)}).")
        return row["by_layer"] if arm.endswith("_flexible") else row["median"]
    row = table.get(G, table.get(str(G)))
    if row is None:
        raise ValueError(f"{arm}: dstar[{key!r}] has no entry for G={G} (has {sorted(table)}).")
    return row["by_layer"] if arm.endswith("_flexible") else row["median"]

def mla_entry(arm):
    if arm not in MLA_ARMS:
        raise ValueError(f"{arm!r} is not an MLA arm (MLA_ARMS: {sorted(MLA_ARMS)})")
    ent = MLA_ARMS[arm]
    if not (isinstance(ent, tuple) and len(ent) == 5):
        raise TypeError(f"MLA_ARMS[{arm!r}] must be (init_module, rotary_mode, needs, bias_set, "
                        f"scale_rule), got {ent!r}")
    name, mode, needs, bias_set, rule = ent
    needs, bias_set = frozenset(needs), tuple(bias_set)
    if mode not in dt_mla_arch.MLA_MODES:
        raise ValueError(f"MLA_ARMS[{arm!r}]: rotary_mode {mode!r} not in {dt_mla_arch.MLA_MODES}")
    if not needs <= set(_MLA_INPUTS):
        raise ValueError(f"MLA_ARMS[{arm!r}]: needs {sorted(needs)} outside {_MLA_INPUTS}")
    if list(bias_set) != sorted(set(bias_set)) or not set(bias_set) <= set(dt_mla_arch.MLA_BIAS_MEMBERS):
        raise ValueError(f"MLA_ARMS[{arm!r}]: bias_set {list(bias_set)} must be a sorted, duplicate-free "
                         f"subset of {dt_mla_arch.MLA_BIAS_MEMBERS} (R1)")
    if rule not in _MLA_SCALE_RULES:
        raise ValueError(f"MLA_ARMS[{arm!r}]: scale_rule {rule!r} not in {_MLA_SCALE_RULES}")
    mod = importlib.import_module(name)
    for const, want in (("ARM", arm), ("ROTARY_MODE", mode)):
        got = getattr(mod, const, None)
        if got != want:
            raise ValueError(f"the MLA registry disagrees with {name}: {name}.{const} = {got!r} but "
                             f"MLA_ARMS[{arm!r}] says {const.lower()} = {want!r}")
    got = getattr(mod, "BIAS_SET", None)
    if got is None or sorted(got) != list(bias_set):
        raise ValueError(f"the MLA registry disagrees with {name}: {name}.BIAS_SET = {got!r} but "
                         f"MLA_ARMS[{arm!r}] says bias_set = {list(bias_set)} (R1)")
    if not callable(getattr(mod, "build_weights", None)):
        raise TypeError(f"{name} has no callable build_weights")
    got = getattr(mod, "WIDTH_RULE", "argmin")
    if got != MLA_WIDTH_RULE.get(arm, "argmin"):
        raise ValueError(f"the MLA registry disagrees with {name}: {name}.WIDTH_RULE = {got!r} but "
                         f"MLA_WIDTH_RULE.get({arm!r}, 'argmin') = {MLA_WIDTH_RULE.get(arm, 'argmin')!r}")
    if not callable(getattr(mod, "mla_scale_of", None)):
        raise TypeError(f"{name}: scale_rule {rule!r} needs mla_scale_of(w, d_h) to agree with")
    return mod, mode, needs, bias_set, rule

def check_mla_registry(arms=None):
    for a in (list(MLA_ARMS) if arms is None else arms):
        mla_entry(a)
    return True

def mla_scale_for(arm, w, d_h):
    rule = MLA_ARMS[arm][4]
    if rule == "d_h^-1/2":
        return float(d_h ** -0.5)
    if rule == "(d_h+w)^-1/2":
        return float((d_h + w) ** -0.5)
    raise ValueError(f"MLA_ARMS[{arm!r}]: unknown scale_rule {rule!r}")

def _mla_table_sha256(tab, G, row):
    body = {"schema": tab["schema"], "rule": tab["rule"], "rounding": tab["rounding"],
            "base_fingerprint": tab["base_fingerprint"], "fisher_sha256": tab["fisher_sha256"],
            "sigma_sha256": tab["sigma_sha256"], "G": int(G) if G is not None else "none", "B": int(row["B"]),
            "m_by_layer": [int(x) for x in row["m_by_layer"]], "m": int(row["m"]), "w": int(row["w"]),
            "d_c": int(row["d_c"])}
    for k, v in body.items():
        if not isinstance(v, (str, int, list)) or isinstance(v, bool):
            raise TypeError(f"table_sha256 field {k} = {v!r} is not an int or a str")
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()

def _mla_widths(arm, G, dstar, fisher, sigma, B=None):
    import dt_feasibility_train as _T
    if arm not in MLA_ARMS:
        raise ValueError(f"_mla_widths: {arm!r} is not an MLA arm (MLA_ARMS: {sorted(MLA_ARMS)})")
    if B is not None:
        if isinstance(B, bool) or not isinstance(B, (int, np.integer)):
            raise TypeError(f"{arm}: B must be an int, got {B!r}")
        B = int(B)
        if G is not None and (isinstance(G, bool) or not isinstance(G, (int, np.integer))):
            raise TypeError(f"{arm}: G must be an int or None, got {G!r}")
        G = None if G is None else int(G)
    else:
        if isinstance(G, bool) or not isinstance(G, (int, np.integer)):
            raise TypeError(f"{arm}: G must be an int, got {G!r}")
        G = int(G)
    if dstar is None:
        raise ValueError(f"{arm} at G={G} needs the MLA width table and no `dstar` was passed. Run the d* "
                         f"cell (c10, @@MLA@@ block) and pass DSTAR through init_all_branches(dstar=...).")
    tab = dstar.get("mla")
    if not tab:
        raise ValueError(f"{arm} needs dstar['mla'], which is absent: the c10 @@MLA@@ block has not run "
                         f"(or ran on a notebook without it).")
    if tab.get("schema") != MLA_WIDTH_SCHEMA:
        raise ValueError(f"dstar['mla'] schema {tab.get('schema')!r} != {MLA_WIDTH_SCHEMA!r}: a table "
                         f"from a different block version. Re-run c10.")
    bf = dstar.get("base_fingerprint")
    if bf is None or tab.get("base_fingerprint") != bf:
        raise ValueError(f"dstar['mla'] base_fingerprint {tab.get('base_fingerprint')!r} != DSTAR's "
                         f"{bf!r}: the MLA table was computed from a different checkpoint.")
    if fisher is None:
        raise ValueError(f"{arm}: the MLA width is routed on the FISHER it was computed from (D-M1); "
                         f"none was passed (metric=FISHER).")
    if sigma is None:
        raise ValueError(f"{arm}: the MLA width is routed on the SIGMA it was computed from (D-M1); "
                         f"none was passed.")
    if len(fisher) != 2:
        raise ValueError(f"{arm}: fisher must be the pair (GK, GV), got {len(fisher)} arrays")
    GK, GV = (np.asarray(a) for a in fisher)
    if GK.ndim != 4 or GV.ndim != 4 or GK.shape[:2] != GV.shape[:2] or GK.shape[2] != GK.shape[3] \
            or GV.shape[2] != GV.shape[3]:
        raise ValueError(f"{arm}: fisher shapes {GK.shape}, {GV.shape}; expected (L, H, d_k, d_k) and "
                         f"(L, H, d_v, d_v)")
    L, unit = int(GK.shape[0]), int(GK.shape[2] + GV.shape[2])
    if len(sigma) != L:
        raise ValueError(f"{arm}: sigma has {len(sigma)} layers, the Fisher {L}")
    fh = _T.fisher_digest(fisher)
    if fh != tab.get("fisher_sha256"):
        raise ValueError(f"the FISHER passed hashes to {fh}, but dstar['mla'] was computed from "
                         f"{tab.get('fisher_sha256')}: the width table does not belong to this Fisher.")
    sh = _T.sigma_digest(sigma)
    if sh != tab.get("sigma_sha256"):
        raise ValueError(f"the SIGMA passed hashes to {sh}, but dstar['mla'] was computed from "
                         f"{tab.get('sigma_sha256')}: the width table does not belong to this Sigma.")
    keyed_B = _budget_keyed(dstar)
    if keyed_B:
        import dt_grid_models as _GM
        mtag = tab.get("model_tag")
        if mtag not in _GM.MODELS:
            raise ValueError(f"dstar['mla'] is budget-keyed but names no registry model (model_tag {mtag!r})")
        has_ref = "regression" in _GM.MODELS[mtag]
        if not (tab.get("crosscheck_ok") is True or (tab.get("crosscheck_ok") is None and not has_ref)):
            raise ValueError(f"dstar['mla'] crosscheck_ok is {tab.get('crosscheck_ok')!r} for {mtag}, which "
                             f"{'HAS' if has_ref else 'has no'} registry regression block: "
                             f"{'it must reproduce it' if has_ref else 'it must be None or True'}. Do not build.")
        if B is None:
            raise ValueError(f"{arm}: dstar['mla'] is keyed by the budget B and no B was passed (G={G}).")
        by = tab.get("by_B") or {}
        row = by.get(B, by.get(str(B)))
        if row is None:
            raise ValueError(f"dstar['mla'] has no row for B={B} (has B in {sorted(str(k) for k in by)}).")
    else:
        if tab.get("crosscheck_ok") is not True:
            raise ValueError(f"dstar['mla'] crosscheck_ok is {tab.get('crosscheck_ok')!r}, not True: the "
                             f"Fisher+Sigma d* here does not reproduce decoupled_optwide_fisher_sigma_dstar "
                             f"as built (8/16/36/134), so these are not that arm's arrays. Do not build.")
        if B is not None and G is None:
            raise ValueError(f"{arm}: dstar['mla'] is keyed by G, and the budget B = {B} has no whole G. A star "
                             f"budget needs its model's budget-keyed table (dstar['budget_key'] = 'B').")
        if B is not None and B != unit * G:
            raise ValueError(f"{arm}: budget B = {B} != (d_k + d_v) * G = {unit * G}")
        by = tab.get("by_G") or {}
        row = by.get(G, by.get(str(G)))
        if row is None:
            raise ValueError(f"dstar['mla'] has no row for G={G} (has G in {sorted(str(k) for k in by)}).")
    budget = B if keyed_B else unit * G
    rule = MLA_WIDTH_RULE.get(arm, "argmin")
    if rule not in _MLA_WIDTH_RULES:
        raise ValueError(f"MLA_WIDTH_RULE[{arm!r}] = {rule!r} not in {_MLA_WIDTH_RULES}")
    if rule == "eigen":
        if keyed_B:
            raise ValueError(f"{arm}: the eigen-width rule is Pythia-410m's retired arm; it has no budget-keyed route")
        return _mla_widths_eigen(arm, G, tab, row, L, unit, fh, sh)
    if rule in _MLA_FIXED_W:
        return _mla_widths_fixed(arm, G, tab, row, L, unit, fh, sh, rule, budget=budget)
    unrope_alloc = None
    if rule == "unrope_argmin":
        if not keyed_B:
            raise ValueError(f"{arm}: the un-rope KL width is a full-RoPE model's, and its table is budget-keyed")
        unrope_alloc = getattr(importlib.import_module(MLA_ARMS[arm][0]), "ROTARY_ALLOC", None)
        _rows = (tab.get("by_alloc") or {}).get(unrope_alloc)
        if not _rows:
            raise ValueError(f"{arm}: dstar['mla'] has no un-rope rows for rotary allocation {unrope_alloc!r} "
                             f"(by_alloc has {sorted(tab.get('by_alloc') or {})}): run this model's width cell")
        row = _rows.get(B, _rows.get(str(B)))
        if row is None:
            raise ValueError(f"{arm}: dstar['mla']['by_alloc'][{unrope_alloc!r}] has no row for B={B}")
    if row.get("blocked") is not False:
        raise ValueError(
            f"the MLA width at G={G} is BLOCKED (D4; blocked = {row.get('blocked')!r}): the half-up median "
            f"m* is 0 or fills the budget (m_by_layer {row.get('m_by_layer')}). No arm is built at G={G}. "
            f"The fix to put to the authors is a floor of m = 1 (w = 2); it is never applied automatically.")

    def _int(k):
        v = row.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"dstar['mla'] G={G} row: {k} = {v!r} is not an int")
        return int(v)

    B, w, d_c, m = _int("B"), _int("w"), _int("d_c"), _int("m")
    mbl = row.get("m_by_layer")
    if not isinstance(mbl, list) or len(mbl) != L or \
            any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in mbl):
        raise TypeError(f"dstar['mla'] G={G} row: m_by_layer must be {L} ints, got {mbl!r}")
    if B != budget:
        raise ValueError(f"dstar['mla'] G={G} row: B = {B} != the budget {budget}"
                         + ("" if keyed_B else f" = (d_k + d_v) * G = {unit} * {G} (128*G on this model)"))
    if w % 2:
        raise ValueError(f"dstar['mla'] G={G} row: w = {w} is odd; the MLA rotary width must be even "
                         f"(one complex direction = 2 cached dims)")
    if not 0 < w < B:
        raise ValueError(f"dstar['mla'] G={G} row: w = {w} outside (0, B = {B})")
    if w + d_c != B:
        raise ValueError(f"dstar['mla'] G={G} row: w + d_c = {w} + {d_c} = {w + d_c} != B = {B}")
    if w != 2 * m:
        raise ValueError(f"dstar['mla'] G={G} row: m = {m} but w = {w}; w must be 2*m")
    if L % 2:
        raise ValueError(f"the half-up median rule assumes an even layer count; L = {L}")
    srt = sorted(int(x) for x in mbl)
    m_rule = (srt[L // 2 - 1] + srt[L // 2] + 1) // 2
    if m_rule != m:
        raise ValueError(f"dstar['mla'] G={G} row: m = {m} is not the half-up median of its own "
                         f"m_by_layer ({m_rule}, D2) -- the row was edited by hand.")
    rec = dict(row)
    rec.update(schema=tab["schema"], rule=tab["rule"], rounding=tab["rounding"],
               base_fingerprint=tab["base_fingerprint"], fisher_sha256=fh, sigma_sha256=sh, G=G,
               table_sha256=_mla_table_sha256(tab, G, row))
    if unrope_alloc is not None:
        rec.update(width_rule="unrope_argmin", rotary_alloc=unrope_alloc,
                   table_sha256=hashlib.sha256(f"{rec['table_sha256']}|unrope_argmin|{unrope_alloc}"
                                               .encode("utf-8")).hexdigest())
    rec = json.loads(json.dumps(rec))
    return w, d_c, rec

def _mla_widths_eigen(arm, G, tab, row, L, unit, fh, sh):
    def _int(k):
        v = row.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"dstar['mla'] G={G} row: {k} = {v!r} is not an int")
        return int(v)

    B, proxy_w = _int("B"), _int("proxy_w")
    mbl = row.get("proxy_m_by_layer")
    if not isinstance(mbl, list) or len(mbl) != L or \
            any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in mbl):
        raise TypeError(f"dstar['mla'] G={G} row: proxy_m_by_layer must be {L} ints, got {mbl!r}")
    mbl = [int(x) for x in mbl]
    if B != unit * G:
        raise ValueError(f"dstar['mla'] G={G} row: B = {B} != (d_k + d_v) * G = {unit * G}")
    if L % 2:
        raise ValueError(f"the half-up median rule assumes an even layer count; L = {L}")
    srt = sorted(mbl)
    lo, hi = srt[L // 2 - 1], srt[L // 2]
    m = (lo + hi + 1) // 2
    if 2 * m != proxy_w:
        raise ValueError(f"dstar['mla'] G={G} row: proxy_w = {proxy_w} is not 2 x the half-up median of its "
                         f"own proxy_m_by_layer ({2 * m}) -- the row was edited by hand or the block changed.")
    if m == 0 or 2 * m >= B:
        raise ValueError(f"the EIGEN-energy MLA width at G={G} is BLOCKED (D4): the half-up median m* = {m} "
                         f"is 0 or fills the budget B = {B} (proxy_m_by_layer {mbl}). Not built.")
    w, d_c = 2 * m, B - 2 * m
    rec = dict(row)
    rec["argmin"] = {k: row.get(k) for k in ("w", "d_c", "m", "m_by_layer", "m_median_raw", "half_tie",
                                             "m_half_even", "blocked")}
    rec.update(width_rule="eigen", w=w, d_c=d_c, m=m, m_by_layer=mbl, m_median_raw=(lo + hi) / 2,
               half_tie=(lo + hi) % 2 == 1, m_half_even=int(round((lo + hi) / 2)), blocked=False,
               diagnostics_note="margin_in / margin_out / runner_up_* / at_cap / at_budget describe the "
                                "ARGMIN rule's per-layer search, not these eigen widths")
    body = {"schema": tab["schema"], "rule": tab["rule"], "rounding": tab["rounding"],
            "base_fingerprint": tab["base_fingerprint"], "fisher_sha256": tab["fisher_sha256"],
            "sigma_sha256": tab["sigma_sha256"], "G": int(G), "B": int(B), "width_rule": "eigen",
            "m_by_layer": mbl, "m": int(m), "w": int(w), "d_c": int(d_c)}
    rec.update(schema=tab["schema"], rule=tab["rule"], rounding=tab["rounding"],
               base_fingerprint=tab["base_fingerprint"], fisher_sha256=fh, sigma_sha256=sh, G=G,
               table_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest())
    rec = json.loads(json.dumps(rec))
    return w, d_c, rec

def _mla_widths_fixed(arm, G, tab, row, L, unit, fh, sh, rule, budget=None):
    v = row.get("B")
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"dstar['mla'] G={G} row: B = {v!r} is not an int")
    B, w = int(v), int(_MLA_FIXED_W[rule])
    want = budget if budget is not None else unit * G
    if B != want:
        raise ValueError(f"dstar['mla'] G={G} row: B = {B} != the budget {want}")
    if w % 2:
        raise ValueError(f"{arm}: fixed width w = {w} is odd; the MLA rotary width must be even "
                         f"(one complex direction = 2 cached dims)")
    if not 0 < w < B:
        raise ValueError(f"{arm} is NOT REPRESENTABLE at G={G}: the fixed width w = {w} leaves no latent "
                         f"in B = {B} (d_c = {B - w}).")
    d_c = B - w
    rec = dict(row)
    rec["argmin"] = {k: row.get(k) for k in ("w", "d_c", "m", "m_by_layer", "m_median_raw", "half_tie",
                                             "m_half_even", "blocked")}
    rec.update(width_rule=rule, w=w, d_c=d_c, m=w // 2, m_by_layer=[w // 2] * L, m_median_raw=float(w // 2),
               half_tie=False, m_half_even=w // 2, blocked=False,
               diagnostics_note="margin_in / margin_out / runner_up_* / at_cap / at_budget describe the "
                                "ARGMIN rule's per-layer search, not this fixed width")
    body = {"width_rule": rule, "G": int(G) if G is not None else "none", "B": int(B), "w": int(w), "d_c": int(d_c),
            "base_fingerprint": tab["base_fingerprint"], "fisher_sha256": tab["fisher_sha256"],
            "sigma_sha256": tab["sigma_sha256"]}
    rec.update(schema=tab["schema"], rule=tab["rule"], rounding=tab["rounding"],
               base_fingerprint=tab["base_fingerprint"], fisher_sha256=fh, sigma_sha256=sh, G=G,
               table_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest())
    rec = json.loads(json.dumps(rec))
    return w, d_c, rec

@torch.no_grad()
def _mla_logprobs(model, ids):
    was = model.training
    model.eval()
    try:
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64))
        return torch.log_softmax(model(x).logits.float(), dim=-1)
    finally:
        model.train(was)

def _mla_probe_kl(t_lp, s_lp):
    kl = []
    for b in range(t_lp.shape[0]):
        lt, ls = t_lp[b].double(), s_lp[b].double()
        kl.append((lt.exp() * (lt - ls)).sum(-1))
    kl = torch.cat(kl)
    return float(kl.mean()), float(kl.max()), int(kl.numel())

def _build_mla(model, arm, G, *, dstar, fisher, sigma, s2norm, route_sigma, calib_batch, B=None):
    import dt_feasibility_train as _T
    mod, mode, needs, bias_set, rule = mla_entry(arm)
    if calib_batch is not None:
        raise ValueError(f"{arm}: calib_batch is a Procrustes input; no MLA initialiser takes one")
    for nm, val in (("sigma", sigma), ("s2norm", s2norm)):
        if nm in needs and val is None:
            raise ValueError(f"{arm} needs {nm}; none was passed. Its initialiser {mod.__name__} takes "
                             f"{sorted(needs)}, and without it this is not that method.")
        if nm not in needs and val is not None:
            raise ValueError(
                f"{'an' if nm == 's2norm' else 'a'} {nm} was passed for {arm}, whose initialiser takes "
                f"only {sorted(needs)}. Building with "
                f"it would move a second variable into the comparison, so it is refused."
                + (" (Its WIDTH still routes on SIGMA: that one goes in as route_sigma.)"
                   if nm == "sigma" else ""))
    if fisher is None:
        raise ValueError(f"{arm}: every MLA width comes from the Fisher+Sigma table (D-M1), so the FISHER "
                         f"(metric=) is required to route, even where the initialiser does not use it.")
    if route_sigma is None:
        raise ValueError(f"{arm}: route_sigma (the SIGMA dstar['mla'] was computed from) is required to "
                         f"route every MLA arm (D-M1), whether or not its initialiser takes a Sigma.")
    if sigma is not None and sigma is not route_sigma and \
            _T.sigma_digest(sigma) != _T.sigma_digest(route_sigma):
        raise ValueError(f"{arm}: the initialiser's sigma ({_T.sigma_digest(sigma)}) and route_sigma "
                         f"({_T.sigma_digest(route_sigma)}) differ -- the width would be chosen under one "
                         f"second moment and the subspace under another.")
    w, d_c, rec = _mla_widths(arm, G, dstar, fisher, route_sigma, B=B)
    llama = dt_arch_llama.is_llama_layout(model)
    A = dt_mla_arch_llama if llama else dt_mla_arch
    L, H, d_k, d_v, dm, r = A.geometry(model)
    if B is None and rec["B"] != G * (d_k + d_v):
        raise ValueError(f"{arm}: the table's B = {rec['B']} != G*(d_k+d_v) = {G * (d_k + d_v)} of the "
                         f"live model")
    if B is not None and (rec["B"] != B or (G is not None and B != G * (d_k + d_v))):
        raise ValueError(f"{arm}: the table's B = {rec['B']}, the budget B = {B}, G = {G}: they disagree on the "
                         f"live model (d_k + d_v = {d_k + d_v})")
    scale = mla_scale_for(arm, w, d_k)
    mscale = mod.mla_scale_of(w, d_k)
    if mscale != scale:
        raise ValueError(f"{arm}: {mod.__name__}'s scale {mscale!r} != the registry's {rule} = {scale!r} (D1)")
    s2h = _T.s2norm_digest(s2norm) if "s2norm" in needs else None

    teacher = A.teacher_attn_digest(model)
    probe_ids = A.mla_probe(model)["ids"]
    t_lp = _mla_logprobs(model, probe_ids)
    kw = {k: v for k, v in (("fisher", fisher), ("sigma", sigma), ("s2norm", s2norm)) if k in needs}
    print(f"[{arm}] G={G}: w={w}, d_c={d_c} from dstar['mla'] (table {rec['table_sha256'][:16]}); "
          f"{mode}, bias {list(bias_set)}, scale {scale!r}; initialiser inputs {sorted(kw)}")
    if llama:
        with dt_mla_arch_llama.llama_compat(model):
            weights = mod.build_weights(model, w=w, d_c=d_c, G=G, **kw)
    else:
        weights = mod.build_weights(model, w=w, d_c=d_c, G=G, **kw)

    for k, v in (("arm", arm), ("rotary_mode", mode), ("w", w), ("d_c", d_c), ("G", G)):
        if weights.get(k) != v:
            raise ValueError(f"{arm}: the initialiser returned {k} = {weights.get(k)!r}; the registry / "
                             f"width table say {v!r}")
    if list(weights.get("bias_set") or ()) != list(bias_set):
        raise ValueError(f"{arm}: the initialiser's bias_set {weights.get('bias_set')!r} != the "
                         f"registry's {list(bias_set)} (R1)")
    sc = weights.get("mla_scale")
    if not isinstance(sc, float) or sc != scale:
        raise ValueError(f"{arm}: the initialiser's mla_scale {sc!r} != {rule} = {scale!r} (D1)")
    prov = weights.get("provenance") or {}
    for k, v in (("fisher_sha256", rec["fisher_sha256"] if "fisher" in needs else None),
                 ("sigma_sha256", rec["sigma_sha256"] if "sigma" in needs else None),
                 ("s2norm_sha256", s2h)):
        if prov.get(k) != v:
            raise ValueError(f"{arm}: the initialiser's provenance {k} = {prov.get(k)!r}, but the input it "
                             f"was given hashes to {v!r}")
    expect = {"arm": arm, "G": G, "w": w, "d_c": d_c, "rotary_mode": mode, "bias_set": list(bias_set),
              "mla_scale": scale, "teacher_attn_sha256": teacher}
    if set(expect) != set(dt_mla_arch.EXPECT_KEYS):
        raise RuntimeError(f"expectation keys {sorted(expect)} != dt_mla_arch.EXPECT_KEYS "
                           f"{sorted(dt_mla_arch.EXPECT_KEYS)}")
    version, text = weights.get("version"), prov.get("text", "")
    A.restructure_mla(model, expect, weights)
    del weights
    s_lp = _mla_logprobs(model, probe_ids)
    kl_mean, kl_max, n_pos = _mla_probe_kl(t_lp, s_lp)
    del t_lp, s_lp
    inputs = {"fisher_sha256": rec["fisher_sha256"], "teacher_attn_sha256": teacher}
    if "sigma" in needs:
        inputs["sigma_sha256"] = rec["sigma_sha256"]
    if "s2norm" in needs:
        inputs["s2norm_sha256"] = s2h
    model.mla_width_record = rec
    model.mla_inputs = inputs
    model.mla_t0_probe = {"t0_probe_kl": kl_mean, "t0_probe_kl_max": kl_max}
    print(f"[{arm}] t=0 probe KL(teacher || student): mean {kl_mean:.4g}, max {kl_max:.4g} over {n_pos} "
          f"positions")
    return (model,
            f"{arm} ({version}): {mode}, w={w}, d_c={d_c} at G={G} from dstar['mla'] table "
            f"{rec['table_sha256'][:16]}"
            + (f" (width rule {rec['width_rule']})" if rec.get("width_rule") in _MLA_FIXED_W else "")
            + f"; bias {list(bias_set)}, scale {scale!r}. {text}",
            None)

def build_and_restructure(model, model_name, arm, G, calib_batch=None, n_search_restarts=8, seed=0,
                           search_max_evals=4000, dstar=None, metric=None, sigma=None, s2norm=None,
                           route_sigma=None, B=None):
    if arm in MLA_ARMS:
        return _build_mla(model, arm, G, dstar=dstar, fisher=metric, sigma=sigma, s2norm=s2norm,
                          route_sigma=route_sigma, calib_batch=calib_batch, B=B)
    if B is not None and arm not in _BUDGET_ARMS:
        L_, H_, d_k_, d_v_, _dm, _r = gc.geometry(model)
        if G is None or B != G * (d_k_ + d_v_):
            raise ValueError(f"{arm} groups KV heads, so it needs a whole G; the budget B = {B} has G = {G} "
                             f"(d_k + d_v = {d_k_ + d_v_}). Not representable at this budget.")
        B = None
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")
    if s2norm is not None:
        raise ValueError(f"an s2norm was passed for {arm}, which is not an MLA arm; only "
                         f"mla_mha2mla_s2norm takes one.")
    if route_sigma is not None:
        raise ValueError(f"a route_sigma was passed for {arm}, which is not an MLA arm; it routes the MLA "
                         f"width table only.")
    _gqa, _opt, _dec = _arch_of(model)

    if arm == "mha":
        return model, "untouched checkpoint", None

    if arm == "pool":
        prov = dt_arms.build(model, model_name, f"pool_G{G}")
        _gqa(model, G)
        return model, prov, None

    if arm == "gauge3B":
        prov = dt_arms.build(model, model_name, f"gauge3B_G{G}")
        _gqa(model, G)
        return model, prov, None

    if arm == "optimal":
        prov = dt_optimal.build(model, G, variant="full")
        _opt(model, model_name, G)
        return model, prov, None

    if arm == "optwide":
        prov = dt_optimal.build(model, G, variant="full", wide=True, B=B)
        _opt(model, model_name, G, wide=True, B=B)
        return model, prov, None

    if arm == "decoupled_optwide":
        prov = dt_decoupled.build(model, G, w=dt_decoupled.W_KV_DEFAULT, B=B)
        _dec(model, model_name, G, w=dt_decoupled.W_KV_DEFAULT, B=B)
        return model, prov, None

    if arm in ("decoupled_optwide_median", "decoupled_optwide_flexible",
               "decoupled_optwide_fisher", "decoupled_optwide_fisher_sigma",
               "decoupled_optwide_fisher_sigma_dstar", "decoupled_optwide_fisher_sigma_dkr64",
               "decoupled_optwide_fisher_sigma_dkr16"):
        d = _width_from(dstar, arm, G, B=B)
        GK, GV = (metric or (None, None))
        _wants_fisher = arm in ("decoupled_optwide_fisher", "decoupled_optwide_fisher_sigma",
                                "decoupled_optwide_fisher_sigma_dstar",
                                "decoupled_optwide_fisher_sigma_dkr64",
                                "decoupled_optwide_fisher_sigma_dkr16")
        if _wants_fisher and GK is None:
            raise ValueError(f"{arm} needs a measured metric; none was passed. "
                             "Without it this arm IS decoupled_optwide_median under another name.")
        if not _wants_fisher:
            GK = GV = None
        _S = None
        if arm in ("decoupled_optwide_fisher_sigma", "decoupled_optwide_fisher_sigma_dstar",
                   "decoupled_optwide_fisher_sigma_dkr64", "decoupled_optwide_fisher_sigma_dkr16"):
            if sigma is None:
                raise ValueError(
                    f"{arm} needs a measured Sigma; none was passed. Without it this arm IS "
                    f"decoupled_optwide_fisher under another name -- it would train, log a "
                    f"different arm string, and duplicate a branch in the grid. Run the Sigma cell "
                    f"and pass its SIGMA through init_all_branches(sigma=...).")
            _S = sigma
        elif sigma is not None:
            raise ValueError(
                f"a Sigma was passed for {arm}, which is defined at Sigma = I. Only the "
                f"decoupled_optwide_fisher_sigma* arms take one; building this arm with it would move "
                f"the metric AND the input factor against decoupled_optwide_fisher at once.")
        prov, _bases = dt_decoupled.build(model, G, w=dt_decoupled.W_KV_DEFAULT, d_k_R=d,
                                          GK=GK, GV=GV, SIGMA=_S, return_bases=True, B=B)
        _dec(model, model_name, G, w=dt_decoupled.W_KV_DEFAULT, d_k_R=d,
             bases=_bases if _S is not None else None, B=B)
        return model, prov, None

    if arm == "optwide_fisher":
        GK, GV = (metric or (None, None))
        if GK is None:
            raise ValueError("optwide_fisher needs a measured metric; none was passed. Without it "
                             "this arm IS optwide, and the 2x2 it exists to complete collapses.")
        prov = dt_optimal.build(model, G, variant="full", wide=True, GK=GK, GV=GV, B=B)
        _opt(model, model_name, G, wide=True, B=B)
        return model, prov, None

    if arm == "optimal_search":
        gbl = _search_weights_invariant_by_layer(model, G, n_restarts=n_search_restarts, seed=seed,
                                                  max_evals_per_restart=search_max_evals)
        prov = dt_optimal.build(model, G, variant="full", groups_by_layer=gbl)
        _opt(model, model_name, G, groups_by_layer=gbl)
        return model, prov + " (gauge-aware per-layer search)", gbl

    if calib_batch is None:
        raise ValueError(f"arm {arm!r} requires calib_batch (dt_calib.load_c4_calibration)")

    if arm == "procrustes":
        prov = dt_procrustes.build(model, G, calib_batch=calib_batch)
        _gqa(model, G)
        return model, prov, None

    if arm == "procrustes_search":
        gbl = _search_weights_invariant_by_layer(model, G, n_restarts=n_search_restarts, seed=seed,
                                                  max_evals_per_restart=search_max_evals)
        prov = dt_procrustes.build(model, G, groups_by_layer=gbl, calib_batch=calib_batch)
        _gqa(model, G, groups_by_layer=gbl)
        return model, prov + " (gauge-aware per-layer search)", gbl

    gbl = _search_simk_by_layer(model, calib_batch, G, n_restarts=n_search_restarts, seed=seed,
                                 max_evals_per_restart=search_max_evals)
    prov = dt_procrustes.build(model, G, groups_by_layer=gbl, calib_batch=calib_batch)
    _gqa(model, G, groups_by_layer=gbl)
    return model, prov + " (their SimK per-layer search, no gauge fix)", gbl

def fingerprint(model, arm):
    if arm not in MLA_ARMS and arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}: no fingerprint route (ARM_NAMES: {ARM_NAMES})")
    L = model is not None and dt_arch_llama.is_llama_layout(model)
    if arm in MLA_ARMS:
        return (dt_mla_arch_llama if L else dt_mla_arch).mla_fingerprint(model)
    if arm.startswith("decoupled_optwide"):
        return (dt_arch_llama.decoupled_fingerprint if L else dt_decoupled_arch.decoupled_fingerprint)(model)
    if arm in ("optimal", "optimal_search", "optwide", "optwide_fisher"):
        return (dt_arch_llama.optimal_fingerprint if L else dt_optimal_arch.optimal_fingerprint)(model)
    if arm == "mha":
        return dt_arms.fingerprint(model)
    if arm in ("pool", "gauge3B", "procrustes", "procrustes_search", "procrustes_simk"):
        return (dt_arch_llama.gqa_fingerprint if L else dt_gqa_arch.gqa_fingerprint)(model)
    raise ValueError(f"arm {arm!r} is registered but has no fingerprint route -- add it here")
