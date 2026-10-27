import json
import os

import dt_arch_llama
import dt_arms
import dt_decoupled_arch
import dt_gqa_arch
import dt_mla_arch
import dt_mla_arch_llama
import dt_optimal_arch
import kvlab

META_NAME = "kv_tying_meta.json"

_SEARCHED_ARMS = ("optimal_search", "procrustes_search", "procrustes_simk")

def _groups_by_layer(meta):
    gbl = meta.get("groups_by_layer")
    if gbl is None and meta.get("arm") in _SEARCHED_ARMS and meta.get("G", 1) > 1:
        raise ValueError(
            f"checkpoint for searched arm {meta.get('arm')!r} has no 'groups_by_layer' in its meta. "
            f"Its partition was found by a per-layer search and CANNOT be reconstructed from (H, G); "
            f"restructuring with the contiguous default would silently tie the wrong heads and read "
            f"the wrong group's K/V at inference, while passing every shape and fingerprint check.\n"
            f"This checkpoint predates the field. Rebuild the branch (delete it and re-run init) "
            f"rather than reloading it.")
    return gbl

WIDE_ARMS = frozenset({"optwide", "optwide_fisher"})

def _optimal_B(meta):
    B = meta.get("B")
    if B is None or meta.get("arm") in WIDE_ARMS:
        return B
    if not isinstance(meta.get("G"), int):
        raise ValueError(f"a grouped {meta.get('arm')!r} sidecar records budget B = {B} with G = {meta.get('G')!r}; "
                         f"a grouping arm needs a whole G")
    return None

_RESTRUCTURE = {
    "mha": None,
    "gqa": lambda model, meta: dt_gqa_arch.restructure(
        model, meta["G"], check_tie=False, groups_by_layer=_groups_by_layer(meta)),
    "optimal": lambda model, meta: dt_optimal_arch.restructure_optimal(
        model, meta["base_model"], meta["G"], meta.get("G_v"), check_tie=False,
        groups_by_layer=_groups_by_layer(meta), wide=(meta.get("arm") in WIDE_ARMS), B=_optimal_B(meta)),
    "decoupled": lambda model, meta: dt_decoupled_arch.restructure_decoupled(
        model, meta["base_model"], meta["G"], check_tie=False,
        w=meta.get("w", dt_decoupled_arch.dt_decoupled.W_KV_DEFAULT),
        d_k_R=meta.get("d_k_R", dt_decoupled_arch.dt_decoupled.DKR_DEFAULT), B=meta.get("B")),
    "mla": dt_mla_arch.skeleton_mla,
}
_FINGERPRINT = {
    "mha": dt_arms.fingerprint,
    "gqa": dt_gqa_arch.gqa_fingerprint,
    "optimal": dt_optimal_arch.optimal_fingerprint,
    "decoupled": dt_decoupled_arch.decoupled_fingerprint,
    "mla": dt_mla_arch.mla_fingerprint,
}

_RESTRUCTURE_LLAMA = {
    "mha": None,
    "gqa": lambda model, meta: dt_arch_llama.restructure_gqa(
        model, meta["G"], check_tie=False, groups_by_layer=_groups_by_layer(meta)),
    "optimal": lambda model, meta: dt_arch_llama.restructure_optimal(
        model, meta["base_model"], meta["G"], meta.get("G_v"), check_tie=False,
        groups_by_layer=_groups_by_layer(meta), wide=(meta.get("arm") in WIDE_ARMS), B=_optimal_B(meta)),
    "decoupled": lambda model, meta: dt_arch_llama.restructure_decoupled(
        model, meta["base_model"], meta["G"], check_tie=False,
        w=meta.get("w", dt_decoupled_arch.dt_decoupled.W_KV_DEFAULT),
        d_k_R=meta.get("d_k_R", dt_decoupled_arch.dt_decoupled.DKR_DEFAULT), B=meta.get("B")),
    "mla": dt_mla_arch_llama.skeleton_mla,
}
_FINGERPRINT_LLAMA = {
    "mha": dt_arms.fingerprint,
    "gqa": dt_arch_llama.gqa_fingerprint,
    "optimal": dt_arch_llama.optimal_fingerprint,
    "decoupled": dt_arch_llama.decoupled_fingerprint,
    "mla": dt_mla_arch_llama.mla_fingerprint,
}

def _tables(model):
    if dt_arch_llama.is_llama_layout(model):
        return _RESTRUCTURE_LLAMA, _FINGERPRINT_LLAMA, dt_mla_arch_llama
    return _RESTRUCTURE, _FINGERPRINT, dt_mla_arch

def fingerprint_of(arch, model):
    return _tables(model)[1][arch](model)

MLA_META_REQUIRED = ("G", "arm", "mla_schema", "rotary_mode", "w", "d_c", "mla_scale", "bias_set",
                     "mla_layers", "alloc_sha256", "attn_params_by_layer", "mla_probe", "mla_width",
                     "mla_provenance", "fisher_sha256", "teacher_attn_sha256")

def save_checkpoint(model, path, meta):
    for k in ("arch", "base_model", "fingerprint"):
        if k not in meta:
            raise ValueError(f"meta missing required key {k!r}")
    if meta["arch"] not in _RESTRUCTURE:
        raise ValueError(f"unknown arch {meta['arch']!r} (known: {sorted(_RESTRUCTURE)})")
    if meta["arch"] == "mla":
        miss = [k for k in MLA_META_REQUIRED if k not in meta]
        if miss:
            raise ValueError(f"an 'mla' sidecar must carry {list(MLA_META_REQUIRED)}; missing {miss}")
        _tables(model)[2].check_alloc(model, meta)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    with open(os.path.join(path, META_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return path

def _load_raw_state_dict(path):
    from safetensors.torch import load_file
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"no model.safetensors or .index.json under {path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    state = {}
    for shard_file in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(path, shard_file)))
    return state

def _reconcile_legacy_keys(state, model):
    ALIASES = {"embed_out.weight": "lm_head.weight"}
    model_keys = set(model.state_dict().keys())
    for on_disk, live in ALIASES.items():
        if on_disk in state and on_disk not in model_keys and live in model_keys:
            state[live] = state.pop(on_disk)
    return state

def _tied_aliases(model):
    if not getattr(model.config, "tie_word_embeddings", False):
        return {}
    keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(keys, dict) or not keys:
        raise RuntimeError(f"{type(model).__name__} ties its embeddings but has no {{alias: source}} "
                           f"_tied_weights_keys (got {keys!r}); this transformers version is not supported")
    return dict(keys)

def _check_tied(model, tied):
    if not tied:
        return
    p = dict(model.named_parameters(remove_duplicate=False))
    for alias, src in tied.items():
        if alias not in p or src not in p or p[alias].data_ptr() != p[src].data_ptr():
            raise RuntimeError(f"the tie {alias} -> {src} did not survive the reload")

def read_meta(path):
    with open(os.path.join(path, META_NAME), "r", encoding="utf-8") as f:
        return json.load(f)

def load_checkpoint(path, device="cpu"):
    meta = read_meta(path)
    arch = meta["arch"]
    if arch not in _RESTRUCTURE:
        raise ValueError(f"unknown arch {arch!r} in checkpoint meta (known: {sorted(_RESTRUCTURE)})")

    model, _ = kvlab.load_model(meta["base_model"])
    restructure, fingerprints, mla = _tables(model)
    step = restructure[arch]
    if step is not None:
        step(model, meta)

    state = _load_raw_state_dict(path)
    state = _reconcile_legacy_keys(state, model)
    tied = _tied_aliases(model)
    for alias, src in tied.items():
        if alias not in state and src in state:
            state[alias] = state[src]
    model.load_state_dict(state, strict=True)
    _check_tied(model, tied)
    if arch == "mla":
        mla.check_alloc(model, meta)

    fp = fingerprints[arch](model)
    if fp != meta["fingerprint"]:
        raise RuntimeError(
            f"reload fingerprint mismatch: saved {meta['fingerprint']}, reloaded {fp}. "
            "state_dict load silently dropped or misconnected a tensor."
        )
    if arch == "mla":
        mla.check_probe(model, meta["mla_probe"])

    if device != "cpu":
        model = model.to(device)
    return model
