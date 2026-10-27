import json
import os
import shutil
import time

import numpy as np
import torch

import dt_checkpoint
import dt_data
import dt_arch_llama
import dt_decoupled_arch
import dt_eval_tasks
import dt_feasibility_arms
import dt_gqa_arch
import dt_logs
import dt_mla_arch
import dt_optimal_arch
import dt_score
import dt_smoltalk
import kvlab

MODEL = "EleutherAI/pythia-410m"
MODEL_TAG = "p410m"

def use_model(tag):
    import dt_grid_models
    global MODEL, MODEL_TAG
    if tag not in dt_grid_models.MODELS:
        raise KeyError(f"unknown model tag {tag!r}; the registry has {sorted(dt_grid_models.MODELS)}")
    m = dt_grid_models.MODELS[tag]
    if m["family"] not in ("neox", "llama_layout"):
        raise NotImplementedError(f"{tag} is {m['family']}: the pipeline converts GPT-NeoX and LLaMA-layout "
                                  f"models only")
    MODEL, MODEL_TAG = m["hf_id"], tag
    return MODEL

def _refuse_foreign_base(path, branch_name):
    bm = dt_checkpoint.read_meta(path).get("base_model")
    if bm != MODEL:
        raise RuntimeError(f"{branch_name} at {path} is a valid checkpoint of {bm!r}, but this session converts "
                           f"{MODEL!r} (MODEL_TAG {MODEL_TAG!r}). Never adopted: point the init root at this "
                           f"model's own directory, or delete the foreign checkpoint.")

GROUP_SIZES = [1, 4, 8]
ARMS = [a for a in dt_feasibility_arms.ARM_NAMES
        if a != "mha" and a not in dt_feasibility_arms.MLA_ARMS]
PROCRUSTES_ARMS = ("procrustes", "procrustes_search", "procrustes_simk")
OPTIMAL_ARCH_ARMS = ("optimal", "optimal_search", "optwide", "optwide_fisher")
DECOUPLED_ARCH_ARMS = ("decoupled_optwide", "decoupled_optwide_median",
                       "decoupled_optwide_flexible", "decoupled_optwide_fisher",
                       "decoupled_optwide_fisher_sigma",
                       "decoupled_optwide_fisher_sigma_dstar",
                       "decoupled_optwide_fisher_sigma_dkr64",
                       "decoupled_optwide_fisher_sigma_dkr16")
WIDTH_ARMS = ("decoupled_optwide_median", "decoupled_optwide_flexible",
              "decoupled_optwide_fisher", "decoupled_optwide_fisher_sigma_dstar",
              "decoupled_optwide_fisher_sigma_dkr64", "decoupled_optwide_fisher_sigma_dkr16")
SIGMA_ARMS = ("decoupled_optwide_fisher_sigma", "decoupled_optwide_fisher_sigma_dstar",
              "decoupled_optwide_fisher_sigma_dkr64", "decoupled_optwide_fisher_sigma_dkr16")
MLA_ARCH_ARMS = dt_feasibility_arms.MLA_ARMS.keys()
SIGMA_ARMS = SIGMA_ARMS + tuple(a for a, e in dt_feasibility_arms.MLA_ARMS.items() if "sigma" in e[2])
S2NORM_ARMS = tuple(a for a, e in dt_feasibility_arms.MLA_ARMS.items() if "s2norm" in e[2])

def sigma_digest(sigma):
    import hashlib
    h = hashlib.sha256()
    for S in sigma:
        h.update(np.ascontiguousarray(S, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]

def fisher_digest(fisher):
    import hashlib
    if len(fisher) != 2:
        raise ValueError(f"fisher must be the pair (GK, GV), got {len(fisher)} arrays")
    h = hashlib.sha256()
    for a in fisher:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]

def s2norm_digest(s2norm):
    import dt_mla_mha2mla
    return dt_mla_mha2mla.s2norm_digest(s2norm)

G1_DEDUP_BASE = {"optimal_search": "optimal", "procrustes_search": "procrustes",
                 "procrustes_simk": "procrustes", "optwide": "optimal"}
N_SEARCH_RESTARTS = 3

LOG_STEP_EVERY = 100
EVAL_FRACTION = 0.10
EVAL_N_PER_TASK = dt_eval_tasks.DEFAULT_N_PERIODIC
EVAL_EARLY_FIRST = 80
EVAL_EARLY_POINTS = 14
LR = 2e-5
BATCH_SIZE = 16
EPOCHS = 2
MAX_LEN = 512
GRAD_CLIP = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))

def paths(out_dir=None):
    base = out_dir or HERE
    return {"logs": os.path.join(base, "logs"),
            "ckpt": os.path.join(base, "checkpoints"),
            "init": os.path.join(base, "checkpoints", "init")}

def _ce_loss(student_logits, labels):
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    ce = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
        ignore_index=-100, reduction="mean")
    return ce, ce.detach(), torch.tensor(0.0)

def _run_pass(model, tok, examples, params, G, train_logger, eval_hook=None, batch_size=BATCH_SIZE,
              lr=LR, max_len=MAX_LEN, log_every=LOG_STEP_EVERY, epochs=EPOCHS):
    device = next(model.parameters()).device
    model.train()
    opt = torch.optim.AdamW(params, lr=lr)
    step, cum_examples, t0 = 0, 0, time.time()

    for _epoch in range(epochs):
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            encoded = [dt_smoltalk.render_and_label(tok, ex["messages"]) for ex in chunk]
            encoded = [(ids[:max_len], lab[:max_len]) for ids, lab in encoded]
            batch = dt_smoltalk.collate_batch(tok, encoded)
            batch = {k: v.to(device) for k, v in batch.items()}

            opt.zero_grad()
            logits = model(batch["input_ids"]).logits
            total, ce, kl = _ce_loss(logits, batch["labels"])
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP)
            opt.step()

            step += 1
            cum_examples += len(chunk)
            is_last = (_epoch == epochs - 1) and (s + batch_size) >= len(examples)
            if train_logger is not None and (step % log_every == 0 or is_last):
                train_logger.log_step(step=step, cum_examples=cum_examples, G=G,
                                       wall_clock_s=round(time.time() - t0, 1), lr=lr,
                                       loss_total=float(total.item()), loss_ce=float(ce.item()),
                                       loss_kl=float(kl.item()), grad_norm=float(grad_norm))
            if eval_hook is not None:
                eval_hook(step, cum_examples)

    return step, cum_examples

def _score_and_log(model, tok, eval_items, eval_logger, branch, G, timing, step, cum_examples):
    for task, items in eval_items.items():
        with torch.no_grad():
            lls, grd = dt_score.score(model, tok, items)
        m = dt_score.evaluate(task, items, lls, grd)
        s = dt_score.summarise(task, m)
        eval_logger.log_eval(branch=branch, G=G, timing=timing, step=step, cum_examples=cum_examples,
                              task=task, acc=s[dt_data.primary_metric(task)], ppl=s.get("ppl"),
                              n=s["n"])

def eval_checkpoints(n_examples, fraction=EVAL_FRACTION, early_first=EVAL_EARLY_FIRST,
                     early_points=EVAL_EARLY_POINTS, batch_size=BATCH_SIZE):
    linear = [max(1, round(n_examples * k * fraction)) for k in range(1, 11)]
    first_linear = linear[0]
    early = []
    if early_points >= 2 and 0 < early_first < first_linear:
        ratio = (first_linear / early_first) ** (1.0 / (early_points - 1))
        for i in range(early_points):
            c = early_first * ratio ** i
            early.append(max(batch_size, int(round(c / batch_size)) * batch_size))
    return sorted(set(early) | set(linear))

def _eval_hook(model, tok, eval_items, eval_logger, branch_name, G, timing, n_examples,
               fraction=EVAL_FRACTION):
    checkpoints = eval_checkpoints(n_examples, fraction)
    done = set()

    def hook(step, cum_examples):
        due = [c for c in checkpoints if cum_examples >= c and c not in done]
        if not due:
            return
        done.update(due)
        model.eval()
        _score_and_log(model, tok, eval_items, eval_logger, branch_name, G, timing, step,
                        cum_examples)
        model.train()
    return hook

def load_eval_items(tasks, n=None, seed=0, phase="periodic"):
    out = {}
    for t in tasks:
        k = dt_eval_tasks.n_for(t, phase) if n is None else n
        items, _ = dt_data.load(t, k, seed)
        out[t] = items
    return out

def _t0_already_logged(log_dir, tasks):
    path = os.path.join(log_dir, "eval_mha_control.jsonl")
    seen = {r["task"] for r in dt_logs.read_jsonl(path) if r.get("cum_examples") == 0}
    return set(tasks) <= seen

def score_t0(eval_items, log_dir, device):
    tasks = list(eval_items.keys())
    if _t0_already_logged(log_dir, tasks):
        print("T0 score already logged, skipping")
        return

    os.makedirs(log_dir, exist_ok=True)
    model, tok = kvlab.load_model(MODEL)
    model = model.to(device)
    model.eval()

    eval_logger = dt_logs.EvalLogger(os.path.join(log_dir, "eval_mha_control.jsonl"))
    print(f"T0 (untouched base) score:")
    print(f"  {'task':<12}{'acc':>10}{'ppl':>10}{'published':>12}{'chance':>9}")
    for task, items in eval_items.items():
        with torch.no_grad():
            lls, grd = dt_score.score(model, tok, items)
        m = dt_score.evaluate(task, items, lls, grd)
        s = dt_score.summarise(task, m)
        metric = dt_data.primary_metric(task)
        acc = s[metric]
        ppl = s.get("ppl")
        eval_logger.log_eval(branch="mha_control", G=None, timing=1.0, step=0, cum_examples=0,
                              task=task, acc=acc, ppl=ppl, n=s["n"])
        published = dt_data.PUBLISHED.get("pythia-410m", {}).get(task, {}).get(metric)
        chance = dt_data.CHANCE.get(task)
        pub_str = f"{published:.4f}" if published is not None else "n/a"
        chance_str = f"{chance:.4f}" if chance is not None else "n/a"
        ppl_str = f"{ppl:.2f}" if ppl is not None else ""
        print(f"  {task:<12}{acc:>10.4f}{ppl_str:>10}{pub_str:>12}{chance_str:>9}")

def run_mha_control(examples, log_dir, ckpt_dir, device, eval_items=None, batch_size=BATCH_SIZE,
                     lr=LR, max_len=MAX_LEN, epochs=EPOCHS):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    model, tok = kvlab.load_model(MODEL)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    train_logger = dt_logs.TrainLogger(os.path.join(log_dir, "train_mha_control.jsonl"))
    hook = None
    if eval_items is not None:
        eval_logger = dt_logs.EvalLogger(os.path.join(log_dir, "eval_mha_control.jsonl"))
        hook = _eval_hook(model, tok, eval_items, eval_logger, "mha_control", None, 1.0,
                           epochs * len(examples))

    opt_params = list(model.parameters())
    _run_pass(model, tok, examples, opt_params, G=None, train_logger=train_logger, eval_hook=hook,
              batch_size=batch_size, lr=lr, max_len=max_len, epochs=epochs)

    fp = dt_arms_fingerprint_mha(model)
    dt_checkpoint.save_checkpoint(
        model, os.path.join(ckpt_dir, "mha_control"),
        {"arch": "mha", "base_model": MODEL, "arm": "mha", "fingerprint": fp})
    return model, tok

def _dt_arms():
    import dt_arms
    return dt_arms

def dt_arms_fingerprint_mha(model):
    return _dt_arms().fingerprint(model)

def _refuse_gpu_conversion(branch_name, device, init_ckpt_dir, allow_gpu_conversion):
    if device.type != "cuda" or allow_gpu_conversion:
        return
    if init_ckpt_dir and os.path.isdir(init_ckpt_dir):
        have = sorted(os.listdir(init_ckpt_dir))
        detail = (f"{init_ckpt_dir} exists and holds {len(have)} branch(es) ({have}) but not "
                  f"{branch_name}, so cell 12b's CPU conversion pass was started but never finished "
                  f"this one.")
    else:
        detail = (f"there is no init checkpoint directory at {init_ckpt_dir!r}, so cell 12b's CPU "
                  f"conversion pass has not been run for this grid at all.")
    raise RuntimeError(
        f"[{branch_name}] refusing to convert on a CUDA device: {detail} Conversion is pure CPU work "
        f"(~4.6h for a G=4 search arm) and would spend this paid GPU session doing what a free CPU "
        f"runtime does identically. Fix: run init_all_branches on a CPU runtime (cell 12b), then "
        f"re-run this cell. To convert on a GPU deliberately anyway, pass allow_gpu_conversion=True.")

def run_branch(arm, G, examples, ckpt_dir, log_dir, device, eval_items=None, batch_size=BATCH_SIZE,
                lr=LR, max_len=MAX_LEN, epochs=EPOCHS, n_search_restarts=N_SEARCH_RESTARTS,
                search_max_evals=4000, calib_n_seqs=128, calib_seq_len=2048, init_ckpt_dir=None,
                allow_gpu_conversion=False):
    branch_name = f"{arm}_G{G}_t0"
    init_path = os.path.join(init_ckpt_dir, branch_name) if init_ckpt_dir else None

    if init_path and _checkpoint_valid(init_path):
        print(f"[{branch_name}] loading preconverted init checkpoint (no reconversion needed)")
        model = dt_checkpoint.load_checkpoint(init_path, device="cpu")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        groups_by_layer = dt_checkpoint.read_meta(init_path).get("groups_by_layer")
        prov = "loaded from init checkpoint"
    else:
        if arm in MLA_ARCH_ARMS:
            raise RuntimeError(
                f"[{branch_name}] MLA arms are built only by init_branch (c11): no valid init checkpoint "
                f"at {init_path!r}. Run init_all_branches for it first.")
        _refuse_gpu_conversion(branch_name, device, init_ckpt_dir, allow_gpu_conversion)
        print(f"[{branch_name}] no init checkpoint found, converting inline")
        model, tok = kvlab.load_model(MODEL)

        calib_batch = None
        if arm in PROCRUSTES_ARMS:
            import dt_calib
            calib_batch = dt_calib.load_c4_calibration(tok, n_seqs=calib_n_seqs, seq_len=calib_seq_len)

        print(f"[{branch_name}] starting conversion...")
        t_conv = time.time()
        model, prov, groups_by_layer = dt_feasibility_arms.build_and_restructure(
            model, MODEL, arm, G, calib_batch=calib_batch, n_search_restarts=n_search_restarts,
            search_max_evals=search_max_evals)
        print(f"[{branch_name}] conversion done ({time.time() - t_conv:.1f}s)")

    model = model.to(device)
    if arm == "optimal" or arm == "optimal_search":
        _, params = dt_optimal_arch.trainable_parameters_optimal(model)
    else:
        _, params = dt_gqa_arch.trainable_parameters(model)

    train_logger = dt_logs.TrainLogger(os.path.join(log_dir, f"train_{branch_name}.jsonl"))
    hook = None
    if eval_items is not None:
        eval_logger = dt_logs.EvalLogger(os.path.join(log_dir, f"eval_{branch_name}.jsonl"))
        hook = _eval_hook(model, tok, eval_items, eval_logger, branch_name, G, 0.0,
                           epochs * len(examples))

    _run_pass(model, tok, examples, params, G, train_logger, eval_hook=hook, batch_size=batch_size,
              lr=lr, max_len=max_len, epochs=epochs)

    arch = ("mla" if arm in MLA_ARCH_ARMS else
            "decoupled" if arm in DECOUPLED_ARCH_ARMS else
            "optimal" if arm in OPTIMAL_ARCH_ARMS else "gqa")
    fp = dt_feasibility_arms.fingerprint(model, arm)
    meta = {"arch": arch, "base_model": MODEL, "G": G, "arm": arm, "fingerprint": fp}
    if arch == "decoupled":
        meta.update(dt_decoupled_arch.decoupled_meta(model))
    if arch == "mla":
        imeta = dt_checkpoint.read_meta(init_path)
        meta.update(dt_mla_arch.mla_meta(model))
        for k in _MLA_CARRY:
            if k in imeta:
                meta[k] = imeta[k]
        meta["init_mla_probe"] = imeta["mla_probe"]
        model = model.to("cpu")
        meta["mla_probe"] = dt_mla_arch.mla_probe(model)
    if groups_by_layer is not None:
        meta["groups_by_layer"] = groups_by_layer
    dt_checkpoint.save_checkpoint(model, os.path.join(ckpt_dir, branch_name), meta)
    return branch_name, prov

_MLA_CARRY = ("mla_width", "mla_provenance", "fisher_sha256", "sigma_sha256", "s2norm_sha256",
              "teacher_attn_sha256", "t0_probe_kl", "t0_probe_kl_max")

def _mla_sidecar(model):
    out = dict(dt_checkpoint._tables(model)[2].mla_meta(model))
    out["mla_width"] = model.mla_width_record
    out["mla_provenance"] = model.mla_build_record
    out.update(model.mla_inputs)
    out.update(model.mla_t0_probe)
    out["mla_probe"] = dt_checkpoint._tables(model)[2].mla_probe(model)
    return out

def _mla_expected(arm, G, dstar, metric, sigma, s2norm, B=None):
    w, d_c, rec = dt_feasibility_arms._mla_widths(arm, G, dstar, metric, sigma, B=B)
    _mod, mode, needs, bias_set, _rule = dt_feasibility_arms.mla_entry(arm)
    d_h = int(np.asarray(metric[0]).shape[-1])
    exp = {"arm": arm, "G": G, "w": w, "d_c": d_c, "table_sha256": rec["table_sha256"],
           "rotary_mode": mode, "bias_set": list(bias_set),
           "mla_scale": dt_feasibility_arms.mla_scale_for(arm, w, d_h), "fisher_sha256": rec["fisher_sha256"]}
    if B is not None:
        exp["B"] = B
    if "sigma" in needs:
        exp["sigma_sha256"] = rec["sigma_sha256"]
    if "s2norm" in needs:
        if s2norm is None:
            raise ValueError(f"{arm}: the existing checkpoint cannot be checked against this session's "
                             f"s2norm -- none was passed (S2NORM, the @@S2@@ block in c11).")
        exp["s2norm_sha256"] = s2norm_digest(s2norm)
    return exp

def _mla_resume_guard(init_path, branch_name, arm, G, dstar, metric, sigma, s2norm, B=None):
    meta = dt_checkpoint.read_meta(init_path)
    exp = _mla_expected(arm, G, dstar, metric, sigma, s2norm, B=B)
    have = {k: meta.get(k) for k in exp}
    have["table_sha256"] = (meta.get("mla_width") or {}).get("table_sha256")
    bad = {k: (have[k], v) for k, v in exp.items() if have[k] != v}
    if bad:
        raise RuntimeError(
            f"{branch_name} already exists but was built from something other than what this session "
            f"would build -- " + "; ".join(f"{k}: checkpoint {a!r}, this session {b!r}"
                                           for k, (a, b) in bad.items())
            + f". A valid MLA checkpoint built from a different width table, Sigma or s2norm is "
            f"invisible to every later check, so it is refused here rather than skipped. Delete "
            f"{init_path} to rebuild it, or pass the inputs it was built from.")

def init_branch(arm, G, init_ckpt_dir, n_search_restarts=N_SEARCH_RESTARTS, search_max_evals=4000,
                 calib_n_seqs=128, calib_seq_len=2048, dstar=None, metric=None, sigma=None, s2norm=None,
                 B=None):
    branch_name = f"{arm}_G{G}_t0" if B is None else f"{arm}_B{B}_t0"
    init_path = os.path.join(init_ckpt_dir, branch_name)
    if _checkpoint_valid(init_path):
        _refuse_foreign_base(init_path, branch_name)
        if dstar is not None and arm in WIDTH_ARMS:
            _mp = os.path.join(init_path, dt_checkpoint.META_NAME)
            if os.path.exists(_mp):
                _have = json.load(open(_mp, encoding="utf-8")).get("d_k_R")
                _want = dt_feasibility_arms._width_from(dstar, arm, G, B=B)
                if _have != _want:
                    def _fmt(x):
                        return (f"{min(x)}..{max(x)} ({len(x)} layers)"
                                if isinstance(x, list) else str(x))
                    raise RuntimeError(
                        f"{branch_name} already exists but was built at d_k^R = {_fmt(_have)}, "
                        f"while the d* table passed here wants {_fmt(_want)}. A valid checkpoint at "
                        f"the wrong width is invisible to every later check -- it reloads, its "
                        f"fingerprint matches, and the cache identity holds -- so it is refused "
                        f"here rather than skipped. Delete {init_path} to rebuild it, or pass the "
                        f"d* table it was actually built from.")
        if sigma is not None and arm in SIGMA_ARMS:
            _mp = os.path.join(init_path, dt_checkpoint.META_NAME)
            if os.path.exists(_mp):
                _have = json.load(open(_mp, encoding="utf-8")).get("sigma_sha256")
                _want = sigma_digest(sigma)
                if _have != _want:
                    raise RuntimeError(
                        f"{branch_name} already exists but was built against Sigma "
                        f"{_have or '(none recorded)'}, while the table passed here is {_want}. "
                        f"A checkpoint built from a different second moment is invisible to every "
                        f"later check -- same arm, same width, same cache size. Delete "
                        f"{init_path} to rebuild it, or pass the Sigma it was built from.")
        if arm in MLA_ARCH_ARMS:
            _mla_resume_guard(init_path, branch_name, arm, G, dstar, metric, sigma, s2norm, B=B)
        if B is not None and dt_checkpoint.read_meta(init_path).get("B") != B:
            raise RuntimeError(f"{branch_name} already exists but its sidecar records budget "
                               f"{dt_checkpoint.read_meta(init_path).get('B')!r}, not {B}. Not adopted.")
        print(f"init: {branch_name} already converted, skipping")
        return branch_name

    model, tok = kvlab.load_model(MODEL)
    calib_batch = None
    if arm in PROCRUSTES_ARMS:
        import dt_calib
        calib_batch = dt_calib.load_c4_calibration(tok, n_seqs=calib_n_seqs, seq_len=calib_seq_len)

    if dstar is not None and dstar.get("base_fingerprint") is not None:
        here = dt_checkpoint._FINGERPRINT["mha"](model)
        if here != dstar["base_fingerprint"]:
            raise RuntimeError(
                f"the d* table was computed from a different checkpoint: it carries "
                f"{dstar['base_fingerprint']}, this base model is {here}. Re-run the d* cell.")

    print(f"[{branch_name}] starting conversion...")
    t_conv = time.time()
    model, _prov, groups_by_layer = dt_feasibility_arms.build_and_restructure(
        model, MODEL, arm, G, calib_batch=calib_batch, n_search_restarts=n_search_restarts,
        search_max_evals=search_max_evals, dstar=dstar, metric=metric,
        sigma=sigma if arm in SIGMA_ARMS else None,
        s2norm=s2norm if arm in S2NORM_ARMS else None,
        route_sigma=sigma if arm in MLA_ARCH_ARMS else None, B=B)
    print(f"[{branch_name}] conversion done ({time.time() - t_conv:.1f}s)")

    arch = ("mla" if arm in MLA_ARCH_ARMS else
            "decoupled" if arm in DECOUPLED_ARCH_ARMS else
            "optimal" if arm in OPTIMAL_ARCH_ARMS else "gqa")
    fp = dt_feasibility_arms.fingerprint(model, arm)
    meta = {"arch": arch, "base_model": MODEL, "G": G, "arm": arm, "fingerprint": fp}
    if B is not None:
        meta["B"] = B
    if arch == "decoupled":
        meta.update((dt_arch_llama.decoupled_meta if dt_arch_llama.is_llama_layout(model)
                     else dt_decoupled_arch.decoupled_meta)(model))
    if arch == "mla":
        meta.update(_mla_sidecar(model))
    if arm in SIGMA_ARMS and sigma is not None:
        meta["sigma_sha256"] = sigma_digest(sigma)
    if groups_by_layer is not None:
        meta["groups_by_layer"] = groups_by_layer
    dt_checkpoint.save_checkpoint(model, init_path, meta)
    print(f"[{branch_name}] saved init checkpoint")
    return branch_name

def init_all_branches(group_sizes, arms, init_ckpt_dir, n_search_restarts=N_SEARCH_RESTARTS,
                       search_max_evals=4000, calib_n_seqs=128, calib_seq_len=2048,
                       dstar=None, metric=None, sigma=None, s2norm=None):
    _mla = [a for a in arms if a in MLA_ARCH_ARMS]
    if _mla:
        dt_feasibility_arms.check_mla_registry(_mla)
    for G in group_sizes:
        for arm in arms:
            if G == 1 and arm in G1_DEDUP_BASE:
                base_arm = G1_DEDUP_BASE[arm]
                init_branch(base_arm, 1, init_ckpt_dir, n_search_restarts=n_search_restarts,
                            search_max_evals=search_max_evals, calib_n_seqs=calib_n_seqs,
                            calib_seq_len=calib_seq_len, dstar=dstar, metric=metric, sigma=sigma,
                            s2norm=s2norm)
                branch_name = f"{arm}_G1_t0"
                dst = os.path.join(init_ckpt_dir, branch_name)
                if _checkpoint_valid(dst):
                    _refuse_foreign_base(dst, branch_name)
                    print(f"init: {branch_name} already converted, skipping")
                    continue
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                src = os.path.join(init_ckpt_dir, f"{base_arm}_G1_t0")
                shutil.copytree(src, dst)
                meta = dt_checkpoint.read_meta(dst)
                meta["arm"] = arm
                with open(os.path.join(dst, dt_checkpoint.META_NAME), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=1)
                print(f"init: {branch_name} == {base_arm}_G1_t0 (G=1 has no combinatorial choice) -- "
                      f"copied, not reconverted")
            else:
                init_branch(arm, G, init_ckpt_dir, n_search_restarts=n_search_restarts,
                            search_max_evals=search_max_evals, calib_n_seqs=calib_n_seqs,
                            calib_seq_len=calib_seq_len, dstar=dstar, metric=metric, sigma=sigma,
                            s2norm=s2norm)

def init_cells(cells, init_ckpt_dir, n_search_restarts=N_SEARCH_RESTARTS, search_max_evals=4000,
               calib_n_seqs=128, calib_seq_len=2048, dstar=None, metric=None, sigma=None, s2norm=None):
    cells = [(a, None if g is None else int(g), int(b)) for a, g, b in cells]
    _mla = sorted({a for a, _, _ in cells if a in MLA_ARCH_ARMS})
    if _mla:
        dt_feasibility_arms.check_mla_registry(_mla)
    kw = dict(n_search_restarts=n_search_restarts, search_max_evals=search_max_evals, calib_n_seqs=calib_n_seqs,
              calib_seq_len=calib_seq_len, dstar=dstar, metric=metric, sigma=sigma, s2norm=s2norm)
    for arm, G, B in cells:
        if G == 1 and arm in G1_DEDUP_BASE:
            base_arm = G1_DEDUP_BASE[arm]
            init_branch(base_arm, 1, init_ckpt_dir, B=B, **kw)
            branch_name = f"{arm}_B{B}_t0"
            dst = os.path.join(init_ckpt_dir, branch_name)
            if _checkpoint_valid(dst):
                _refuse_foreign_base(dst, branch_name)
                if dt_checkpoint.read_meta(dst).get("B") != B:
                    raise RuntimeError(f"{branch_name} exists but its sidecar records budget "
                                       f"{dt_checkpoint.read_meta(dst).get('B')!r}, not {B}. Not adopted.")
                print(f"init: {branch_name} already converted, skipping")
                continue
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(init_ckpt_dir, f"{base_arm}_B{B}_t0"), dst)
            meta = dt_checkpoint.read_meta(dst)
            meta["arm"] = arm
            with open(os.path.join(dst, dt_checkpoint.META_NAME), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            print(f"init: {branch_name} == {base_arm}_B{B}_t0 (one KV head has no combinatorial choice) -- "
                  f"copied, not reconverted")
        else:
            init_branch(arm, G, init_ckpt_dir, B=B, **kw)

def score_branches_t0(group_sizes, arms, eval_items, log_dir, device, n_search_restarts=N_SEARCH_RESTARTS,
                       search_max_evals=4000, calib_n_seqs=128, calib_seq_len=2048, init_ckpt_dir=None,
                       allow_gpu_conversion=False):
    for G in group_sizes:
        for arm in arms:
            branch_name = f"{arm}_G{G}_t0"
            eval_init_path = os.path.join(log_dir, f"eval_{branch_name}_init.jsonl")
            if any(r.get("cum_examples") == 0 for r in dt_logs.read_jsonl(eval_init_path)):
                print(f"t0 already logged for {branch_name}, skipping")
                continue

            init_path = os.path.join(init_ckpt_dir, branch_name) if init_ckpt_dir else None
            if init_path and _checkpoint_valid(init_path):
                print(f"[{branch_name}] loading preconverted init checkpoint (no reconversion needed)")
                model = dt_checkpoint.load_checkpoint(init_path, device="cpu")
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(MODEL)
            else:
                if arm in MLA_ARCH_ARMS:
                    raise RuntimeError(
                        f"[{branch_name}] MLA arms are built only by init_branch (c11) and scored from "
                        f"that init checkpoint; none valid at {init_path!r}. Run init_all_branches first.")
                _refuse_gpu_conversion(branch_name, device, init_ckpt_dir, allow_gpu_conversion)
                print(f"[{branch_name}] no init checkpoint found, converting inline")
                model, tok = kvlab.load_model(MODEL)
                calib_batch = None
                if arm in PROCRUSTES_ARMS:
                    import dt_calib
                    calib_batch = dt_calib.load_c4_calibration(tok, n_seqs=calib_n_seqs,
                                                                 seq_len=calib_seq_len)
                print(f"[{branch_name}] starting conversion...")
                t_conv = time.time()
                model, _prov, _groups = dt_feasibility_arms.build_and_restructure(
                    model, MODEL, arm, G, calib_batch=calib_batch, n_search_restarts=n_search_restarts,
                    search_max_evals=search_max_evals)
                print(f"[{branch_name}] conversion done ({time.time() - t_conv:.1f}s)")

            model = model.to(device)
            model.eval()

            os.makedirs(log_dir, exist_ok=True)
            eval_logger = dt_logs.EvalLogger(eval_init_path)
            _score_and_log(model, tok, eval_items, eval_logger, branch_name, G, 0.0, step=0,
                            cum_examples=0)
            print(f"t0 logged for {branch_name}")

def _checkpoint_valid(path):
    if not os.path.isdir(path):
        return False
    try:
        dt_checkpoint.load_checkpoint(path, device="cpu")
        return True
    except Exception as e:
        if _sidecar_arch(path) == "mla":
            raise RuntimeError(f"MLA checkpoint at {path} failed its reload checks: "
                               f"{type(e).__name__}: {e}") from e
        return False

def _sidecar_arch(path):
    try:
        return dt_checkpoint.read_meta(path).get("arch")
    except Exception:
        return None

def _clear_stale_logs(log_dir, branch_name):
    for prefix in ("train_", "eval_"):
        p = os.path.join(log_dir, f"{prefix}{branch_name}.jsonl")
        if os.path.exists(p):
            os.remove(p)

def main(examples=None, group_sizes=None, arms=None, out_dir=None, eval_tasks=None, eval_items=None,
         batch_size=BATCH_SIZE, lr=LR, max_len=MAX_LEN, epochs=EPOCHS, n_search_restarts=N_SEARCH_RESTARTS,
         search_max_evals=4000, calib_n_seqs=128, calib_seq_len=2048, device=None, init_ckpt_dir=None,
         allow_gpu_conversion=False):
    if device is None:
        raise ValueError("main() requires an explicit device= (e.g. torch.device('cuda') or "
                          "torch.device('cpu')) -- no auto-detect fallback")
    examples = examples if examples is not None else dt_smoltalk.load_cached()
    group_sizes = group_sizes if group_sizes is not None else GROUP_SIZES
    arms = arms if arms is not None else ARMS
    run_paths = paths(out_dir)
    if init_ckpt_dir is None:
        init_ckpt_dir = run_paths["init"]

    eval_items_end = eval_items
    if eval_items is None and eval_tasks is not None:
        eval_items = load_eval_items(eval_tasks, phase="periodic")
        eval_items_end = load_eval_items(eval_tasks, phase="endpoint")

    if _checkpoint_valid(os.path.join(run_paths["ckpt"], "mha_control")):
        print("resume: mha_control already complete, skipping")
    else:
        _clear_stale_logs(run_paths["logs"], "mha_control")
        run_mha_control(examples, run_paths["logs"], run_paths["ckpt"], device, eval_items=eval_items,
                         batch_size=batch_size, lr=lr, max_len=max_len, epochs=epochs)

    if eval_items_end is not None:
        score_t0(eval_items_end, run_paths["logs"], device)

    results = []
    for G in group_sizes:
        for arm in arms:
            branch_name = f"{arm}_G{G}_t0"
            if _checkpoint_valid(os.path.join(run_paths["ckpt"], branch_name)):
                print(f"resume: {branch_name} already complete, skipping")
                results.append((branch_name, "skipped -- already completed (resume)"))
                continue
            _clear_stale_logs(run_paths["logs"], branch_name)
            name, prov = run_branch(arm, G, examples, run_paths["ckpt"], run_paths["logs"], device,
                                     eval_items=eval_items, batch_size=batch_size, lr=lr,
                                     max_len=max_len, epochs=epochs,
                                     n_search_restarts=n_search_restarts,
                                     search_max_evals=search_max_evals,
                                     calib_n_seqs=calib_n_seqs, calib_seq_len=calib_seq_len,
                                     init_ckpt_dir=init_ckpt_dir,
                                     allow_gpu_conversion=allow_gpu_conversion)
            results.append((name, prov))
    return results

if __name__ == "__main__":
    main(eval_tasks=dt_data.FEASIBILITY_TASKS)
