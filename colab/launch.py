import glob
import json
import os
import re
import subprocess
import sys

REPO_DIR = "/content/repo"
DRIVE = "/content/drive/MyDrive/gaugemla"
PINS = ('transformers==5.15.0', 'datasets==5.0.1', 'zstandard==0.25.0')
FOLDER = {
    "olmo1b_gc": "grad_clip",
    "p410m_gc": "grad_clip",
    "slm1p6b_gc": "grad_clip",
    "q0p5b_gc": "grad_clip",
    "l3_1b_gc": "grad_clip",
    "tl1b_gc": "grad_clip",
    "sm360m_gc": "grad_clip",
    "p410m_gc5m": "grad_clip_5m",
    "slm1p6b_urx1_seeds": "true_fisher_x1_seeds",
    "p410m_urx1_seeds": "true_fisher_x1_seeds",
    "tl1b_tx1_seeds": "true_fisher_x1_seeds",
    "sm360m_tx1_seeds": "true_fisher_x1_seeds",
    "olmo1b_tx1_seeds": "true_fisher_x1_seeds",
    "q0p5b_tx1_seeds": "true_fisher_x1_seeds",
    "l3_1b_tx1_seeds": "true_fisher_x1_seeds",
    "olmo1b_klgc": "kl_standin_50m",
    "p410m_klgc": "kl_standin_50m",
    "slm1p6b_klgc": "kl_standin_50m",
    "q0p5b_klgc": "kl_standin_50m",
    "l3_1b_klgc": "kl_standin_50m",
    "tl1b_klgc": "kl_standin_50m",
    "sm360m_klgc": "kl_standin_50m",
    "p410m_kl5m": "kl_standin_5m",
    "slm1p6b_urx1_kls": "kl_standin_ablation",
    "p410m_urx1_kls": "kl_standin_ablation",
    "tl1b_tx1_kls": "kl_standin_ablation",
    "sm360m_tx1_kls": "kl_standin_ablation",
    "olmo1b_tx1_kls": "kl_standin_ablation",
    "q0p5b_tx1_kls": "kl_standin_ablation",
    "l3_1b_tx1_kls": "kl_standin_ablation",
    "slm1p6b_urx1_klsi": "kl_standin_ablation",
    "p410m_urx1_klsi": "kl_standin_ablation",
    "tl1b_tx1_klsi": "kl_standin_ablation",
    "sm360m_tx1_klsi": "kl_standin_ablation",
    "olmo1b_tx1_klsi": "kl_standin_ablation",
    "q0p5b_tx1_klsi": "kl_standin_ablation",
    "l3_1b_tx1_klsi": "kl_standin_ablation",
    "p410m_rgkv": "regauge_test",
    "p410m_rgkv_gqa": "regauge_test"
}
NAMES = {
    "mla_w64_ours_unrope_cost_kl": "GaugeMLA",
    "mla_w16_ours_unrope_cost_kl": "GaugeMLA",
    "mla_w64_ours_unrope_cost": "A_p^mean (un-roping ablation)",
    "mla_w16_ours_unrope_cost": "A_p^mean (un-roping ablation)",
    "mla_w64_ours_unrope_cost_klsi": "GaugeMLA, Σ = I",
    "mla_w16_ours_unrope_cost_klsi": "GaugeMLA, Σ = I",
    "mla_w64_transmla_sigma": "TransMLA",
    "mla_w16_transmla_native_sigma": "TransMLA",
    "mla_w64_mha2mla_s2norm": "MHA2MLA",
    "mla_w16_mha2mla_native_s2norm": "MHA2MLA",
    "mla_w64_care_sigma": "CARE",
    "mla_w16_care_native_sigma": "CARE",
    "decoupled_optwide_fisher_sigma_dkr64": "Decoupled, KL objective",
    "pool": "GQA (pool)",
    "gauge3B": "GQA + optimal gauge",
    "procrustes_simk": "Procrustes alignment",
    "procrustes": "Procrustes alignment, one KV head",
    "optimal": "Per-group subspace",
    "optwide": "Joint key, joint value subspace",
    "mha": "MHA control"
}
ORDER = ["pool", "gauge3B", "procrustes_simk", "procrustes", "optimal", "optwide", "decoupled_optwide_fisher_sigma_dkr64", "mla_w64_ours_unrope_cost_kl", "mla_w64_ours_unrope_cost", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm", "mla_w64_care_sigma", "mla_w16_transmla_native_sigma", "mla_w16_mha2mla_native_s2norm", "mla_w16_care_native_sigma", "mha"]
SOURCE_CACHE = {"p410m_gc": 2048, "p410m_gc5m": 2048, "p410m_klgc": 2048, "p410m_kl5m": 2048}
RESULTS = {
    "p410m_gc": "p410m_gc/runs",
    "p410m_klgc": "kl_standin_50m/p410m"
}
# the regauge test: each kit's conditions, and the paper's two regauge tables as printed (regauge_table below)
REGAUGE = {
 "kits": {
  "p410m_rgkv": [
   "identity",
   "kvs",
   "kvd",
   "kvs_xf",
   "kvd_xf"
  ],
  "p410m_rgkv_gqa": [
   "identity",
   "kvs",
   "kvd"
  ]
 },
 "tables": [
  "4",
  "5"
 ],
 "full_cols": [
  [
   "KL to the teacher",
   "kl_mean"
  ],
  [
   "KL to the original",
   "kl_orig_mean"
  ]
 ],
 "full": [
  [
   "Identity",
   "identity",
   [
    "0.3374",
    "0"
   ]
  ],
  [
   "Per-head scalar",
   "kvs",
   [
    "0.3374",
    "5.3e-9"
   ]
  ],
  [
   "Diagonal",
   "kvd",
   [
    "0.3374",
    "4.1e-9"
   ]
  ]
 ],
 "budgets": [
  [
   "6.25%",
   1
  ],
  [
   "12.5%",
   2
  ],
  [
   "25%",
   4
  ]
 ],
 "cols": [
  [
   "Identity",
   "identity"
  ],
  [
   "Scalar",
   "kvs"
  ],
  [
   "Diag.",
   "kvd"
  ]
 ],
 "conv": [
  [
   "GaugeMLA",
   "p410m_rgkv",
   "mla_w64_ours_unrope_cost_kl",
   "",
   [
    "3.077",
    "+0.016",
    "+0.002",
    "0.905",
    "-0.001",
    "+0.013",
    "0.158",
    "+0.001",
    "+0.003"
   ]
  ],
  [
   "GaugeMLA, F_i carried over",
   "p410m_rgkv",
   "mla_w64_ours_unrope_cost_kl",
   "_xf",
   [
    "3.077",
    "0.000",
    "0.000",
    "0.905",
    "0.000",
    "0.000",
    "0.158",
    "0.000",
    "0.000"
   ]
  ],
  [
   "TransMLA",
   "p410m_rgkv",
   "mla_w64_transmla_sigma",
   "",
   [
    "4.252",
    "+0.397",
    "+0.352",
    "2.685",
    "+0.710",
    "+0.888",
    "0.822",
    "+0.116",
    "+0.284"
   ]
  ],
  [
   "MHA2MLA",
   "p410m_rgkv",
   "mla_w64_mha2mla_s2norm",
   "",
   [
    "4.466",
    "-0.111",
    "-0.256",
    "3.628",
    "+0.108",
    "+0.138",
    "2.556",
    "+0.286",
    "+0.343"
   ]
  ],
  [
   "CARE",
   "p410m_rgkv",
   "mla_w64_care_sigma",
   "",
   [
    "4.260",
    "+0.399",
    "+0.343",
    "2.612",
    "+0.712",
    "+0.961",
    "0.776",
    "+0.135",
    "+0.319"
   ]
  ],
  [
   "GQA",
   "p410m_rgkv_gqa",
   "pool",
   "",
   [
    "4.587",
    "+1.734",
    "+1.734",
    "4.452",
    "+1.137",
    "+1.137",
    "4.642",
    "+0.466",
    "+0.466"
   ]
  ]
 ]
}


def run(tag, stage, **env):
    """Run experiments/<folder>/<tag>/<tag>_<stage>.py in its own process, streaming its output here."""
    folder = f"experiments/{FOLDER[tag]}/{tag}"
    script = f"{REPO_DIR}/{folder}/{tag}_{stage}.py"
    if not os.path.isdir(f"{REPO_DIR}/{folder}"):   # a sparse clone (the notebooks'); a full clone has it already
        subprocess.run(["git", "-C", REPO_DIR, "sparse-checkout", "add", folder], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
    child = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", **{k: str(v) for k, v in env.items()})
    from google.colab import userdata
    try:
        child["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except userdata.SecretNotFoundError:
        pass
    print(f"=== {tag} {stage}: {script}  {env}", flush=True)
    p = subprocess.Popen([sys.executable, "-u", script], env=child, cwd="/content", text=True, bufsize=1,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
        rc = p.wait()
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait()
    if rc:
        raise RuntimeError(f"{tag} {stage} exited with status {rc}: the error is in the output above")
    print(f"=== {tag} {stage}: done", flush=True)


def _rows(files):
    out = {}
    for f in files:
        rows = [json.loads(line) for line in open(f) if line.strip()]
        rows = [r for r in rows if r.get("run_id") == rows[-1].get("run_id")] if rows else []
        if rows:
            out[rows[-1]["branch"]] = rows
    return out


KL_KEYS = (("kl_mean", "KL to the teacher"), ("kl_orig_mean", "KL to the original model"))


def _table(title, rows, cache, key="kl_mean"):
    cols = sorted({r[0]["G"] for r in rows.values()}, reverse=True)
    def label(rs):
        m = re.search(r"_B(\d+)_t0$", rs[0].get("ckpt") or "")
        if m and cache:
            return f"{100 * int(m.group(1)) / cache:g}%"
        return "100%" if rs[0]["G"] == 0 else f"G{rs[0]['G']}"
    arms = sorted(dict.fromkeys(r[0]["arm"] for r in rows.values()),
                  key=lambda a: (ORDER.index(a) if a in ORDER else len(ORDER), a))
    head = {g: next(label(rs) for rs in rows.values() if rs[0]["G"] == g) for g in cols}
    print(f"\n{title}")
    print(f"{'':<34}" + "".join(f"{head[g]:>22}" for g in cols))
    print(f"{'':<34}" + "".join(f"{'t = 0':>11}{'end':>11}" for _ in cols))
    ends = set()
    for a in arms:
        line = f"{NAMES.get(a, a):<34}"
        for g in cols:
            rs = next((v for v in rows.values() if v[0]["arm"] == a and v[0]["G"] == g), None)
            if rs is None:
                line += f"{'':>22}"
                continue
            t0 = next((r[key] for r in rs if r["tokens"] == 0), None)
            end = max(rs, key=lambda r: r["tokens"])
            ends.add(end["tokens"])
            line += (f"{t0:>11.4f}" if t0 is not None else f"{'-':>11}") + f"{end[key]:>11.4f}"
        print(line)
    print(f"end of training: {', '.join(f'{t:,}' for t in sorted(ends))} tokens")


def kl_table(tag, test_tokens=0):
    """KL to the teacher and to the original model, at t = 0 and at the end of training: this run (Drive) beside the
    paper's logged run (results/), for the methods this run trained."""
    runs = f"{DRIVE}/{FOLDER[tag]}/{tag}/runs"
    stamps = [s for s in glob.glob(f"{runs}/grid-{tag}-*")
              if (s.endswith(f"-test{test_tokens}") if test_tokens else "-test" not in os.path.basename(s))]
    cache = SOURCE_CACHE.get(tag)
    mine = {}
    if len(stamps) != 1:
        print(f"this run: {len(stamps)} matching stamp folders under {runs}: {stamps}")
    else:
        mine = _rows(glob.glob(f"{stamps[0]}/kd_*.jsonl"))
        for key, what in KL_KEYS:
            _table(f"THIS RUN ({os.path.basename(stamps[0])}): {what}, mean over the eval documents", mine, cache, key)
    if tag not in RESULTS:
        print(f"\nresults/ holds the paper's 50M training runs only, so it has no logged run of {tag} to set beside "
              "this one.")
        return
    if not os.path.isdir(f"{REPO_DIR}/results/{RESULTS[tag]}"):
        subprocess.run(["git", "-C", REPO_DIR, "sparse-checkout", "add", f"results/{RESULTS[tag]}"], check=True)
    ref = _rows(glob.glob(f"{REPO_DIR}/results/{RESULTS[tag]}/**/*.jsonl", recursive=True))
    arms = {r[0]["arm"] for r in mine.values()}
    if arms:
        ref = {b: r for b, r in ref.items() if r[0]["arm"] in arms}
    for key, what in KL_KEYS:
        _table(f"FROM THE PAPER, not run in this session (results/{RESULTS[tag]}"
               + (", the methods this run trained" if arms else "") + f"): {what}", ref, cache, key)
    print("\nt = 0 is the initialisation each branch trains from (Fisher label seed 20261014). The paper's tables "
          "print GaugeMLA's and A_p^mean's t = 0 as the mean over five draws of the Fisher's labels (five label "
          "seeds, this one the first); the other methods' t = 0 and every 50M value are these runs' own.")
    if test_tokens and mine:
        full = {max(r["tokens"] for r in rs) for rs in ref.values()} == {test_tokens}
        print(f"This was a test run of {test_tokens:,} training tokens: its t = 0 should be close to the paper's t = 0 "
              "above (the fits are redone on this machine, so expect differences in the last digits), while its end "
              + ("is at the paper's length, on the paper's schedule, so it is comparable with the paper's end too "
                 "(the training also runs on this machine, so expect small differences)." if full else
                 "is after a few optimiser steps and is not comparable with the paper's 50M."))


def _regauge_rows(tag, cond):
    """one regauge condition's scored rows on Drive, {branch: row} (each file's last row, as the run script reads it),
    or None when the condition has no folder yet"""
    runs = f"{DRIVE}/{FOLDER[tag]}/{tag}/runs"
    stamps = glob.glob(f"{runs}/grid-{tag}-*-g{cond}")
    if len(stamps) > 1:
        raise RuntimeError(f"{tag} {cond}: {len(stamps)} folders under {runs}, expected one: {stamps}")
    if not stamps:
        return None
    out = {}
    for f in glob.glob(f"{stamps[0]}/t0_only/*_t0only.jsonl"):
        rows = [json.loads(line) for line in open(f) if line.strip()]
        if rows:
            out[rows[-1]["branch"]] = rows[-1]
    return out


def regauge_table():
    """The regauge test's numbers from Drive, in the layout of the paper's two regauge tables, beside the values the
    paper prints."""
    got = {(t, c): _regauge_rows(t, c) for t, conds in REGAUGE["kits"].items() for c in conds}
    absent = [f"{t} {c}" for (t, c), v in got.items() if v is None]
    short = {}

    def value(tag, cond, branch, key):
        rows = got[(tag, cond)]
        if rows is None:
            return None
        if branch not in rows:
            short[f"{tag} {cond} {branch}"] = 1
            return None
        return rows[branch][key]

    n_full, n_conv = REGAUGE["tables"]
    rg = next(iter(REGAUGE["kits"]))
    cols = REGAUGE["full_cols"]
    w = 3 + max(len(h) for h, _ in cols)
    print(f"TABLE {n_full} OF THE PAPER: the regauged checkpoints themselves, uncompressed (100% of the KV cache)")
    print(f"{'':<18}{'this run':^{w * len(cols)}}    {'the paper':^{w * len(cols)}}")
    print(f"{'Regauge':<18}" + "".join(f"{h:>{w}}" for h, _ in cols) + "    " + "".join(f"{h:>{w}}" for h, _ in cols))
    for name, cond, paper in REGAUGE["full"]:
        line = f"{name:<18}"
        for _, key in cols:
            v = value(rg, cond, "full_cache_t0only", key)
            line += f"{'-':>{w}}" if v is None else (f"{v:>{w}.4f}" if key == "kl_mean" else f"{v:>{w}.3e}")
        print(line + "    " + "".join(f"{p:>{w}}" for p in paper))

    heads = [h for h, _ in REGAUGE["cols"]]
    print()
    print(f"TABLE {n_conv} OF THE PAPER: KL to the original model at conversion. {heads[0]} is the KL; "
          f"{' and '.join(heads[1:])} are its change under that gauge (regauged minus {heads[0].lower()}).")
    for j, (budget, g) in enumerate(REGAUGE["budgets"]):
        print()
        print(f"{'KV cache ' + budget:<30}{'this run':^30}    {'the paper':^30}")
        print(f"{'Method':<30}" + "".join(f"{h:>10}" for h in heads) + "    " + "".join(f"{h:>10}" for h in heads))
        for name, tag, arm, suf, paper in REGAUGE["conv"]:
            branch = f"{arm}_G{g}_t0only"
            base = value(tag, "identity", branch, "kl_orig_mean")
            line = f"{name:<30}" + (f"{'-':>10}" if base is None else f"{base:>10.4f}")
            for _, cond in REGAUGE["cols"][1:]:
                v = value(tag, cond + suf, branch, "kl_orig_mean")
                line += f"{'-':>10}" if v is None or base is None else f"{v - base:>+10.4f}"
            print(line + "    " + "".join(f"{p:>10}" for p in paper[3 * j:3 * j + 3]))
    print()
    print("Every checkpoint is rebuilt and scored on this machine, so expect differences from the paper's values in "
          "their last digits.")
    if absent:
        print(f"Not on Drive yet (run steps 1 and 2 for them): {', '.join(absent)}")
    if short:
        print(f"Missing from their condition's folder on Drive: {', '.join(short)}")
