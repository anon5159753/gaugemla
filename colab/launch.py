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
    "slm1p6b_urx1_fi": "init_ablation",
    "p410m_urx1_fi": "init_ablation",
    "tl1b_tx1_fi": "init_ablation",
    "sm360m_tx1_fi": "init_ablation",
    "olmo1b_tx1_fi": "init_ablation",
    "q0p5b_tx1_fi": "init_ablation",
    "l3_1b_tx1_fi": "init_ablation",
    "slm1p6b_urx1_si": "init_ablation",
    "p410m_urx1_si": "init_ablation",
    "tl1b_tx1_si": "init_ablation",
    "sm360m_tx1_si": "init_ablation",
    "olmo1b_tx1_si": "init_ablation",
    "q0p5b_tx1_si": "init_ablation",
    "l3_1b_tx1_si": "init_ablation",
    "slm1p6b_urx1_ii": "init_ablation",
    "p410m_urx1_ii": "init_ablation",
    "tl1b_tx1_ii": "init_ablation",
    "sm360m_tx1_ii": "init_ablation",
    "olmo1b_tx1_ii": "init_ablation",
    "q0p5b_tx1_ii": "init_ablation",
    "l3_1b_tx1_ii": "init_ablation",
    "slm1p6b_urx1_seeds": "true_fisher_x1_seeds",
    "p410m_urx1_seeds": "true_fisher_x1_seeds",
    "tl1b_tx1_seeds": "true_fisher_x1_seeds",
    "sm360m_tx1_seeds": "true_fisher_x1_seeds",
    "olmo1b_tx1_seeds": "true_fisher_x1_seeds",
    "q0p5b_tx1_seeds": "true_fisher_x1_seeds",
    "l3_1b_tx1_seeds": "true_fisher_x1_seeds"
}
NAMES = {"mla_w64_ours_unrope_cost": "GaugeMLA", "mla_w64_transmla_sigma": "TransMLA",
         "mla_w64_mha2mla_s2norm": "MHA2MLA", "mla_w64_care_sigma": "CARE",
         "mla_ours_unrope_cost": "GaugeMLA, KL width", "decoupled_optwide_fisher_sigma_dkr64": "Decoupled, KL objective",
         "pool": "GQA (pool)", "gauge3B": "GQA + optimal gauge", "procrustes_simk": "Procrustes alignment",
         "optimal": "Per-group subspace", "optwide": "Joint key, joint value subspace", "mha": "MHA control"}
ORDER = ["pool", "gauge3B", "procrustes_simk", "optimal", "optwide", "decoupled_optwide_fisher_sigma_dkr64",
         "mla_w64_ours_unrope_cost", "mla_ours_unrope_cost", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm",
         "mla_w64_care_sigma", "mha"]
SOURCE_CACHE = {"p410m_gc": 2048, "p410m_gc5m": 2048}


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


def _table(title, rows, cache):
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
            t0 = next((r["kl_mean"] for r in rs if r["tokens"] == 0), None)
            end = max(rs, key=lambda r: r["tokens"])
            ends.add(end["tokens"])
            line += (f"{t0:>11.4f}" if t0 is not None else f"{'-':>11}") + f"{end['kl_mean']:>11.4f}"
        print(line)
    print(f"end of training: {', '.join(f'{t:,}' for t in sorted(ends))} tokens")


def kl_table(tag, test_tokens=0):
    """KL to the teacher at t = 0 and at the end of training: this run (Drive) beside the paper's (results/)."""
    runs = f"{DRIVE}/{FOLDER[tag]}/{tag}/runs"
    stamps = [s for s in glob.glob(f"{runs}/grid-{tag}-*")
              if (s.endswith(f"-test{test_tokens}") if test_tokens else "-test" not in os.path.basename(s))]
    cache = SOURCE_CACHE.get(tag)
    mine = {}
    if len(stamps) != 1:
        print(f"this run: {len(stamps)} matching stamp folders under {runs}: {stamps}")
    else:
        mine = _rows(glob.glob(f"{stamps[0]}/kd_*.jsonl"))
        _table(f"this run ({os.path.basename(stamps[0])}): KL to the teacher, mean over the eval documents", mine, cache)
    if not os.path.isdir(f"{REPO_DIR}/results/{tag}"):
        subprocess.run(["git", "-C", REPO_DIR, "sparse-checkout", "add", f"results/{tag}"], check=True)
    ref = _rows(glob.glob(f"{REPO_DIR}/results/{tag}/runs/**/*.jsonl", recursive=True))
    arms = {r[0]["arm"] for r in mine.values()}
    if arms:
        ref = {b: r for b, r in ref.items() if r[0]["arm"] in arms}
    _table(f"FROM THE PAPER, not run in this session: the logged run in results/{tag}"
           + (", the methods this run trained" if arms else ""), ref, cache)
    print("t = 0 is the draw each branch trains from; the paper's GaugeMLA t = 0 entries average five draws "
          "(results/tfx1_seeds).")
