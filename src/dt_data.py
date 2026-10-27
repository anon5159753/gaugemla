import argparse
import io
import json
import os
import re

import numpy as np

D_HERE = os.path.dirname(os.path.abspath(__file__))
D_DATA = os.environ.get("KVTYING_DATA_DIR", os.path.join(D_HERE, "data"))
TOKEN_FILE = os.path.abspath(os.path.join(D_HERE, "..", "..", "HF_token.txt"))

PUBLISHED = {
    "pythia-410m": {"lambada": {"acc": 0.5005, "ppl": 11.7510},
                    "sciq": {"acc": 0.8080, "acc_norm": 0.7210},
                    "arc_easy": {"acc": 0.5122, "acc_norm": 0.4453},
                    "piqa": {"acc": 0.6670, "acc_norm": 0.6681}},
    "pythia-1.4b": {"lambada": {"acc": 0.5919, "ppl": 6.7072},
                    "sciq": {"acc": 0.8730, "acc_norm": 0.8150},
                    "arc_easy": {"acc": 0.5943, "acc_norm": 0.5295},
                    "piqa": {"acc": 0.7046, "acc_norm": 0.7035}},
}

CHANCE = {"lambada": 0.0, "sciq": 0.25, "arc_easy": 0.25, "piqa": 0.50}

TASKS = ["lambada", "sciq", "arc_easy", "piqa"]

def hf_token():
    if os.path.exists(TOKEN_FILE):
        t = open(TOKEN_FILE, "r", encoding="utf-8").read().strip()
        if t:
            return t
    return os.environ.get("HF_TOKEN") or None

def _dl(repo, fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", token=hf_token())

def _parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()

def build_lambada():
    p = _dl("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl")
    out = []
    with io.open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"].strip()
            ctx, last = text.rsplit(" ", 1)
            out.append({"ctx": ctx, "conts": [last], "gold": 0})
    return out

def build_sciq():
    rows = _parquet(_dl("allenai/sciq", "data/test-00000-of-00001.parquet"))
    out = []
    for d in rows:
        ctx = f"{d['support'].lstrip()}\nQuestion: {d['question']}\nAnswer:"
        out.append({"ctx": ctx,
                    "conts": [d["distractor1"], d["distractor2"], d["distractor3"],
                              d["correct_answer"]],
                    "gold": 3})
    return out

def build_arc_easy():
    rows = _parquet(_dl("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"))
    num2let = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    out = []
    for d in rows:
        key = num2let.get(d["answerKey"], d["answerKey"])
        letters = ["A", "B", "C", "D", "E"]
        if key not in letters:
            continue
        gold = letters.index(key)
        texts = list(d["choices"]["text"])
        if gold >= len(texts):
            continue
        out.append({"ctx": "Question: " + d["question"] + "\nAnswer:",
                    "conts": texts, "gold": gold})
    return out

def build_piqa():
    rows = _parquet(_dl("baber/piqa", "piqa_validation.parquet"))
    return [{"ctx": "Question: " + d["goal"] + "\nAnswer:",
             "conts": [d["sol1"], d["sol2"]], "gold": int(d["label"])} for d in rows]

def build_boolq():
    rows = _parquet(_dl("aps/super_glue", "boolq/validation-00000-of-00001.parquet"))
    out = []
    for d in rows:
        ctx = f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:"
        out.append({"ctx": ctx, "conts": ["no", "yes"], "gold": int(d["label"])})
    return out

def build_siqa():
    rows = _parquet(_dl("lighteval/siqa", "data/validation-00000-of-00001-90bb85eab85fcfa8.parquet"))
    out = []
    for d in rows:
        ctx = f"Question: {d['context']} {d['question']}\nAnswer:"
        out.append({"ctx": ctx, "conts": [d["answerA"], d["answerB"], d["answerC"]],
                    "gold": int(d["label"]) - 1})
    return out

def build_openbookqa():
    rows = _parquet(_dl("allenai/openbookqa", "main/test-00000-of-00001.parquet"))
    out = []
    for d in rows:
        labels, texts = list(d["choices"]["label"]), list(d["choices"]["text"])
        out.append({"ctx": d["question_stem"], "conts": texts, "gold": labels.index(d["answerKey"])})
    return out

def _hellaswag_preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text

def build_hellaswag():
    rows = _parquet(_dl("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"))
    out = []
    for d in rows:
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        query = _hellaswag_preprocess(d["activity_label"] + ": " + ctx)
        conts = [_hellaswag_preprocess(e) for e in d["endings"]]
        out.append({"ctx": query, "conts": conts, "gold": int(d["label"])})
    return out

_SYNTH_MODEL = "EleutherAI/pythia-410m"
_SYNTH_TOKEN_MIN, _SYNTH_TOKEN_MAX = 2, 50253
_SYNTH_VALUE_LO, _SYNTH_VALUE_HI = 100, 500

def _synth_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_SYNTH_MODEL)

def _synth_key_pool(tok, min_chars=4, min_id=2000):
    out = []
    for i in range(min_id, _SYNTH_TOKEN_MAX + 1):
        s = tok.decode([i])
        if not s.startswith(" "):
            continue
        w = s[1:]
        if len(w) >= min_chars and w.isalpha() and w.islower() and w.isascii():
            out.append((i, w))
    return out

def build_induction(n_items=2000, seq_len=48, prefix=4, m=8, seed=0, control=False):
    tok = _synth_tokenizer()
    ids = [i for i, _ in _synth_key_pool(tok)]
    rng = np.random.default_rng(seed if not control else seed + 10_000)
    out = []
    for _ in range(n_items):
        s = [ids[j] for j in rng.choice(len(ids), size=seq_len, replace=False)]
        head = tok.decode(s[:prefix])
        target = tok.decode(s[prefix:prefix + m])
        ctx = head if control else tok.decode(s) + head
        out.append({"ctx": ctx.rstrip(), "conts": [target.lstrip()], "gold": 0})
    return out

def build_kv_synth_N50(n_items=2000, n_pairs=50, seed=0):
    tok = _synth_tokenizer()
    words = [w for _, w in _synth_key_pool(tok)]
    rng = np.random.default_rng(seed)
    values = list(range(_SYNTH_VALUE_LO, _SYNTH_VALUE_HI))
    out = []
    for _ in range(n_items):
        ks = rng.choice(len(words), size=n_pairs, replace=False)
        vs = rng.choice(len(values), size=n_pairs, replace=False)
        q = int(rng.integers(0, n_pairs))
        lines = "".join(f"The code for {words[k]} is {values[v]}.\n" for k, v in zip(ks, vs))
        out.append({"ctx": lines + f"The code for {words[ks[q]]} is",
                    "conts": [str(values[vs[q]])], "gold": 0})
    return out

def build_arith_1d(n_items=2000, k_shot=8, seed=0, op="plus"):
    rng = np.random.default_rng(seed)
    fn = (lambda a, b: a + b) if op == "plus" else (lambda a, b: a - b)

    def one():
        a, b = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        return a, b, fn(a, b)

    out = []
    for _ in range(n_items):
        shots = []
        for _ in range(k_shot):
            a, b, c = one()
            shots.append(f"\n\nQ: What is {a} {op} {b}?\n\nA: {c}")
        a, b, c = one()
        out.append({"ctx": "".join(shots) + f"\n\nQ: What is {a} {op} {b}?\n\nA:",
                    "conts": [str(c)], "gold": 0})
    return out

def build_swde(window=2048):
    rows = _parquet(_dl("hazyresearch/based-swde", "data/validation-00000-of-00001.parquet"))
    out, dropped = [], 0
    for r in rows:
        ctx = r["text"].replace("﻿", "").strip()
        val = r["value"].strip()
        if not ctx or not val:
            dropped += 1
            continue
        out.append({"ctx": ctx, "conts": [val], "gold": 0})
    if dropped:
        print(f"  swde: dropped {dropped} empty item(s)")
    return out

def build_arithmetic_2da():
    p = _dl("EleutherAI/arithmetic", "data/two_digit_addition.jsonl")
    out = []
    with io.open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({"ctx": r["context"].rstrip(), "conts": [r["completion"].strip()], "gold": 0})
    return out

BUILDERS = {"lambada": build_lambada, "sciq": build_sciq,
            "arc_easy": build_arc_easy, "piqa": build_piqa,
            "boolq": build_boolq, "siqa": build_siqa,
            "openbookqa": build_openbookqa, "hellaswag": build_hellaswag,
            "induction": build_induction, "kv_synth_N50": build_kv_synth_N50,
            "arith_1d": build_arith_1d, "swde": build_swde,
            "arithmetic_2da": build_arithmetic_2da}

ALL_TASKS = list(BUILDERS)

FEASIBILITY_TASKS = ["lambada", "piqa", "arc_easy", "boolq", "siqa", "openbookqa", "hellaswag",
                     "induction", "kv_synth_N50", "arith_1d"]
CHANCE.update({"boolq": 0.50, "siqa": 1 / 3, "openbookqa": 0.25, "hellaswag": 0.25,
               "induction": 0.0, "kv_synth_N50": 0.0, "arith_1d": 0.098})

PRIMARY_METRIC = {"openbookqa": "acc_norm", "hellaswag": "acc_norm"}

def primary_metric(task):
    return PRIMARY_METRIC.get(task, "acc")

def select_tasks(include=None, exclude=None, base=None, everything=False):
    if base is None:
        base = ALL_TASKS if everything else FEASIBILITY_TASKS
    base = list(base)
    unknown = [t for t in list(include or []) + list(exclude or []) if t not in BUILDERS]
    if unknown:
        raise KeyError(f"unknown task(s) {unknown} -- known: {sorted(BUILDERS)}")
    chosen = set(include) if include is not None else set(base)
    chosen -= set(exclude or [])
    out = [t for t in base if t in chosen]
    out += [t for t in (include or []) if t not in base]
    if not out:
        raise ValueError("select_tasks() selected nothing -- check include/exclude")
    return out

def subsample(items, n, seed=0):
    if n is None or n >= len(items):
        return items, list(range(len(items)))
    idx = sorted(np.random.default_rng(seed).permutation(len(items))[:n].tolist())
    return [items[i] for i in idx], idx

def load(task, n=None, seed=0):
    p = os.path.join(D_DATA, f"{task}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run dt_data.py first")
    items = json.load(open(p, "r", encoding="utf-8"))
    return subsample(items, n, seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    os.makedirs(D_DATA, exist_ok=True)
    print(f"dt_data -- token {'found' if hf_token() else 'NOT found (public repos only)'}")
    for t in a.tasks.split(","):
        items = BUILDERS[t]()
        with open(os.path.join(D_DATA, f"{t}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f)
        nopt = sorted({len(i["conts"]) for i in items})
        clen = int(np.mean([len(i["ctx"]) for i in items]))
        print(f"  {t:<10} {len(items):>5} items   options {nopt}   mean ctx {clen:>5} chars   "
              f"chance {CHANCE[t]:.2f}")

if __name__ == "__main__":
    main()
