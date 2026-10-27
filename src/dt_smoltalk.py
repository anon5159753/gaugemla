import json
import os

import dt_data

REPO = "HuggingFaceTB/smoltalk"

SUBSETS = [
    "apigen-80k", "everyday-conversations", "explore-instruct-rewriting", "longalign",
    "metamathqa-50k", "numina-cot-100k", "openhermes-100k", "self-oss-instruct",
    "smol-constraints", "smol-magpie-ultra", "smol-rewrite", "smol-summarize", "systemchats-30k",
]

ROLE_TAGS = {
    "system": "\n<|system|>\n",
    "user": "\n<|user|>\n",
    "assistant": "\n<|assistant|>\n",
}

N_DEFAULT = 20_000

def fetch_all_subsets():
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq

    files = list_repo_files(REPO, repo_type="dataset")
    out = {}
    for subset in SUBSETS:
        prefix = f"data/{subset}/train-"
        shard_files = sorted(f for f in files if f.startswith(prefix))
        if not shard_files:
            raise RuntimeError(f"no train shards found for subset {subset!r} under {prefix}")
        rows = []
        for sf in shard_files:
            p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=sf,
                                 token=dt_data.hf_token())
            rows.extend(pq.read_table(p).to_pylist())
        out[subset] = rows
    return out

def _stratified_sample(all_rows, n, seed=0):
    total = sum(len(v) for v in all_rows.values())
    out = []
    for subset in sorted(all_rows):
        rows = all_rows[subset]
        if not rows or total == 0:
            continue
        quota = min(round(n * len(rows) / total), len(rows))
        sampled, _ = dt_data.subsample(rows, quota, seed)
        for r in sampled:
            out.append({"subset": subset, "messages": r["messages"]})
    return out

def cache_path(n=N_DEFAULT):
    return os.path.join(dt_data.D_DATA, f"smoltalk_{n}.json")

def build_and_cache(n=N_DEFAULT, seed=0, out_path=None):
    all_rows = fetch_all_subsets()
    out = _stratified_sample(all_rows, n, seed)
    p = out_path or cache_path(n)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out, p

def load_cached(n=N_DEFAULT):
    p = cache_path(n)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run `python dt_smoltalk.py` first")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _turn_text(turns, upto):
    s = ""
    for t in turns[:upto]:
        s += ROLE_TAGS[t["role"]] + t["content"]
    return s

def render_and_label(tok, turns, add_eos=True):
    convo_text = _turn_text(turns, len(turns))
    full_text = convo_text + (tok.eos_token if add_eos else "")
    whole = tok(full_text, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(whole)
    prev_n = 0
    for i, t in enumerate(turns):
        prefix_ids = tok(_turn_text(turns, i + 1), add_special_tokens=False)["input_ids"]
        n = len(prefix_ids)
        if whole[:n] != prefix_ids:
            n = 0
            while n < min(len(whole), len(prefix_ids)) and whole[n] == prefix_ids[n]:
                n += 1
        if t["role"] == "assistant":
            for j in range(prev_n, n):
                labels[j] = whole[j]
        prev_n = n
    if add_eos and turns and turns[-1]["role"] == "assistant":
        for j in range(prev_n, len(whole)):
            labels[j] = whole[j]
    return whole, labels

def collate_batch(tok, encoded, max_len=None):
    import torch

    lens = [len(ids) for ids, _ in encoded]
    T = max(lens) if max_len is None else min(max(lens), max_len)
    B = len(encoded)
    input_ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for b, (ids, lab) in enumerate(encoded):
        n = min(len(ids), T)
        input_ids[b, :n] = torch.tensor(ids[:n], dtype=torch.long)
        labels[b, :n] = torch.tensor(lab[:n], dtype=torch.long)
        attn[b, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

def main():
    print(f"dt_smoltalk -- fetching {len(SUBSETS)} subsets from {REPO}, "
          f"token {'found' if dt_data.hf_token() else 'NOT found (public repo, should still work)'}")
    out, p = build_and_cache()
    counts = {}
    for r in out:
        counts[r["subset"]] = counts.get(r["subset"], 0) + 1
    print(f"wrote {p}  ({len(out)} examples)")
    for s in SUBSETS:
        print(f"  {s:<28} {counts.get(s, 0):>6}")

if __name__ == "__main__":
    main()
