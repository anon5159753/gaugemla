import gzip
import json
import os
import time

import numpy as np
import torch

C4_REPO = "allenai/c4"
C4_PATTERN = "en/c4-train.{:05d}-of-01024.json.gz"

def c4_shard_path(index):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C4_REPO, C4_PATTERN.format(index), repo_type="dataset")

def iter_docs_jsonl_gz(path, text_key="text"):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get(text_key)
            if t:
                yield t

def pack(tok, doc_iter, seq_len, max_blocks=None):
    eos = tok.eos_token_id
    buf, made = [], 0
    for text in doc_iter:
        ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len:
            yield np.asarray(buf[:seq_len], dtype=np.int64)
            del buf[:seq_len]
            made += 1
            if max_blocks is not None and made >= max_blocks:
                return

def load_c4_calibration(tok, n_seqs=128, seq_len=2048, shard=0, local_paths=None):
    paths = local_paths if local_paths is not None else [c4_shard_path(shard)]
    docs = (t for p in paths for t in iter_docs_jsonl_gz(p))
    blocks = list(pack(tok, docs, seq_len, max_blocks=n_seqs))
    if len(blocks) < n_seqs:
        raise RuntimeError(f"only {len(blocks)} blocks of seq_len={seq_len} available, need {n_seqs} "
                           f"-- pass more shards")
    return torch.tensor(np.stack(blocks[:n_seqs]), dtype=torch.long)

class XCapture:

    def __init__(self, model):
        import kvlab
        self.model = model
        self.layers = kvlab.get_layers(model)
        self.captured = {}
        self._handles = []

    def __enter__(self):
        for li, lay in enumerate(self.layers):
            def _hook(module, args, kwargs, li=li):
                x = args[0] if args else kwargs["hidden_states"]
                self.captured[li] = x.detach()
            attn = lay.self_attn if hasattr(lay, "self_attn") else lay.attention
            self._handles.append(attn.register_forward_pre_hook(_hook, with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

def capture_calibration_x(model, calib_ids, batch_size=8):
    import kvlab
    device = next(model.parameters()).device
    n_seqs = calib_ids.shape[0]
    out = {}
    n_batches = (n_seqs + batch_size - 1) // batch_size
    t_start = time.time()
    with torch.no_grad():
        for bi, s in enumerate(range(0, n_seqs, batch_size)):
            batch = calib_ids[s:s + batch_size].to(device)
            with XCapture(model) as cap:
                model(batch)
            for li, x in cap.captured.items():
                xb = x.cpu().float().numpy()
                if li not in out:
                    out[li] = np.empty((n_seqs,) + xb.shape[1:], dtype=np.float32)
                out[li][s:s + xb.shape[0]] = xb
            print(f"  [calibration capture] batch {bi + 1}/{n_batches} "
                  f"({time.time() - t_start:.1f}s total)")
    assert len(out) == len(kvlab.get_layers(model)), \
        f"captured {len(out)} layers, model has {len(kvlab.get_layers(model))}"
    return out

def _fixture(path, n_docs=40, words=400, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["the", "model", "attention", "gauge", "tokens", "layer", "value", "key", "query",
             "rotary", "corpus", "training", "damage", "head", "projection", "matrix"]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n_docs):
            txt = " ".join(str(vocab[j]) for j in rng.integers(0, len(vocab), size=words))
            f.write(json.dumps({"text": txt, "timestamp": "2019-01-01", "url": f"http://x/{i}"}) + "\n")
    return path

class _NoNetwork:

    def __enter__(self):
        self.saved = globals()["c4_shard_path"]

        def _refuse(index):
            raise AssertionError(
                f"reached the network for C4 shard {index}; this is supposed to run entirely "
                "against the synthetic fixture"
            )
        globals()["c4_shard_path"] = _refuse
        return self

    def __exit__(self, *a):
        globals()["c4_shard_path"] = self.saved
        return False
