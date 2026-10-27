import numpy as np
import torch

MAX_BATCH = 16
MAX_BATCH_TOKENS = 1536
MAX_LEN = 2048

def _encode(tok, ctx, cont):
    whole = tok(ctx + cont, add_special_tokens=False)["input_ids"]
    c = tok(ctx, add_special_tokens=False)["input_ids"]
    n = len(c)
    if whole[:n] != c:
        n = 0
        while n < min(len(whole), len(c)) and whole[n] == c[n]:
            n += 1
    return whole[:n], whole[n:]

def _requests(tok, items):
    reqs = []
    for i, it in enumerate(items):
        for j, cont in enumerate(it["conts"]):
            ce, ke = _encode(tok, it["ctx"], " " + cont)
            if len(ke) == 0:
                ke, ce = ce[-1:], ce[:-1]
            inp = (ce + ke)[-(MAX_LEN + 1):][:-1]
            reqs.append({"i": i, "j": j, "inp": inp, "cont": ke,
                         "inplen": len(inp), "contlen": len(ke),
                         "nbytes": float(len(cont))})
    return reqs

def _batches(reqs):
    order = sorted(range(len(reqs)), key=lambda k: -reqs[k]["inplen"])
    cur = []
    for k in order:
        T = reqs[k]["inplen"]
        if cur and (len(cur) + 1 > MAX_BATCH or (len(cur) + 1) * T > MAX_BATCH_TOKENS):
            yield cur
            cur = []
        cur.append(k)
    if cur:
        yield cur

@torch.no_grad()
def score(model, tok, items, progress=None):
    device = next(model.parameters()).device
    reqs = _requests(tok, items)
    lls = [[0.0] * len(it["conts"]) for it in items]
    grd = [[False] * len(it["conts"]) for it in items]
    done = 0
    for batch in _batches(reqs):
        T = max(reqs[k]["inplen"] for k in batch)
        inp = torch.zeros((len(batch), T), dtype=torch.long)
        for b, k in enumerate(batch):
            inp[b, :reqs[k]["inplen"]] = torch.tensor(reqs[k]["inp"], dtype=torch.long)
        out = model(inp.to(device)).logits.float()
        for b, k in enumerate(batch):
            r = reqs[k]
            sl = out[b, r["inplen"] - r["contlen"]:r["inplen"], :]
            sl = torch.log_softmax(sl, dim=-1)
            ct = torch.tensor(r["cont"], dtype=torch.long, device=device)
            grd[r["i"]][r["j"]] = bool((sl.argmax(dim=-1) == ct).all())
            lls[r["i"]][r["j"]] = float(sl.gather(1, ct.unsqueeze(-1)).sum())
        done += len(batch)
        del out
        if progress and done % (50 * MAX_BATCH) < MAX_BATCH:
            progress(done, len(reqs))
    return lls, grd

def evaluate_mc(items, lls, grd):
    assert all(len(it["conts"]) > 1 for it in items), \
        "evaluate_mc got single-continuation items -- argmax over one option always equals gold, " \
        "so accuracy would read 1.0 regardless of the model. Score these by greedy exact match " \
        "(evaluate_lambada); see evaluate()'s shape dispatch."
    n = len(items)
    acc = np.zeros(n, dtype=bool)
    accn = np.zeros(n, dtype=bool)
    marg = np.zeros(n)
    for i, it in enumerate(items):
        g = it["gold"]
        v = np.array(lls[i], dtype=float)
        nb = np.array([max(len(c), 1) for c in it["conts"]], dtype=float)
        acc[i] = int(np.argmax(v)) == g
        accn[i] = int(np.argmax(v / nb)) == g
        other = np.delete(v, g)
        marg[i] = v[g] - (other.max() if len(other) else 0.0)
    return {"acc": acc, "acc_norm": accn, "margin": marg}

def evaluate_lambada(items, lls, grd):
    acc = np.array([bool(g[0]) for g in grd], dtype=bool)
    ll = np.array([v[0] for v in lls], dtype=float)
    return {"acc": acc, "acc_norm": acc, "margin": ll, "ll": ll,
            "ppl": float(np.exp(-ll.mean()))}

def evaluate(task, items, lls, grd):
    single_cont = bool(items) and len(items[0]["conts"]) == 1
    return (evaluate_lambada if single_cont else evaluate_mc)(items, lls, grd)

def summarise(task, m):
    out = {"n": int(len(m["acc"])), "acc": float(m["acc"].mean()),
           "acc_norm": float(m["acc_norm"].mean()), "margin": float(m["margin"].mean())}
    if "ppl" in m:
        out["ppl"] = m["ppl"]
    return out
