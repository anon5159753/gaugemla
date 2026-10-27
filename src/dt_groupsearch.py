import time

import numpy as np

import gam_core as gc
import kvlab

def _tail(Maps, A, g, d):
    if len(g) == 1:
        return 0.0
    Z = np.concatenate([gc.psd_pow(A[i], 0.5) @ Maps[i] for i in g], axis=0)
    tail, _ = gc.tail_svd(Z, d)
    return tail

def score_weights_invariant(Kt, AK, Vt, AV, groups, d_k, d_v, kv_weight=0.5):
    total_k = sum(_tail(Kt, AK, g, d_k) for g in groups)
    total_v = sum(_tail(Vt, AV, g, d_v) for g in groups)
    return kv_weight * total_k + (1.0 - kv_weight) * total_v

def group_scorer_weights_invariant(Kt, AK, Vt, AV, d_k, d_v, kv_weight=0.5):
    def group_score(g):
        return kv_weight * _tail(Kt, AK, g, d_k) + (1.0 - kv_weight) * _tail(Vt, AV, g, d_v)
    return group_score

def score_simk(S, groups):
    total = 0.0
    for g in groups:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                total += S[g[a], g[b]]
    return total

def group_scorer_simk(S):
    def group_score(g):
        total = 0.0
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                total += S[g[a], g[b]]
        return total
    return group_score

def search(H, G, group_score_fn, direction="min", n_restarts=8, seed=0, max_evals_per_restart=4000,
           label=None):
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
    better = (lambda a, b: a < b) if direction == "min" else (lambda a, b: a > b)
    rng = np.random.default_rng(seed)

    def local_search(groups0):
        groups = [list(g) for g in groups0]
        head_of = {h: gi for gi, g in enumerate(groups) for h in g}
        group_scores = [group_score_fn(g) for g in groups]
        total = sum(group_scores)
        n_evals = len(groups)
        improved = True
        while improved and n_evals < max_evals_per_restart:
            improved = False
            for h in range(H):
                gi = head_of[h]
                swapped = False
                for gj in range(len(groups)):
                    if gj == gi:
                        continue
                    for h2 in list(groups[gj]):
                        if n_evals >= max_evals_per_restart:
                            break
                        new_gi = [h2 if x == h else x for x in groups[gi]]
                        new_gj = [h if x == h2 else x for x in groups[gj]]
                        s_gi, s_gj = group_score_fn(new_gi), group_score_fn(new_gj)
                        n_evals += 2
                        new_total = total - group_scores[gi] - group_scores[gj] + s_gi + s_gj
                        if better(new_total, total):
                            groups[gi], groups[gj] = new_gi, new_gj
                            group_scores[gi], group_scores[gj] = s_gi, s_gj
                            total = new_total
                            head_of[h], head_of[h2] = gj, gi
                            improved = True
                            swapped = True
                            break
                    if swapped or n_evals >= max_evals_per_restart:
                        break
        return groups, total, n_evals

    per = H // G
    starts = [kvlab.groups_of(H, G)]
    for _ in range(max(n_restarts - 1, 0)):
        perm = rng.permutation(H)
        starts.append([sorted(perm[j * per:(j + 1) * per].tolist()) for j in range(G)])

    if label is not None:
        print(f"    [{label}] search: H={H} G={G} n_restarts={len(starts)} "
              f"max_evals_per_restart={max_evals_per_restart}")
    best_groups, best_score = None, None
    for ri, s0 in enumerate(starts):
        t0 = time.time()
        g, sc, n_evals = local_search(s0)
        if label is not None:
            converged = "converged" if n_evals < max_evals_per_restart else "hit max_evals"
            print(f"    [{label}] restart {ri + 1}/{len(starts)}: {n_evals} evals ({converged}), "
                  f"score={sc:.4f}, {time.time() - t0:.1f}s")
        if best_score is None or better(sc, best_score):
            best_groups, best_score = g, sc
    return best_groups
