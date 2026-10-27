import time

import numpy as np

import dt_calib
import gam_core as gc
import kvlab

def _orthogonal_procrustes(X, Y, rotation_only):
    M_ = Y @ X.T
    U, _, Vt = np.linalg.svd(M_)
    if rotation_only and np.linalg.det(U @ Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1
    return U @ Vt

def solve_plane_rotation(X, Y):
    Q = _orthogonal_procrustes(X, Y, rotation_only=True)
    return float(Q[0, 0]), float(Q[1, 0])

def solve_free_block(X, Y):
    return _orthogonal_procrustes(X, Y, rotation_only=False)

def solve_head_rotation(X, Y, r):
    d_k = X.shape[0]
    h = r // 2
    Q = np.zeros((d_k, d_k))
    for p in range(h):
        idx = [p, p + h]
        c, s = solve_plane_rotation(X[idx, :], Y[idx, :])
        Q[idx[0], idx[0]], Q[idx[1], idx[1]] = c, c
        Q[idx[0], idx[1]], Q[idx[1], idx[0]] = -s, s
    if d_k > r:
        free = list(range(r, d_k))
        Q[np.ix_(free, free)] = solve_free_block(X[free, :], Y[free, :])
    return Q

def solve_value_rotation(X, Y):
    return _orthogonal_procrustes(X, Y, rotation_only=False)

def _gpa(head_indices, Act, solve_fn, n_iters=5):
    ref = Act[head_indices[0]].copy()
    Q = {}
    for _ in range(n_iters):
        aligned = {}
        for i in head_indices:
            Qi = solve_fn(Act[i], ref)
            Q[i] = Qi
            aligned[i] = Qi @ Act[i]
        ref = np.mean(list(aligned.values()), axis=0)
    return Q

def align_group_gpa_key(head_indices, Kact, r, n_iters=5):
    return _gpa(head_indices, Kact, lambda X, Y: solve_head_rotation(X, Y, r), n_iters)

def align_group_gpa_value(head_indices, Vact, n_iters=5):
    return _gpa(head_indices, Vact, lambda X, Y: solve_value_rotation(X, Y), n_iters)

def layer_activations(lay, H, d_k, d_v, dm, X):
    Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
    Xflat = X.reshape(-1, X.shape[-1])
    Xaug = np.empty((Xflat.shape[1] + 1, Xflat.shape[0]), dtype=np.float64)
    Xaug[:-1, :] = Xflat.T
    Xaug[-1, :] = 1.0
    Kact = {i: Kt[i] @ Xaug for i in range(H)}
    Vact = {i: Vt[i] @ Xaug for i in range(H)}
    return Kact, Vact

def simk_matrix(Kact, H):
    normed = {i: Kact[i] / (np.linalg.norm(Kact[i], axis=0, keepdims=True) + 1e-12) for i in range(H)}
    S = np.eye(H)
    for i in range(H):
        for j in range(i + 1, H):
            s = float((normed[i] * normed[j]).sum(axis=0).mean())
            S[i, j] = S[j, i] = s
    return S

def _pool_groups(layer, H, d_k, dm, groups):
    Wq, Wk, Wv, bq, bk, bv = gc.qkv_split(layer, H, d_k, dm)
    Wk, Wv, bk, bv = Wk.copy(), Wv.copy(), bk.copy(), bv.copy()
    for idx in groups:
        Wk[idx] = Wk[idx].mean(axis=0, keepdims=True)
        bk[idx] = bk[idx].mean(axis=0, keepdims=True)
        Wv[idx] = Wv[idx].mean(axis=0, keepdims=True)
        bv[idx] = bv[idx].mean(axis=0, keepdims=True)
    gc.qkv_merge(layer, Wq, Wk, Wv, bq, bk, bv, H, d_k, dm)

def fuse_and_pool(layer, H, d_k, d_v, dm, Qk, Qv, groups):
    Qt, Kt, Vt, Wo = gc.read_layer(layer, H, d_k, d_v, dm)
    Om = np.stack([Qk[i] for i in range(H)])
    Oi = np.stack([Qk[i].T for i in range(H)])
    Gm = np.stack([Qv[i] for i in range(H)])
    Gi = np.stack([Qv[i].T for i in range(H)])
    Qh, Kh, Vh, Oh = gc.apply_gauge(Qt, Kt, Vt, Wo, Om, Oi, Gm, Gi)
    gc.write_layer(layer, Qh, Kh, Vh, Oh, H, d_k, d_v, dm)
    _pool_groups(layer, H, d_k, dm, groups)

def build(model, G, grouping=None, groups_by_layer=None, calib_batch=None, n_gpa_iters=5):
    if calib_batch is None:
        raise ValueError("dt_procrustes.build requires calib_batch (dt_calib.load_c4_calibration)")

    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if groups_by_layer is not None and len(groups_by_layer) != L:
        raise ValueError(f"groups_by_layer has {len(groups_by_layer)} entries, model has {L} layers")
    single = grouping if grouping is not None else kvlab.groups_of(H, G)
    for gl in (groups_by_layer if groups_by_layer is not None else [single]):
        assert sorted(i for g in gl for i in g) == list(range(H)), "groups must partition all heads"

    x_by_layer = dt_calib.capture_calibration_x(model, calib_batch)

    t_start = time.time()
    for li, lay in enumerate(layers):
        t0 = time.time()
        groups = groups_by_layer[li] if groups_by_layer is not None else single
        Kact, Vact = layer_activations(lay, H, d_k, d_v, dm, x_by_layer.pop(li))
        Qk, Qv = {}, {}
        for g in groups:
            if len(g) == 1:
                Qk[g[0]], Qv[g[0]] = np.eye(d_k), np.eye(d_v)
                continue
            Qk.update(align_group_gpa_key(g, Kact, r, n_gpa_iters))
            Qv.update(align_group_gpa_value(g, Vact, n_gpa_iters))
        fuse_and_pool(lay, H, d_k, d_v, dm, Qk, Qv, groups)
        print(f"  [procrustes build] layer {li + 1}/{L} done ({time.time() - t0:.1f}s, "
              f"{time.time() - t_start:.1f}s total)")

    part = "per-layer search" if groups_by_layer is not None else \
           ("contiguous" if grouping is None else "custom")
    return f"Procrustes rotate(GPA,{n_gpa_iters}iters)+pool G={G} grouping={part}"
