import numpy as np

import gam_core as gc
import kvlab
import dt_optimal

W_KV_DEFAULT = 1.0

DKR_DEFAULT = None

def _widths(G, H, d_k, d_v, r, d_k_R, B=None):
    d_k_R = r if d_k_R is None else int(d_k_R)
    if B is None:
        B = G * (d_k + d_v)
    elif G is not None and B != G * (d_k + d_v):
        raise ValueError(f"budget B = {B} != G*(d_k+d_v) = {G * (d_k + d_v)} at G={G}")
    c_KV = B - d_k_R
    if c_KV <= 0:
        raise ValueError(f"d_k^R={d_k_R} leaves no content width at G={G} (total cache {B})")
    rot_rank = H * r
    if d_k_R > rot_rank:
        raise ValueError(
            f"d_k^R={d_k_R} exceeds rank Xi_rot <= H*r = {H}*{r} = {rot_rank}: the rotary rows of all "
            f"{H} heads span at most {rot_rank} directions, so a wider rotary cache is asking "
            f"tail_svd for singular vectors that do not exist -- it would return fewer rows than "
            f"requested and every shape built from d_k^R would silently disagree.")
    return d_k_R, c_KV

def _widths_by_layer(G, H, d_k, d_v, r, d_k_R, L, B=None):
    scalar = d_k_R is None or isinstance(d_k_R, (int, np.integer))
    if scalar:
        return [_widths(G, H, d_k, d_v, r, d_k_R, B=B)] * L
    seq = [int(x) for x in d_k_R]
    if len(seq) != L:
        raise ValueError(f"d_k^R has {len(seq)} entries but the model has {L} layers; a per-layer "
                         f"width must name every layer, not a prefix")
    return [_widths(G, H, d_k, d_v, r, x, B=B) for x in seq]

def _metrics(Qt, Wo, H, r, theta, wD, GK=None, GV=None):
    if (GK is None) != (GV is None):
        raise ValueError("GK and GV must be supplied together: a measured key metric against a "
                         "guessed value metric prices the two families on different scales, which "
                         "is exactly the exchange rate w, silently re-introduced")
    d_k_full = np.shape(Qt)[1]
    if r > d_k_full:
        raise ValueError(f"r = {r} exceeds d_k = {d_k_full}")
    if r == d_k_full:
        if GK is not None:
            return (np.ascontiguousarray(GK[:, r:, r:]), np.ascontiguousarray(GV),
                    np.ascontiguousarray(GK[:, :r, :r]))
        A_raw = np.stack([Qt[i] @ Qt[i].T for i in range(H)])
        A_avg = np.stack([dt_optimal._rope_avg(A_raw[i], r, theta, wD) for i in range(H)])
        return (A_raw[:, r:, r:], np.stack([Wo[i].T @ Wo[i] for i in range(H)]), A_avg[:, :r, :r])
    if GK is not None:
        return (np.ascontiguousarray(GK[:, r:, r:]),
                np.ascontiguousarray(GV),
                np.ascontiguousarray(GK[:, :r, :r]))
    A_raw = np.stack([Qt[i] @ Qt[i].T for i in range(H)])
    A_avg = np.stack([dt_optimal._rope_avg(A_raw[i], r, theta, wD) for i in range(H)])
    if not np.array_equal(A_avg[:, r:, r:], A_raw[:, r:, r:]):
        raise AssertionError(
            "dt_optimal._rope_avg changed the free-free block of the query Gram. This arm's whole "
            "claim to an EXACT content metric rests on it not doing that, so the derivation and the "
            "code have diverged -- read _rope_avg before going further.")
    A_free = A_raw[:, r:, r:]
    A_rot = A_avg[:, :r, :r]
    A_V = np.stack([Wo[i].T @ Wo[i] for i in range(H)])
    return A_free, A_V, A_rot

def _joint_stack(Kt, Vt, A_free, A_V, H, r, w):
    rows = []
    for i in range(H):
        rows.append(gc.psd_pow(A_free[i], 0.5) @ Kt[i][r:, :])
        rows.append(np.sqrt(w) * (gc.psd_pow(A_V[i], 0.5) @ Vt[i]))
    return np.concatenate(rows, axis=0)

def _rot_stack(Kt, A_rot, H, r):
    return np.concatenate([gc.psd_pow(A_rot[i], 0.5) @ Kt[i][:r, :] for i in range(H)], axis=0)

SIGMA_TOL = 1e-8
SIGMA_MAX_FLOORED = 0.05
SIGMA_MAX_DEAD = 0.25

def _sigma_pair(S, n, tol=SIGMA_TOL):
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (n, n):
        raise ValueError(f"Sigma for this layer is {S.shape}, expected the AUGMENTED ({n}, {n}). "
                         f"gc.read_layer's maps carry the folded bias column, so Sigma must be the "
                         f"second moment of [x, 1], not of x.")
    if not np.isfinite(S).all():
        raise ValueError("Sigma contains non-finite entries")
    if abs(S[-1, -1] - 1.0) > 1e-9:
        raise ValueError(
            f"Sigma[-1, -1] = {S[-1, -1]:.6g}, must be exactly 1.0. That entry is E[1*1] for the "
            f"augmented constant coordinate, so this Sigma was accumulated WITHOUT the constant "
            f"column -- it is the second moment of a different input space than the maps live in.")
    S = (S + S.T) / 2.0
    lam, V = np.linalg.eigh(S)
    lam = np.clip(lam, 0.0, None)
    thr = tol * lam.max()
    keep = lam > thr
    n_floored = int((~keep).sum())
    n_dead = int((np.diag(S)[:-1] <= thr).sum())
    if n_dead > SIGMA_MAX_DEAD * n:
        raise ValueError(
            f"Sigma has {n_dead}/{n} dead input coordinates (E[x_i^2] <= tol * lambda_max, so their directions "
            f"are floored), above the {SIGMA_MAX_DEAD:.0%} ceiling. A model does not switch off that much of its "
            f"own input: check that the hook read the right activation.")
    if n_floored - n_dead > SIGMA_MAX_FLOORED * n:
        raise ValueError(
            f"Sigma needs {n_floored}/{n} directions floored at tol={tol:g} ({n_dead} of them explained by "
            f"dead input coordinates), so {n_floored - n_dead} unexplained, above the "
            f"{SIGMA_MAX_FLOORED:.0%} ceiling. A second moment this degenerate is not a rank "
            f"deficiency to work around -- check that it was averaged over enough tokens "
            f"(n(n+1)/2 = {n * (n + 1) // 2} free parameters) and that the hook read the right "
            f"activation.")
    inv = np.zeros_like(lam)
    inv[keep] = lam[keep] ** -0.5
    return (V * np.sqrt(lam)) @ V.T, (V * inv) @ V.T, n_floored

def allocation_report(model, G, w=W_KV_DEFAULT, d_k_R=DKR_DEFAULT, layers_to_check=None):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    d_k_R, c_KV = _widths(G, H, d_k, d_v, r, d_k_R)
    theta, wD = gc.inv_freq(model), dt_optimal._offset_weights(dt_optimal.T_EVAL)
    idx = range(L) if layers_to_check is None else layers_to_check

    out = []
    for li in idx:
        Qt, Kt, Vt, Wo = gc.read_layer(layers[li], H, d_k, d_v, dm)
        A_free, A_V, _ = _metrics(Qt, Wo, H, r, theta, wD)
        Zk = np.concatenate([gc.psd_pow(A_free[i], 0.5) @ Kt[i][r:, :] for i in range(H)], axis=0)
        Zv = np.concatenate([np.sqrt(w) * (gc.psd_pow(A_V[i], 0.5) @ Vt[i]) for i in range(H)], 0)
        lk = np.linalg.svd(Zk, compute_uv=False) ** 2
        lv = np.linalg.svd(Zv, compute_uv=False) ** 2
        merged = sorted([(v, "K") for v in lk] + [(v, "V") for v in lv], reverse=True)
        t_star = sum(1 for _, s in merged[:c_KV] if s == "K")
        Xk, Xv = Zk.T @ Zk, Zv.T @ Zv
        denom = np.linalg.norm(Xk) * np.linalg.norm(Xv)
        overlap = float(np.linalg.norm(Xk @ Xv) / denom) if denom > 0 else 0.0
        out.append({"layer": li, "c_KV": c_KV, "d_k_R": d_k_R,
                    "t_star": int(t_star), "t_optwide": int(G * d_k - d_k_R),
                    "overlap_frac": overlap})
    return out

def build(model, G, w=W_KV_DEFAULT, d_k_R=DKR_DEFAULT, GK=None, GV=None, SIGMA=None,
          return_bases=False, B=None):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0:
        raise ValueError("decoupled_optwide has nothing to decouple on a model with no rotary rows")
    widths = _widths_by_layer(G, H, d_k, d_v, r, d_k_R, L, B=B)
    theta, wD = gc.inv_freq(model), dt_optimal._offset_weights(dt_optimal.T_EVAL)
    if SIGMA is not None and len(SIGMA) != L:
        raise ValueError(f"SIGMA has {len(SIGMA)} layers, the model has {L}")
    n_floored = 0
    Wbar_all, WKR_all = [], []

    for li, lay in enumerate(layers):
        d_k_R_l, c_KV_l = widths[li]
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        A_free, A_V, A_rot = _metrics(Qt, Wo, H, r, theta, wD,
                                      None if GK is None else GK[li],
                                      None if GV is None else GV[li])

        Z_c = _joint_stack(Kt, Vt, A_free, A_V, H, r, w)
        Z_r = _rot_stack(Kt, A_rot, H, r)
        if SIGMA is None:
            Wbar = gc.tail_svd(Z_c, c_KV_l)[1]
            W_KR = gc.tail_svd(Z_r, d_k_R_l)[1]
            P_c, P_r = Wbar.T @ Wbar, W_KR.T @ W_KR
            Wbar_serve, WKR_serve = Wbar, W_KR
        else:
            S_h, S_ih, nfl = _sigma_pair(SIGMA[li], dm + 1)
            n_floored += nfl
            Wbar = gc.tail_svd(Z_c @ S_h, c_KV_l)[1]
            W_KR = gc.tail_svd(Z_r @ S_h, d_k_R_l)[1]
            P_c = S_h @ Wbar.T @ Wbar @ S_ih
            P_r = S_h @ W_KR.T @ W_KR @ S_ih
            Wbar_serve = np.linalg.qr((Wbar @ S_ih).T)[0].T
            WKR_serve = np.linalg.qr((W_KR @ S_ih).T)[0].T
        Wbar_all.append(Wbar_serve)
        WKR_all.append(WKR_serve)

        Kh, Vh = Kt.copy(), Vt.copy()
        for i in range(H):
            Kh[i][:r, :] = Kt[i][:r, :] @ P_r
            Kh[i][r:, :] = Kt[i][r:, :] @ P_c
            Vh[i] = Vt[i] @ P_c
        gc.write_layer(lay, Qt.copy(), Kh, Vh, Wo.copy(), H, d_k, d_v, dm)

    uniform = len({x for x in widths}) == 1
    d_txt = (f"d_k^R={widths[0][0]}, c_KV={widths[0][1]}" if uniform else
             f"d_k^R PER LAYER {[x[0] for x in widths]}, c_KV {[x[1] for x in widths]}")
    s_txt = ("Sigma = I (orthogonal projection)" if SIGMA is None else
             f"Sigma MEASURED (oblique, Sigma-weighted projection; {n_floored} directions floored "
             f"over {L} layers)")
    rot_txt = ("rotary metric MEASURED, raw (no rope_avg: a Fisher has already integrated the "
               "corpus's offsets, and averaging again would average twice)" if GK is not None else
               f"rotary metric offset-averaged at T={dt_optimal.T_EVAL}")
    b_txt = (f"=({G}*{d_k + d_v}) at every layer, cache-equivalent to optwide at G={G}" if B is None else
             f" = the budget B at every layer, cache-equivalent to optwide at B={B}")
    prov = (f"decoupled_optwide: one group (all {H} heads), JOINT K/V latent + decoupled rotary; "
            f"{d_txt}, total cache {widths[0][0] + widths[0][1]}{b_txt}; w={w}; content metric EXACT (no rope_avg), "
            f"{rot_txt}; {s_txt}")
    if return_bases:
        return prov, (Wbar_all, WKR_all, widths)
    return prov

def _selftest():
    rng = np.random.default_rng(0)
    H, d_k, d_v, dm, r, G = 16, 64, 64, 1024, 16, 8
    d_k_R, c_KV = _widths(G, H, d_k, d_v, r, None)
    assert (d_k_R, c_KV) == (16, 1008), (d_k_R, c_KV)
    assert c_KV + d_k_R == G * (d_k + d_v) == 1024
    for g, want in ((1, 112), (4, 496), (8, 1008)):
        assert _widths(g, H, d_k, d_v, r, None)[1] == want
    print(f"  widths            OK   G=8 -> c_KV={c_KV} + d_k^R={d_k_R} = {c_KV + d_k_R}")

    theta = np.array([1.0 / (10000 ** (2 * p / r)) for p in range(r // 2)])
    wD = dt_optimal._offset_weights(64)
    A = rng.standard_normal((d_k, d_k)); A = A @ A.T
    Aavg = dt_optimal._rope_avg(A, r, theta, wD)
    assert np.array_equal(Aavg[r:, r:], A[r:, r:]), "rope_avg touched the free block"
    assert not np.allclose(Aavg[:r, :r], A[:r, :r]), "rope_avg did NOT touch the rotary block"
    print("  rope_avg scope    OK   free-free block bitwise unchanged; rotary block changed")

    Kt = rng.standard_normal((H, d_k, dm + 1))
    Vt = rng.standard_normal((H, d_v, dm + 1))
    Af = np.stack([np.eye(d_k - r) for _ in range(H)])
    Av = np.stack([np.eye(d_v) for _ in range(H)])
    for w in (1.0, 0.25, 4.0):
        Z = _joint_stack(Kt, Vt, Af, Av, H, r, w)
        Xi = Z.T @ Z
        Xk = sum(Kt[i][r:, :].T @ Kt[i][r:, :] for i in range(H))
        Xv = sum(Vt[i].T @ Vt[i] for i in range(H))
        assert np.allclose(Xi, Xk + w * Xv), f"joint stack != Xi^K + w Xi^V at w={w}"
    print("  joint stack       OK   Z^T Z == Xi^K_free + w Xi^V at w in {1, 0.25, 4}")

    assert _widths(8, H, d_k, d_v, r, H * r) == (H * r, 8 * (d_k + d_v) - H * r)
    for bad in (H * r + 1, 999):
        try:
            _widths(8, H, d_k, d_v, r, bad)
            raise AssertionError(f"d_k^R={bad} was accepted; the rank bound H*r={H * r} is not enforced")
        except ValueError as e:
            assert "rank Xi_rot" in str(e) and str(H * r) in str(e), str(e)
    assert _widths(8, H, d_k, d_v, r, r + 1)[0] == r + 1
    print(f"  d_k^R bound       OK   {H * r} accepted, {H * r + 1} refused, r+1={r + 1} now allowed")

    L = 24
    assert _widths_by_layer(8, H, d_k, d_v, r, None, L) == [(16, 1008)] * L
    assert _widths_by_layer(8, H, d_k, d_v, r, 96, L) == [(96, 928)] * L
    per = _widths_by_layer(8, H, d_k, d_v, r, list(range(10, 10 + L)), L)
    assert [p[0] for p in per] == list(range(10, 10 + L))
    assert len({p[0] + p[1] for p in per}) == 1, "the TOTAL must not vary between layers"
    for bad, why in ((list(range(L - 1)), "too short"), (list(range(L + 1)), "too long")):
        try:
            _widths_by_layer(8, H, d_k, d_v, r, bad, L)
            raise AssertionError(f"a {why} per-layer width list was accepted")
        except ValueError:
            pass
    print(f"  per-layer widths  OK   scalar unchanged, {L} varying widths share one total, "
          f"wrong-length refused")

    nn = 24
    for bad, why in (((np.eye(nn), nn + 1), "wrong size"),
                     ((np.eye(nn) * 2.0, nn), "constant coordinate != 1")):
        try:
            _sigma_pair(*bad)
            raise AssertionError(f"a Sigma with {why} was accepted")
        except ValueError:
            pass
    _Sbad = np.eye(nn) * 1e-14
    _Sbad[-1, -1] = 1.0
    try:
        _sigma_pair(_Sbad, nn)
        raise AssertionError("a Sigma needing every direction floored was accepted")
    except ValueError as e:
        assert "floored" in str(e), str(e)
    nd = 40
    _X = rng.standard_normal((4000, nd - 1))
    _X[:, :6] = 0.0
    _Sd = np.ones((nd, nd))
    _Sd[:nd - 1, :nd - 1] = _X.T @ _X / len(_X)
    _Sd[:nd - 1, -1] = _Sd[-1, :nd - 1] = _X.mean(0)
    assert _sigma_pair(_Sd, nd)[2] == 6, "the dead coordinates were not floored"
    _Xs = rng.standard_normal((10, nd - 1))
    _Ss = np.ones((nd, nd))
    _Ss[:nd - 1, :nd - 1] = _Xs.T @ _Xs / len(_Xs)
    _Ss[:nd - 1, -1] = _Ss[-1, :nd - 1] = _Xs.mean(0)
    try:
        _sigma_pair(_Ss, nd)
        raise AssertionError("a Sigma averaged over too few tokens was accepted")
    except ValueError as e:
        assert "unexplained" in str(e), str(e)
    print(f"  Sigma guards      OK   wrong size, Sigma[-1,-1] != 1, and over-floored all refused; 6/39 dead "
          f"inputs excused, a 10-token average refused")

    _S1 = np.eye(nn)
    _h, _ih, _nf = _sigma_pair(_S1, nn)
    assert _nf == 0 and np.allclose(_h, _S1, atol=1e-13) and np.allclose(_ih, _S1, atol=1e-13)
    _Zt = rng.standard_normal((40, nn))
    _U1 = gc.tail_svd(_Zt @ _h, 7)[1]
    assert np.allclose(_h @ _U1.T @ _U1 @ _ih, gc.tail_svd(_Zt, 7)[1].T @ gc.tail_svd(_Zt, 7)[1],
                       atol=1e-12), "Sigma = I does not reproduce the orthogonal projector"
    print("  Sigma = I         OK   reproduces the orthogonal projector to 1e-12")

    _M = rng.standard_normal((nn - 1, nn - 1))
    _S = np.eye(nn)
    _S[:nn - 1, :nn - 1] = _M @ _M.T + 0.5 * np.eye(nn - 1)
    _S[:nn - 1, -1] = _S[-1, :nn - 1] = rng.standard_normal(nn - 1) * 0.1
    _S = (_S + _S.T) / 2.0
    _S += (abs(min(np.linalg.eigvalsh(_S).min(), 0.0)) + 1e-6) * np.eye(nn)
    _S /= _S[-1, -1]
    _h, _ih, _ = _sigma_pair(_S, nn)
    _c = 7
    _Uw = gc.tail_svd(_Zt @ _h, _c)[1]
    _B = _Uw @ _ih
    assert np.linalg.norm(_B @ _S @ _B.T - np.eye(_c)) < 1e-9, "B Sigma B^T != I"
    _P = _h @ _Uw.T @ _Uw @ _ih
    assert np.linalg.norm(_P @ _P - _P) < 1e-9, "the Sigma-weighted projector is not idempotent"
    assert np.linalg.norm(_P - _P.T) > 1e-3, "P is symmetric -- then Sigma was effectively I"
    print(f"  whitened solve    OK   B Sigma B^T = I to {np.linalg.norm(_B @ _S @ _B.T - np.eye(_c)):.1e}"
          f"; P idempotent and OBLIQUE (||P - P^T|| = {np.linalg.norm(_P - _P.T):.3f})")

    def _damage(Bmat):
        _R = _Zt @ _S @ Bmat.T @ np.linalg.inv(Bmat @ _S @ Bmat.T)
        _E = _Zt - _R @ Bmat
        return float(np.trace(_E.T @ _E @ _S))
    _Bo = gc.tail_svd(_Zt, _c)[1]
    _dw, _do = _damage(_B), _damage(_Bo)
    assert _dw < _do, f"the whitened solve took MORE damage ({_dw:.6e} vs {_do:.6e})"
    _Pt = _h @ _Bo.T @ np.linalg.inv(_Bo @ _S @ _Bo.T) @ _Bo @ _h
    assert abs(np.linalg.norm(_Zt @ _h @ (np.eye(nn) - _Pt), "fro") ** 2 - _do) < 1e-8 * _do
    print(f"  damage            OK   whitened {_dw:.6e} < orthogonal {_do:.6e} "
          f"({100 * (_do - _dw) / _do:.1f}% less), and == ||Z Sigma^1/2 (I - Ptilde)||_F^2")

    print("dt_decoupled selftest: ALL PASS")

if __name__ == "__main__":
    _selftest()
