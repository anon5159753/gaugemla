import numpy as np

import dt_decoupled
import dt_optimal
import gam_core as gc
import kvlab

READOUTS = ("drop", "verbatim", "damped")
COST_ATOL = 1e-12

def phase_mean(theta, T):
    d, w = gc.offset_weights(T)
    th = np.asarray(theta, dtype=np.float64)
    return (w[None, :] * np.exp(-1j * th[:, None] * d[None, :])).sum(axis=1)

def readout_scalars(theta, T, readout):
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    P = len(theta)
    if readout == "drop":
        return np.zeros(P, dtype=np.complex128)
    if readout == "verbatim":
        return np.ones(P, dtype=np.complex128)
    return phase_mean(theta, T)

def plane_cost(theta, T, readout):
    A = phase_mean(theta, T)
    mu = readout_scalars(theta, T, readout)
    c = 1.0 - 2.0 * np.real(np.conj(mu) * A) + np.abs(mu) ** 2
    if not np.isfinite(c).all():
        raise ValueError("plane_cost: non-finite cost")
    if (c < -COST_ATOL).any():
        raise ValueError(f"plane_cost: negative cost (min {c.min():.3e}), so |e^(i th d) - mu|^2 "
                         f"averaged negative -- phase_mean or offset_weights is wrong")
    return np.maximum(c, 0.0)

def derot_coupling(theta, T, readout):
    d, wD = gc.offset_weights(T)
    th = np.asarray(theta, dtype=np.float64)
    mu = readout_scalars(th, T, readout)
    z = np.exp(1j * th[:, None] * d[None, :]) * mu[:, None] - 1.0
    Om = np.empty((len(th), len(d), 2, 2))
    Om[:, :, 0, 0] = Om[:, :, 1, 1] = z.real
    Om[:, :, 0, 1] = -z.imag
    Om[:, :, 1, 0] = z.imag
    return np.einsum("t,ptab,qtcd->pqabcd", wD, Om, Om, optimize=True)

def derot_damage(F, C, K, P):
    H = F.shape[0]
    Fv = F.reshape(H, 2, P, 2, P)
    Cv = C.reshape(H, 2, P, 2, P)
    return float(np.einsum("hapcq,pqabcd,hbpdq->", Fv, K, Cv, optimize=True))

def layer_curves(Kt, Vt, Qt, Wo, GK_l, GV_l, SIGMA_l, H, r, dm, theta, wD, readout, T, m_max=None,
                 m_eval=None, rotary_alloc="energy"):
    P = r // 2
    Kt = np.asarray(Kt, dtype=np.float64)
    if GK_l is None or GV_l is None:
        raise ValueError("the width rule scores under the MEASURED metric: GK and GV are required")
    Gk = np.asarray(GK_l, dtype=np.float64)
    A_free = np.ascontiguousarray(Gk[:, r:, r:])
    A_V = np.ascontiguousarray(np.asarray(GV_l, dtype=np.float64))
    S_h = dt_decoupled._sigma_pair(SIGMA_l, dm + 1)[0]
    n = S_h.shape[0]

    fixed = dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H, r, 1.0) @ S_h
    G_fixed = fixed.T @ fixed

    G_l = np.asarray(GK_l, dtype=np.float64)
    F = np.ascontiguousarray(G_l[:, :r, :r])
    A_rot_h = np.stack([gc.psd_pow(F[i], 0.5) for i in range(H)])

    dg = np.diagonal(G_l, axis1=1, axis2=2)
    cz = np.stack([gc.to_complex(dg[i][:, None], r)[:, 0] for i in range(H)])
    varpi = 0.5 * (cz.real + cz.imag)
    if (varpi < 0).any():
        raise ValueError("a negative Fisher diagonal -- GK is not a Fisher")

    Ks = np.stack([gc.to_complex(Kt[i], r) for i in range(H)]) @ S_h
    Vh, mu_p = [], []
    for p in range(P):
        _, sp, vh = np.linalg.svd(np.sqrt(varpi[:, p])[:, None] * Ks[:, p, :], full_matrices=False)
        Vh.append(vh)
        mu_p.append(sp ** 2)
    if rotary_alloc == "energy":
        order = np.argsort(-np.concatenate(mu_p), kind="stable")
    elif rotary_alloc == "cost":
        c_alloc = plane_cost(theta, T, readout)
        if not c_alloc.any():
            raise ValueError("every plane un-ropes for free (c = 0): the cost order is the tie-break alone")
        order = np.argsort(-np.concatenate([c_alloc[p] * mu_p[p] for p in range(P)]), kind="stable")
    else:
        raise ValueError(f"rotary_alloc {rotary_alloc!r} not in ('energy', 'cost')")

    K_cpl = derot_coupling(theta, T, readout)
    moving = readout != "drop"
    mu_c = readout_scalars(theta, T, readout)[None, :, None] if readout == "damped" else None

    Rc = Ks.copy()
    nxt = [0] * P
    n_m = len(order) + 1 if m_max is None else min(len(order), int(m_max)) + 1
    D = np.empty(n_m)
    lam = np.empty((n_m if moving else 1, n))
    for m in range(n_m):
        if m:
            p = int(order[m - 1]) // H
            v = Vh[p][nxt[p]]
            nxt[p] += 1
            Rc[:, p, :] -= np.outer(Rc[:, p, :] @ v.conj(), v)
        R = np.concatenate([Rc.real, Rc.imag], axis=1)
        C = R @ R.transpose(0, 2, 1)
        D[m] = derot_damage(F, C, K_cpl, P)
        if moving and m_eval is not None and m not in m_eval:
            lam[m] = np.nan
        elif moving:
            if mu_c is not None:
                Rm = mu_c * Rc
                R = np.concatenate([Rm.real, Rm.imag], axis=1)
            Ar = A_rot_h @ R
            Gm = G_fixed + np.einsum("hrn,hrm->nm", Ar, Ar, optimize=True)
            lam[m] = np.linalg.eigvalsh(Gm)[::-1]
        elif m == 0:
            lam[0] = np.linalg.eigvalsh(G_fixed)[::-1]
    np.maximum(lam, 0.0, out=lam)
    return D, lam

def widths(per, group_sizes, L, H, d_k, d_v, r):
    cap = H * (r // 2)
    out = {}
    for G in group_sizes:
        B = G * (d_k + d_v)
        m_by = []
        for D, lam in per:
            ms = np.arange(0, min(cap, B // 2) + 1)
            if B > lam.shape[1]:
                raise ValueError(
                    f"B = {B} exceeds the {lam.shape[1]} directions the joint stack provides, so "
                    f"the content term is an empty sum = 0.0 for every "
                    f"m < {(B - lam.shape[1] + 1) // 2} and the argmin over m is meaningless.")
            if ms[-1] >= len(D):
                raise ValueError(f"G = {G} needs the curves to m = {ms[-1]}; they were computed only "
                                 f"to m = {len(D) - 1} (layer_curves' m_max)")
            tail = np.array([lam[m if lam.shape[0] > 1 else 0, B - 2 * m:].sum() for m in ms])
            tot = D[ms] + tail
            m_by.append(int(ms[int(np.flatnonzero(tot == tot.min())[-1])]))
        srt = sorted(m_by)
        m_med = (srt[L // 2 - 1] + srt[L // 2] + 1) // 2
        blocked = m_med == 0 or 2 * m_med >= B
        out[G] = {"B": B, "m_by_layer": m_by, "m": m_med, "blocked": blocked,
                  "w": None if blocked else 2 * m_med,
                  "d_c": None if blocked else B - 2 * m_med}
    return out

def compute(model, group_sizes, GK, GV, SIGMA, *, readout, T=None, log=None, m_max=None):
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0 or r % 2:
        raise ValueError(f"the MLA width needs an even rotary width; this model has r = {r}")
    if L % 2:
        raise ValueError(f"the half-up median rule assumes an even layer count; L = {L}")
    T = dt_optimal.T_EVAL if T is None else T
    theta, wD = gc.inv_freq(model), dt_optimal._offset_weights(T)
    per = []
    for li, lay in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        per.append(layer_curves(Kt, Vt, Qt, Wo, GK[li], GV[li], SIGMA[li],
                                H, r, dm, theta, wD, readout, T, m_max=m_max))
        if log:
            log("    layer %2d done (%s)" % (li, readout))
    return widths(per, group_sizes, L, H, d_k, d_v, r), per

def _argmin_m(D, rows, B, ms):
    tot = np.array([D[m] + rows[m][B - 2 * m:].sum() for m in ms])
    return int(ms[int(np.flatnonzero(tot == tot.min())[-1])])

def layer_m_star(get_curves, budgets, cap, n, coarse=None):
    for B in budgets:
        if B > n:
            raise ValueError(f"B = {B} exceeds the {n} directions the joint stack provides; the content term "
                             f"would be an empty sum and the argmin meaningless")
    m_hi = min(cap, max(budgets) // 2)
    if coarse is None:
        D, lam = get_curves(None)
        rows = {m: lam[m if lam.shape[0] > 1 else 0] for m in range(m_hi + 1)}
    else:
        s = int(coarse)
        if s < 2:
            raise ValueError(f"coarse = {coarse}: use None for the full search")
        grid = sorted(set(range(0, m_hi + 1, s)) | {min(cap, B // 2) for B in budgets})
        D, lam = get_curves(set(grid))
        rows = {m: lam[m if lam.shape[0] > 1 else 0] for m in grid}
        fine = set()
        for B in budgets:
            hi = min(cap, B // 2)
            m0 = _argmin_m(D, rows, B, [m for m in grid if m <= hi])
            fine |= set(range(max(0, m0 - s + 1), min(hi, m0 + s - 1) + 1))
        fine -= set(grid)
        if fine:
            _D2, lam2 = get_curves(fine)
            rows.update({m: lam2[m if lam2.shape[0] > 1 else 0] for m in fine})
    out = {}
    for B in budgets:
        hi = min(cap, B // 2)
        if hi >= len(D):
            raise ValueError(f"B = {B} needs the curves to m = {hi}; they reach m = {len(D) - 1}")
        out[B] = _argmin_m(D, rows, B, sorted(m for m in rows if m <= hi))
    return out

def rows_by_budget(m_by_layer_B, budgets, L):
    if L % 2:
        raise ValueError(f"the half-up median rule assumes an even layer count; L = {L}")
    out = {}
    for B in budgets:
        m_by = [int(x[B]) for x in m_by_layer_B]
        srt = sorted(m_by)
        lo, hi = srt[L // 2 - 1], srt[L // 2]
        m_med = (lo + hi + 1) // 2
        blocked = m_med == 0 or 2 * m_med >= B
        out[B] = {"B": int(B), "m_by_layer": m_by, "m": int(m_med), "blocked": bool(blocked),
                  "w": None if blocked else 2 * m_med, "d_c": None if blocked else B - 2 * m_med,
                  "m_median_raw": (lo + hi) / 2, "half_tie": (lo + hi) % 2 == 1,
                  "m_half_even": int(round((lo + hi) / 2))}
    return out

def _layer_job(a):
    Kt, Vt, Qt, Wo, GK_l, GV_l, S_l, H, r, dm, theta, wD, readout, T, m_hi, alloc, budgets, cap, coarse = a

    def get_curves(m_eval):
        return layer_curves(Kt, Vt, Qt, Wo, GK_l, GV_l, S_l, H, r, dm, theta, wD, readout, T,
                            m_max=m_hi, m_eval=m_eval, rotary_alloc=alloc)
    return layer_m_star(get_curves, budgets, cap, dm + 1, coarse=coarse)

def compute_budgets(model, budgets, GK, GV, SIGMA, *, readout, rotary_alloc, T=None, coarse=None, log=None,
                    workers=1):
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0 or r % 2:
        raise ValueError(f"the MLA width needs an even rotary width; this model has r = {r}")
    budgets = sorted(int(b) for b in budgets)
    T = dt_optimal.T_EVAL if T is None else T
    theta, wD = gc.inv_freq(model), dt_optimal._offset_weights(T)
    cap = H * (r // 2)
    m_hi = min(cap, max(budgets) // 2)
    layers = kvlab.get_layers(model)

    def job(li):
        Qt, Kt, Vt, Wo = gc.read_layer(layers[li], H, d_k, d_v, dm)
        return (Kt, Vt, Qt, Wo, GK[li], GV[li], SIGMA[li], H, r, dm, theta, wD, readout, T, m_hi,
                rotary_alloc, budgets, cap, coarse)

    def _log(li):
        if log:
            log("    layer %2d done (%s, %s, coarse %s)" % (li, readout, rotary_alloc, coarse))

    per = [None] * L
    if workers and workers > 1:
        import concurrent.futures as _cf
        with _cf.ProcessPoolExecutor(max_workers=int(workers)) as ex:
            live, nxt = {}, 0
            while nxt < L or live:
                while nxt < L and len(live) < int(workers):
                    live[ex.submit(_layer_job, job(nxt))] = nxt
                    nxt += 1
                done, _ = _cf.wait(list(live), return_when=_cf.FIRST_COMPLETED)
                for f in done:
                    li = live.pop(f)
                    per[li] = f.result()
                    _log(li)
    else:
        for li in range(L):
            per[li] = _layer_job(job(li))
            _log(li)
    return rows_by_budget(per, budgets, L), per

def _selftest(T=512):
    rng = np.random.default_rng(7)
    P, H, r = 6, 3, 12
    theta = 1.0 / (10000.0 ** (np.arange(0, 2 * P, 2) / (2 * P)))
    ok = []

    K0 = derot_coupling(theta, T, "drop")
    X = rng.standard_normal((H, r, 9))
    C = X @ X.transpose(0, 2, 1)
    Y = rng.standard_normal((H, r, r))
    F = Y @ Y.transpose(0, 2, 1)
    got = derot_damage(F, C, K0, P)
    want = float(np.einsum("hjk,hjk->", F, C))
    ok.append(("drop == tr(F C)", abs(got - want) <= 1e-9 * abs(want), "%.12e vs %.12e" % (got, want)))

    d, wD = gc.offset_weights(T)
    for name, pred in (("verbatim", lambda a: 2.0 * (1.0 - a.real)),
                       ("damped", lambda a: 1.0 - abs(a) ** 2)):
        Kc = derot_coupling(theta, T, name)
        A = phase_mean(theta, T)
        bad = 0.0
        for p in range(P):
            Cp = np.zeros((1, r, r))
            Cp[0, p, p] = Cp[0, p + P, p + P] = 1.0
            got = derot_damage(np.eye(r)[None], Cp, Kc, P)
            bad = max(bad, abs(got - 2.0 * pred(A[p])))
        ok.append(("%s == %s" % (name, "2(1-Re A)" if name == "verbatim" else "1-|A|^2"),
                   bad <= 1e-9, "worst |got - predicted| = %.3e" % bad))

    Kv, Kd = derot_coupling(theta, T, "verbatim"), derot_coupling(theta, T, "damped")
    dr = np.array([derot_damage(F, C, K, P) for K in (K0, Kv, Kd)])
    ok.append(("damped <= drop", dr[2] <= dr[0] * (1 + 1e-12), "damped %.6e vs drop %.6e" % (dr[2], dr[0])))
    ok.append(("verbatim > drop on a fast-plane mix", dr[1] > dr[0],
               "verbatim %.6e vs drop %.6e  (ratio %.3f)" % (dr[1], dr[0], dr[1] / dr[0])))

    for name, good, detail in ok:
        print("  [%s] %-38s %s" % ("PASS" if good else "FAIL", name, detail))
    return all(g for _, g, _ in ok)

if __name__ == "__main__":
    import sys

    print("dt_mla_unrope_width self-test")
    sys.exit(0 if _selftest() else 1)
