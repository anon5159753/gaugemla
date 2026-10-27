import numpy as np

import gam_core as gc

def _tr(X, A, Y):
    return float(np.sum(A * (X @ Y.T)))

def damage_key(Qh, Kbar, Qc, Kc, Abar_fn):
    per = []
    for i in range(Qh.shape[0]):
        G1 = Qh[i] @ Qh[i].T
        G2 = Qh[i] @ Qc[i].T
        G3 = Qc[i] @ Qc[i].T
        per.append(_tr(Kbar, Abar_fn(G1), Kbar)
                   - 2.0 * _tr(Kbar, Abar_fn(G2), Kc[i])
                   + _tr(Kc[i], Abar_fn(G3), Kc[i]))
    return float(sum(per)), np.array(per)

def damage_key_via_E(Om_inv, Kbar, Qc, Kc, Abar_fn):
    per = []
    for i in range(Qc.shape[0]):
        E = Om_inv[i] @ Kbar - Kc[i]
        per.append(_tr(E, Abar_fn(Qc[i] @ Qc[i].T), E))
    return float(sum(per)), np.array(per)

def damage_val(Oh, Vbar, Wo, Vc):
    per = []
    for i in range(Wo.shape[0]):
        per.append(_tr(Vbar, Oh[i].T @ Oh[i], Vbar)
                   - 2.0 * _tr(Vbar, Oh[i].T @ Wo[i], Vc[i])
                   + _tr(Vc[i], Wo[i].T @ Wo[i], Vc[i]))
    return float(sum(per)), np.array(per)

def make_abar_fn(theta, T, d_k, r, sign=1.0):
    if T is None:
        return lambda G: G
    th = sign * np.asarray(theta, dtype=np.float64)
    return lambda G: gc.abar_emp(G, th, T, d_k, r)

def floor_key(Qc, Kc, d_k, Abar_fn):
    blocks = [gc.psd_pow(Abar_fn(Qc[i] @ Qc[i].T), 0.5) @ Kc[i] for i in range(Qc.shape[0])]
    return gc.tail_svd(np.concatenate(blocks, axis=0), d_k)

def floor_val(Wo, Vc, d_v):
    blocks = [gc.psd_pow(Wo[i].T @ Wo[i], 0.5) @ Vc[i] for i in range(Wo.shape[0])]
    return gc.tail_svd(np.concatenate(blocks, axis=0), d_v)

def arm2(Qc, Kc, Vc, Wo):
    return dict(name="arm2", Kbar=Kc.mean(axis=0), Vbar=Vc.mean(axis=0),
                Qh=Qc.copy(), Oh=Wo.copy(), Om=None, Gm=None, flags={})

def arm2b(Qc, Kc, Vc, Wo, theta=None, T=None, d_k=None, r=None):
    Kbar, Vbar = Kc.mean(axis=0), Vc.mean(axis=0)
    H = Qc.shape[0]
    if T is None:
        P = Kbar @ Kbar.T
        Cs = [Kbar @ Kc[i].T for i in range(H)]
    else:
        th = -np.asarray(theta, dtype=np.float64)
        P = gc.abar_emp(Kbar @ Kbar.T, th, T, d_k, r)
        Cs = [gc.abar_emp(Kbar @ Kc[i].T, th, T, d_k, r) for i in range(H)]
    Pi = np.linalg.inv(P)
    Qh = np.stack([Pi @ Cs[i] @ Qc[i] for i in range(H)])
    Vp = np.linalg.pinv(Vbar)
    Oh = np.stack([Wo[i] @ (Vc[i] @ Vp) for i in range(H)])
    return dict(name="arm2b", Kbar=Kbar, Vbar=Vbar, Qh=Qh, Oh=Oh, Om=None, Gm=None,
                flags={"cond_P": float(np.linalg.cond(P))})

def arm3A(Qc, Kc, Vc, Wo, d_k, d_v, r):
    H, h = Qc.shape[0], r // 2
    TK, TV = Kc.mean(axis=0), Vc.mean(axis=0)
    tK = gc.to_complex(TK, r)
    Oms, Ois, Gms, Gis = [], [], [], []
    plane_min, det_free, det_val = [], [], []
    for i in range(H):
        w = gc.to_complex(Kc[i], r)
        c = np.array([(tK[j] * np.conj(w[j])).sum() / max((np.abs(w[j]) ** 2).sum(), 1e-300)
                      for j in range(h)])
        plane_min.append(float(np.abs(c).min()) if h else np.inf)
        if d_k > r:
            Kf = Kc[i][r:]
            D = TK[r:] @ np.linalg.pinv(Kf)
            det_free.append(float(np.linalg.slogdet(D)[0] * np.exp(min(np.linalg.slogdet(D)[1], 700))))
        else:
            D = None
        Om, Oi = gc.omega_commutant(c, D, d_k, r)
        Oms.append(Om)
        Ois.append(Oi)
        G = TV @ np.linalg.pinv(Vc[i])
        det_val.append(float(np.linalg.slogdet(G)[1]))
        Gms.append(G)
        Gis.append(np.linalg.inv(G))
    Om, Oi = np.stack(Oms), np.stack(Ois)
    Gm, Gi = np.stack(Gms), np.stack(Gis)
    Kbar = np.mean([Om[i] @ Kc[i] for i in range(H)], axis=0)
    Vbar = np.mean([Gm[i] @ Vc[i] for i in range(H)], axis=0)
    Qh = np.stack([Oi[i].T @ Qc[i] for i in range(H)])
    Oh = np.stack([Wo[i] @ Gi[i] for i in range(H)])
    return dict(name="arm3A", Kbar=Kbar, Vbar=Vbar, Qh=Qh, Oh=Oh, Om=Om, Oi=Oi, Gm=Gm, Gi=Gi,
                flags={"plane_min_abs_c": float(np.min(plane_min)),
                       "free_det": det_free, "val_logdet": det_val,
                       "cond_Om": float(max(np.linalg.norm(o, 2) for o in Oi)),
                       "cond_Gm": float(max(np.linalg.norm(g, 2) for g in Gi))})

def arm3B(Qc, Kc, Vc, Wo, d_k, d_v, r):
    H, h = Qc.shape[0], r // 2
    Wcx = np.stack([gc.to_complex(Kc[i], r) for i in range(H)])
    Qcx = np.stack([gc.to_complex(Qc[i], r) for i in range(H)])
    varpi = np.zeros((H, h))
    for j in range(h):
        varpi[:, j] = 0.5 * (np.abs(Qcx[:, j, :]) ** 2).sum(axis=1)

    Kbar, Vbar = Kc.mean(axis=0), Vc.mean(axis=0)
    tau = gc.to_complex(Kbar, r)
    tau_sq = (np.abs(tau) ** 2).sum(axis=1)
    Kbar_nr = Kbar[r:] if d_k > r else None
    if d_k > r:
        Gnr_inv = np.linalg.inv(Kbar_nr @ Kbar_nr.T)

    Oms, Ois, det_free, cosines = [], [], [], []
    rot_resid = np.zeros(h)
    free_resid = 0.0
    for i in range(H):
        ip = np.array([(Wcx[i, j] * np.conj(tau[j])).sum() for j in range(h)])
        knorm = np.sqrt((np.abs(Wcx[i]) ** 2).sum(axis=1))
        cosines.extend((np.abs(ip) / np.maximum(knorm * np.sqrt(tau_sq), 1e-300)).tolist())
        psi = ip / tau_sq
        for j in range(h):
            e = psi[j] * tau[j] - Wcx[i, j]
            rot_resid[j] += varpi[i, j] * float((np.abs(e) ** 2).sum())
        if d_k > r:
            dstar = Kc[i][r:] @ Kbar_nr.T @ Gnr_inv
            det_free.append(float(np.linalg.slogdet(dstar)[1]))
            D = np.linalg.inv(dstar)
            Anr_h = gc.psd_pow(Qc[i][r:] @ Qc[i][r:].T, 0.5)
            free_resid += float(((Anr_h @ (dstar @ Kbar_nr - Kc[i][r:])) ** 2).sum())
        else:
            D = None
        Om, Oi = gc.omega_commutant(1.0 / psi, D, d_k, r)
        Oms.append(Om)
        Ois.append(Oi)
    Om, Oi = np.stack(Oms), np.stack(Ois)

    Gms, Gis, det_val, val_resid = [], [], [], 0.0
    Gv_inv = np.linalg.inv(Vbar @ Vbar.T)
    for i in range(H):
        gstar = Vc[i] @ Vbar.T @ Gv_inv
        det_val.append(float(np.linalg.slogdet(gstar)[1]))
        Gis.append(gstar)
        Gms.append(np.linalg.inv(gstar))
        Wh = gc.psd_pow(Wo[i].T @ Wo[i], 0.5)
        val_resid += float(((Wh @ (gstar @ Vbar - Vc[i])) ** 2).sum())
    Gm, Gi = np.stack(Gms), np.stack(Gis)

    roundtrip = float(max(np.abs(Om[i] @ Oi[i] - np.eye(d_k)).max() for i in range(H)))
    if not np.isfinite(roundtrip) or roundtrip > 1e-8:
        raise ValueError(
            "Gauge 3B: the returned inverse does not round-trip (max|Om Oi - I| = %.3e). "
            "Oi is not Om^-1 at working precision, so the solve has FAILED -- most likely a plane "
            "near-orthogonal to the pooled direction (min |cos vartheta| = %.3e)."
            % (roundtrip, min(cosines) if cosines else float("nan")))

    Qh = np.stack([Oi[i].T @ Qc[i] for i in range(H)])
    Oh = np.stack([Wo[i] @ Gi[i] for i in range(H)])
    return dict(name="arm3B", Kbar=Kbar, Vbar=Vbar, Qh=Qh, Oh=Oh, Om=Om, Oi=Oi, Gm=Gm, Gi=Gi,
                flags={"plane_min_cos": float(min(cosines)) if cosines else float("nan"),
                       "gauge_roundtrip": roundtrip,
                       "free_logdet": det_free, "val_logdet": det_val,
                       "cond_Om": float(max(np.linalg.norm(o, 2) for o in Oi)),
                       "cond_Gm": float(max(np.linalg.norm(g, 2) for g in Gi))},
                rot_resid_total=float(rot_resid.sum()), rot_resid=rot_resid,
                free_resid=float(free_resid), val_resid=float(val_resid))

def _arm3B_prev_optimal_target(Qc, Kc, Vc, Wo, d_k, d_v, r):
    H, h = Qc.shape[0], r // 2
    ncol = Kc.shape[2]
    lines = np.zeros((h, ncol), dtype=np.complex128)
    varpi = np.zeros((H, h))
    Wcx = np.stack([gc.to_complex(Kc[i], r) for i in range(H)])
    Qcx = np.stack([gc.to_complex(Qc[i], r) for i in range(H)])
    rot_tail = np.zeros(h)
    for j in range(h):
        varpi[:, j] = 0.5 * (np.abs(Qcx[:, j, :]) ** 2).sum(axis=1)
        S = np.sqrt(varpi[:, j])[:, None] * Wcx[:, j, :]
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        lines[j] = Vt[0]
        rot_tail[j] = float((s[1:] ** 2).sum())
    if d_k > r:
        blocks = [gc.psd_pow(Qc[i][r:] @ Qc[i][r:].T, 0.5) @ Kc[i][r:] for i in range(H)]
        free_tail, Kfree = gc.tail_svd(np.concatenate(blocks, axis=0), d_k - r)
    else:
        free_tail, Kfree = 0.0, None
    Kbar = gc.from_complex(lines, d_k, r, free=Kfree)
    return dict(name="arm3B_prev_optimal_target", Kbar=Kbar,
                psi_rot=float(rot_tail.sum()), rot_tail=rot_tail, free_tail=float(free_tail))

def arm_gl(Qc, Kc, d_k):
    H = Qc.shape[0]
    tail, Kbar = floor_key(Qc, Kc, d_k, lambda G: G)
    Oms, Ois, logdet = [], [], []
    for i in range(H):
        Mi = Kc[i] @ Kbar.T
        logdet.append(float(np.linalg.slogdet(Mi)[1]))
        Ois.append(Mi)
        Oms.append(np.linalg.inv(Mi))
    Om, Oi = np.stack(Oms), np.stack(Ois)
    Qh = np.stack([Oi[i].T @ Qc[i] for i in range(H)])
    return dict(name="gl", Kbar=Kbar, Qh=Qh, Om=Om, Oi=Oi, tail=float(tail),
                flags={"attain_logdet": logdet,
                       "attain_min_logdet": float(np.min(logdet)),
                       "cond_Om": float(max(np.linalg.norm(o, 2) for o in Oi))})

def solve_grouped(name, Qc, Kc, Vc, Wo, groups, d_k, d_v, r, theta=None, T=None):
    H = Qc.shape[0]
    ncol = Kc.shape[2]
    Kbar = np.zeros((len(groups), d_k, ncol))
    Vbar = np.zeros((len(groups), d_v, ncol))
    Qh = np.zeros_like(Qc)
    Oh = np.zeros_like(Wo)
    Om = Oi = Gm = Gi = None
    flags, extra = {}, {}
    for j, g in enumerate(groups):
        g = list(g)
        q, k, v, o = Qc[g], Kc[g], Vc[g], Wo[g]
        if name == "arm2":
            a = arm2(q, k, v, o)
        elif name == "arm2b":
            a = arm2b(q, k, v, o, theta, T, d_k, r)
        elif name == "arm3A":
            a = arm3A(q, k, v, o, d_k, d_v, r)
        elif name == "arm3B":
            a = arm3B(q, k, v, o, d_k, d_v, r)
        elif name == "gl":
            a = arm_gl(q, k, d_k)
            a["Vbar"] = v.mean(axis=0)
            a["Oh"] = o.copy()
        else:
            raise ValueError(name)
        Kbar[j], Vbar[j] = a["Kbar"], a["Vbar"]
        Qh[g], Oh[g] = a["Qh"], a["Oh"]
        if a.get("Om") is not None:
            if Om is None:
                Om, Oi = np.zeros((H, d_k, d_k)), np.zeros((H, d_k, d_k))
            Om[g], Oi[g] = a["Om"], a["Oi"]
        if a.get("Gm") is not None:
            if Gm is None:
                Gm, Gi = np.zeros((H, d_v, d_v)), np.zeros((H, d_v, d_v))
            Gm[g], Gi[g] = a["Gm"], a["Gi"]
        for k_, v_ in a.get("flags", {}).items():
            flags.setdefault(k_, []).append(v_)
        if name == "arm3B":
            for k_ in ("rot_resid_total", "free_resid", "val_resid"):
                extra[k_] = extra.get(k_, 0.0) + float(a[k_])
    return dict(name=name, Kbar=Kbar, Vbar=Vbar, Qh=Qh, Oh=Oh,
                Om=Om, Oi=Oi, Gm=Gm, Gi=Gi, flags=flags, groups=groups, **extra)

def selftest_a11(P=720, nplanes=8, nfree=3, ncol=11, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(1, nplanes + 1)
    assert len(set(n)) == nplanes and (n > 0).all()
    for j in range(nplanes):
        assert n[j] % P and (2 * n[j]) % P
        for k in range(nplanes):
            if j != k:
                assert (n[j] - n[k]) % P and (n[j] + n[k]) % P
    theta = 2 * np.pi * n / P
    r, d_k = 2 * nplanes, 2 * nplanes + nfree
    Q = rng.normal(size=(d_k, ncol))
    E = rng.normal(size=(d_k, ncol))

    lhs = 0.0
    for delta in range(P):
        R = gc.rot(theta, -delta, d_k, r)
        lhs += float(((Q.T @ R @ E) ** 2).sum())
    lhs /= P

    qc = gc.to_complex(Q, r)
    ec = gc.to_complex(E, r)
    psi = float(sum((np.abs(qc[j]) ** 2).sum() * (np.abs(ec[j]) ** 2).sum()
                    for j in range(nplanes)))
    free = float(np.trace(E[r:].T @ (Q[r:] @ Q[r:].T) @ E[r:])) if nfree else 0.0
    rhs = 0.5 * psi + free
    return abs(lhs - rhs) / max(abs(lhs), 1e-300), lhs, rhs

def selftest_circuits(Qc, Kc, d_k, r, theta, T, seed=0):
    rng = np.random.default_rng(seed)
    H, h = Qc.shape[0], r // 2
    ab = make_abar_fn(theta, T, d_k, r)

    c = np.exp(rng.normal(size=(H, h)) * 0.2 + 1j * rng.normal(size=(H, h)))
    Dl = [np.linalg.qr(rng.normal(size=(d_k - r, d_k - r)))[0] for _ in range(H)]
    Om = np.stack([gc.omega_commutant(c[i], Dl[i], d_k, r)[0] for i in range(H)])
    Oi = np.stack([gc.omega_commutant(c[i], Dl[i], d_k, r)[1] for i in range(H)])
    Kbar = np.mean([Om[i] @ Kc[i] for i in range(H)], axis=0)
    Qh = np.stack([Oi[i].T @ Qc[i] for i in range(H)])
    a, _ = damage_key(Qh, Kbar, Qc, Kc, ab)
    b, _ = damage_key_via_E(Oi, Kbar, Qc, Kc, ab)
    agree = abs(a - b) / max(abs(a), 1e-300)

    Gl = np.stack([rng.normal(size=(d_k, d_k)) for _ in range(H)])
    Gli = np.stack([np.linalg.inv(g) for g in Gl])
    Kbar2 = np.mean([Gl[i] @ Kc[i] for i in range(H)], axis=0)
    Qh2 = np.stack([Gli[i].T @ Qc[i] for i in range(H)])
    a2, _ = damage_key(Qh2, Kbar2, Qc, Kc, ab)
    b2, _ = damage_key_via_E(Gli, Kbar2, Qc, Kc, ab)
    disagree = abs(a2 - b2) / max(abs(a2), 1e-300)
    return agree, disagree
