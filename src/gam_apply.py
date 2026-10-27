import glob
import os

import numpy as np

import gam_core as gc
import gam_solve as gs

T_EVAL = 512

def list_saved(outdir="gauges", model=None):
    out = []
    for p in sorted(glob.glob(os.path.join(outdir, "gauges_*.npz"))):
        stem = os.path.basename(p)[len("gauges_"):-len(".npz")]
        parts = stem.rsplit("_", 3)
        if len(parts) != 4:
            continue
        m, g, arm, case = parts
        if model and model.split("/")[-1] != m:
            continue
        out.append((p, m, int(g[1:]), arm, case))
    return out

def load_meta(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    L, H, d_k, d_v, dm, r, G = [int(x) for x in z["_meta"]]
    groups = [list(map(int, g)) for g in z["_groups"]]
    return dict(L=L, H=H, d_k=d_k, d_v=d_v, dm=dm, r=r, G=G, groups=groups,
                arm=bytes(z["_arm"][0]).decode(), case=bytes(z["_case"][0]).decode())

def factorise(Omega, Gamma, r):
    from gam_record import factor_dense
    H, d_k = Omega.shape[0], Omega.shape[1]
    d_v = Gamma.shape[1]
    h = r // 2
    out = {"rho": np.zeros((H, h)), "phi": np.zeros((H, h))}
    dS1, dS2, ds, dsg = [], [], [], []
    gS1, gS2, gs_, gsg = [], [], [], []
    for i in range(H):
        Om = Omega[i]
        c = np.array([complex(Om[j, j], Om[j + h, j]) for j in range(h)])
        out["rho"][i], out["phi"][i] = np.log(np.abs(c)), np.angle(c)
        if d_k > r:
            a, b, cc, d = factor_dense(Om[r:, r:])
            dS1.append(a); ds.append(b); dS2.append(cc); dsg.append(d)
        a, b, cc, d = factor_dense(Gamma[i])
        gS1.append(a); gs_.append(b); gS2.append(cc); gsg.append(d)
    if dS1:
        out.update(dS1=np.stack(dS1), ds=np.stack(ds), dS2=np.stack(dS2), dsign=np.array(dsg))
    out.update(gS1=np.stack(gS1), gs=np.stack(gs_), gS2=np.stack(gS2), gsign=np.array(gsg))
    return out

def apply_saved(model, npz_path):
    z = np.load(npz_path, allow_pickle=False)
    m = load_meta(npz_path)
    L, H, d_k, d_v, dm, r = m["L"], m["H"], m["d_k"], m["d_v"], m["dm"], m["r"]
    groups, arm, case = m["groups"], m["arm"], m["case"]
    layers = gc.kvlab.get_layers(model)
    Lm, Hm, dkm, dvm, dmm, rm = gc.geometry(model)
    if (Lm, Hm, dkm, dvm, dmm, rm) != (L, H, d_k, d_v, dm, r):
        raise ValueError(f"checkpoint geometry {(Lm, Hm, dkm, dvm, dmm, rm)} != file {(L, H, d_k, d_v, dm, r)}")

    for l, lay in enumerate(layers):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        Kbar = z[f"Kbar_{l}"].astype(np.float64)
        Vbar = z[f"Vbar_{l}"].astype(np.float64)
        if f"Omega_{l}" in z.files:
            Oi = z[f"Omega_inv_{l}"] if f"Omega_inv_{l}" in z.files else \
                np.stack([np.linalg.inv(z[f"Omega_{l}"][i]) for i in range(H)])
            Gi = z[f"Gamma_inv_{l}"] if f"Gamma_inv_{l}" in z.files else \
                np.stack([np.linalg.inv(z[f"Gamma_{l}"][i]) for i in range(H)])
            Qh = np.stack([Oi[i].T @ Qt[i] for i in range(H)])
            Oh = np.stack([Wo[i] @ Gi[i] for i in range(H)])
        elif arm == "arm2b":
            Qc, Kc, Vc, Wo_, sh, shipped = gc.layer_maps(lay, H, d_k, d_v, dm, case)
            a = gs.solve_grouped("arm2b", Qc, Kc, Vc, Wo_, groups, d_k, d_v, r,
                                 theta=gc.inv_freq(model), T=T_EVAL)
            _, _, Qh, Oh = gc.serve_maps(a, sh, shipped, case, groups)
        else:
            Qh, Oh = Qt, Wo
        gc.write_layer(lay, Qh, gc.tie(Kt, Kbar, groups), gc.tie(Vt, Vbar, groups), Oh,
                       H, d_k, d_v, dm)
    return m

def kv_heads(npz_path):
    m = load_meta(npz_path)
    return m["G"], m["H"]

if __name__ == "__main__":
    rows = list_saved()
    if not rows:
        print("no saved conversions under gauges/ -- run gam_record.py first")
    for p, mdl, G, arm, case in rows:
        sz = os.path.getsize(p) / 1e6
        print(f"  {mdl:<14} G={G:<3} {arm:<6} Sigma={case}   {sz:7.1f} MB   {p}")
