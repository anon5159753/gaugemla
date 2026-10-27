import argparse
import os

import numpy as np
import torch
from scipy.linalg import expm, logm

import gam_core as gc
import gam_solve as gs

T_EVAL = 512
GAUGE_ARMS = ("arm3A", "arm3B")

def factor_dense(A):
    n = A.shape[0]
    sgn = float(np.sign(np.linalg.det(A)))
    J = np.eye(n)
    if sgn < 0:
        J[-1, -1] = -1.0
    Ap = A @ J
    U, sig, Vt = np.linalg.svd(Ap)
    if np.linalg.det(U) < 0:
        U, Vt = U.copy(), Vt.copy()
        U[:, -1] *= -1
        Vt[-1, :] *= -1
    return np.real(logm(U)), np.log(sig), np.real(logm(Vt)), sgn

def unfactor_dense(S1, s, S2, sgn):
    n = S1.shape[0]
    J = np.eye(n)
    if sgn < 0:
        J[-1, -1] = -1.0
    return expm(S1) @ np.diag(np.exp(s)) @ expm(S2) @ J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--arms", default="arm3A,arm3B",
                    help="arm2/arm2b have no gauge (Omega=I); include them only for their maps")
    ap.add_argument("--cases", default="B")
    ap.add_argument("--groups", default="1,2,4,8")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "gauges"))
    a = ap.parse_args()
    torch.set_num_threads(8)
    short = a.model.split("/")[-1]
    os.makedirs(a.outdir, exist_ok=True)

    model, tok = gc.kvlab.load_model(a.model)
    layers = gc.kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    theta = gc.inv_freq(model)
    print(f"gam_record -- {a.model}  L={L} H={H} d_k={d_k} d_v={d_v} dm={dm} r={r}")
    print(f"  one gauge per query head per layer: {H} Omega + {H} Gamma per layer, "
          f"{H * L} of each per file\n")

    total = 0.0
    for G in [int(x) for x in a.groups.split(",")]:
        groups = gc.kvlab.groups_of(H, G)
        assert sorted(i for g in groups for i in g) == list(range(H)), "groups must partition heads"
        assert len({len(g) for g in groups}) == 1, "groups must be equal-sized"
        for name in a.arms.split(","):
            for case in a.cases.split(","):
                if name == "arm3B" and case == "A":
                    continue
                store = {"_meta": np.array([L, H, d_k, d_v, dm, r, G], dtype=np.int64),
                         "_groups": np.array(groups, dtype=np.int64),
                         "_arm": np.array([name.encode()]), "_case": np.array([case.encode()])}
                worst = 0.0
                for l in range(L):
                    Qc, Kc, Vc, Wo, sh, shipped = gc.layer_maps(layers[l], H, d_k, d_v, dm, case)
                    arm = gs.solve_grouped(name, Qc, Kc, Vc, Wo, groups, d_k, d_v, r,
                                           theta=theta, T=T_EVAL)
                    Kb, Vb, Qh, Oh = gc.serve_maps(arm, sh, shipped, case, groups)
                    store[f"Kbar_{l}"] = Kb.astype(np.float32)
                    store[f"Vbar_{l}"] = Vb.astype(np.float32)
                    if arm["Om"] is not None:
                        store[f"Omega_{l}"] = arm["Om"]
                        store[f"Gamma_{l}"] = arm["Gm"]
                        store[f"Omega_inv_{l}"] = arm["Oi"]
                        store[f"Gamma_inv_{l}"] = arm["Gi"]
                        if l == 0:
                            for i in range(H):
                                S1, s, S2, sg = factor_dense(arm["Gm"][i])
                                worst = max(worst, float(np.abs(
                                    unfactor_dense(S1, s, S2, sg) - arm["Gm"][i]).max()))
                p = os.path.join(a.outdir, f"gauges_{short}_G{G}_{name}_{case}.npz")
                np.savez_compressed(p, **store)
                mb = os.path.getsize(p) / 1e6
                total += mb
                print(f"  G={G:<2} {name:<6} Sigma={case}  ->  {os.path.basename(p)}  "
                      f"{mb:7.1f} MB   (factorisation round-trip {worst:.1e})", flush=True)
    print(f"\n  total {total:.0f} MB in {a.outdir}/")

if __name__ == "__main__":
    main()
