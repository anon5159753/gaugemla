import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OFE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, OFE)
sys.path.insert(0, HERE)
import dt_grid_models as R
import regauge_gauges as RG

TAG = "p410m_rgkv"
DRIVE = "/content/drive/MyDrive/gaugemla/regauge_test/p410m_rgkv"
STAMP = f"grid-{TAG}-pythia-2.8b-float32-50M-lr2e-05-s42"

def main():
    m = R.MODELS[TAG]
    RG.check_registry(m["regauge"])
    cells = R.cells(TAG)
    build = [c for c in cells if c["arm"] != "mha" and (c["init"] == "BUILD" or c["init"].startswith("DEDUP:"))]
    assert all(c["init"] == "BUILD" and c["train"] == () for c in build), "every cell is built and measured at t = 0 only"
    assert all(c["arm"] != "pool" for c in build), "no mean pool in this kit"
    cells_all = [(c["arm"], c["G"], c["B"]) for c in build]
    H, d_k, r = m["H"], m["d_k"], m["r"]
    d_v = d_k
    full = 2 * m["Hkv"] * d_k
    print(f"{TAG}: {m['hf_id']} @ {m['revision'][:12]}, registry {R.registry_sha(TAG)[:16]}; H={H} d_k={d_k} d_v={d_v} r={r} "
          f"({r // 2} rotary planes, {d_k - r} non-rotary rows); label seed {R.fisher_label_seeds(m)[0]}, "
          f"exclusion {R.fisher_exclude_top(m):.0%}; t0_only {R.t0_only(m)}")
    print(f"registry BUILD cells: {len(build)} = {len(set(c[0] for c in cells_all))} methods x {len(set(c[2] for c in cells_all))} "
          f"budgets, every one at w = {build[0]['rot_w']} ({build[0]['rot_rule']}); no mean pool")
    n = nf = 0
    for cond in RG.CONDITION_ORDER:
        s = RG.CONDITIONS[cond]
        mine = RG.cells_for(cond, cells_all)
        g = RG.GAUGES[s["gauge"]]
        print(f"\n=== {cond}  [{'PRIMARY' if s['fisher'] == 'measured' else 'SECONDARY'}]  gauge {s['gauge']}: {g['text']}")
        print(f"    Fisher: {'measured on the regauged checkpoint (label seed as above, 1% exclusion recomputed)' if s['fisher'] == 'measured' else 'identity/fisher carried over: GK -> Omega^-T GK Omega^-1, GV -> Gamma^-T GV Gamma^-1, excluded positions kept'}")
        print(f"    process: REGAUGE={cond}  gate -> {DRIVE}/{cond}/gate.json  fits -> {DRIVE}/{cond}/{{fisher,sigma,klstandin"
              f"{',s2norm' if s['fisher'] == 'measured' else ''}}}  rows -> {DRIVE}/runs/{STAMP}-g{cond}/t0_only/")
        if s["fisher"] == "measured":
            nf += 1
            print(f"    F{nf}. full cache: the regauged model itself, B={full} (100% of the KV cache)  tensors -> {cond}/full_cache/"
                  f"attention.npz  row -> kd_{STAMP}-g{cond}_full_cache_t0only.jsonl"
                  + ("  [the instrument check: KL to the teacher within 0.01 of 0.3374, KL to the original < 0.01]" if cond == "identity" else ""))
        for arm, G, B in mine:
            n += 1
            print(f"    {n:>2}. {arm:<30} B={B:<4} G={G}  {B / full:.4%} of the KV cache  -> {cond}/init/ckpt/{arm}_B{B}_t0")
    print(f"\n{n} inits in all: {len(RG.PRIMARY)} primary conditions x {len(cells_all)} (the four methods x three budgets) + "
          f"{len(RG.SECONDARY)} secondary x {len(RG.cells_for(RG.SECONDARY[0], cells_all))}; + {nf} full-cache rows; control = "
          f"the identity condition's rows; every row on the same 64 held-out passages (TF32 off).")
    print("the gauges on this geometry (Omega_i = c_i diag(P per plane, both rows alike; N per non-rotary row), Gamma_i = c_i diag(V)):")
    for gname, gs in RG.GAUGES.items():
        sp = RG.spec(gname, H, d_k, r, m["Hkv"], d_v)
        def short(v):
            return f"{v[0]:g}, {v[1]:g}, ..., {v[-1]:g}" if len(v) > 3 and len(set(v)) > 1 else (f"all {v[0]:g}" if len(set(v)) == 1 else str(v))
        print(f"  {gname:<9} c_i over heads 1..{H}: {[int(c) if float(c).is_integer() else c for c in sp['c_per_head']]}; "
              f"P = {short(sp['plane_diag_P'])} ({len(sp['plane_diag_P'])}); N = {short(sp['nonrotary_diag_N'])} "
              f"({len(sp['nonrotary_diag_N'])}); V = {short(sp['value_diag_V'])} ({len(sp['value_diag_V'])})")
    if n != 42 or nf != 3:
        raise SystemExit(f"expected 42 inits and 3 full-cache rows, got {n} and {nf}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
