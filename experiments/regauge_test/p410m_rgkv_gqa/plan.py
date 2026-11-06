import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OFE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, OFE)
sys.path.insert(0, HERE)
import dt_grid_models as R
import regauge_gauges as RG

TAG = "p410m_rgkv_gqa"
DRIVE = "/content/drive/MyDrive/gaugemla/regauge_test/p410m_rgkv_gqa"
STAMP = f"grid-{TAG}-pythia-2.8b-float32-50M-lr2e-05-s42"

def main():
    m = R.MODELS[TAG]
    RG.check_registry(m["regauge"])
    cells = R.cells(TAG)
    build = [c for c in cells if c["arm"] != "mha" and (c["init"] == "BUILD" or c["init"].startswith("DEDUP:"))]
    assert all(c["init"] == "BUILD" and c["train"] == () for c in build), "every cell is built and measured at t = 0 only"
    assert all(c["arm"] == RG.POOL_ARM for c in build), "the mean pool alone in this kit"
    cells_all = [(c["arm"], c["G"], c["B"]) for c in build]
    H, d_k, r = m["H"], m["d_k"], m["r"]
    d_v = d_k
    full = 2 * m["Hkv"] * d_k
    print(f"{TAG}: {m['hf_id']} @ {m['revision'][:12]}, registry {R.registry_sha(TAG)[:16]}; H={H} d_k={d_k} d_v={d_v} r={r} "
          f"({r // 2} rotary planes, {d_k - r} non-rotary rows); t0_only {R.t0_only(m)}")
    print(f"registry BUILD cells: {len(build)} = the mean pool (`{RG.POOL_ARM}`, kvlab.pool_layer over G groups of consecutive "
          f"heads) x {len(set(c[2] for c in cells_all))} budgets; no MLA arm, no *_xf, no full-cache row")
    n = 0
    for cond in RG.CONDITION_ORDER:
        s = RG.CONDITIONS[cond]
        mine = RG.cells_for(cond, cells_all)
        g = RG.GAUGES[s["gauge"]]
        print(f"\n=== {cond}  gauge {s['gauge']}: {g['text']}")
        print(f"    process: REGAUGE={cond}  gate -> {DRIVE}/{cond}/gate.json  fits (the template's, unread by the pool) -> "
              f"{DRIVE}/{cond}/{{fisher,sigma,klstandin}}  rows -> {DRIVE}/runs/{STAMP}-g{cond}/t0_only/")
        for arm, G, B in mine:
            n += 1
            ref = RG.IDENTITY_POOL_REFERENCE[B]
            extra = (f"  [instrument check: KL_o {ref['kl_original']:.6f}, KL_t {ref['kl_teacher']:.6f} +- {RG.IDENTITY_TOL:g}; "
                     f"ckpt {ref['ckpt_fingerprint']}]" if cond == "identity" else "")
            print(f"    {n:>2}. {arm:<5} B={B:<4} G={G}  {B / full:.4%} of the KV cache  -> {cond}/init/ckpt/{arm}_B{B}_t0  "
                  f"row kd_{STAMP}-g{cond}_{arm}_G{G}_t0only.jsonl{extra}")
        if cond == "identity":
            print(f"    then {RG.IDENTITY_CHECK_NAME} -> {DRIVE}/identity/ (kvs and kvd refuse to be measured without its PASS)")
    print(f"\n{n} inits in all: {len(RG.PRIMARY)} conditions x {len(cells_all)} budgets; control = the identity condition's rows; "
          f"every row on the same 64 held-out passages (sha256 {RG.EVAL_DOCS_SHA[:16]}), TF32 off.")
    print("the gauges on this geometry (Omega_i = c_i diag(P per plane, both rows alike; N per non-rotary row), Gamma_i = c_i diag(V)):")
    for gname in RG.GAUGES:
        sp = RG.spec(gname, H, d_k, r, m["Hkv"], d_v)

        def short(v):
            return f"{v[0]:g}, {v[1]:g}, ..., {v[-1]:g}" if len(v) > 3 and len(set(v)) > 1 else (f"all {v[0]:g}" if len(set(v)) == 1 else str(v))
        print(f"  {gname:<9} c_i over heads 1..{H}: {[int(c) if float(c).is_integer() else c for c in sp['c_per_head']]}; "
              f"P = {short(sp['plane_diag_P'])} ({len(sp['plane_diag_P'])}); N = {short(sp['nonrotary_diag_N'])} "
              f"({len(sp['nonrotary_diag_N'])}); V = {short(sp['value_diag_V'])} ({len(sp['value_diag_V'])})")
    print("predicted (exact arithmetic, reported never gated): the kvs and kvd pools are the same function (kvd = c_i D, D "
          "head-independent and commuting with RoPE, cancels on both sides), so their rows agree to float32 rounding")
    if n != 9:
        raise SystemExit(f"expected 9 inits, got {n}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
