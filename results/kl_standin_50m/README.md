# GaugeMLA: the 50M training runs

These are the runs of **GaugeMLA, the method of the paper**: arm `mla_w64_ours_unrope_cost_kl` on the multi-head
models and `mla_w16_ours_unrope_cost_kl` on the grouped-query ones, at 25, 12.5 and 6.25% of the KV cache.

GaugeMLA un-ropes the rotary planes that do not fit in the rotary width. Each plane's rotation is replaced by the
KL stand-in $A^{\mathrm{KL}}_{ip} = (F^{aa}_{ip})^{-1} F^{ab}_{ip}$, the fixed map that minimises an approximation of
the KL objective for that plane of head $i$ (the `_kl` in the arm name).

The arm `mla_w{64,16}_ours_unrope_cost` in `results/<model>_gc/` is **not GaugeMLA but its ablation
$A_p^{\mathrm{ramp}}$**, the earlier version of the method. It is the same construction, except that each un-roped
plane is replaced by its average rotation over the causal offsets, counting every query–key pair equally, in place of
$A^{\mathrm{KL}}_{ip}$.

Both were trained the same way:
- 50M tokens of knowledge distillation toward the same teacher, with the gradient clipped at 1.0;
- the same evaluation documents and training stream;
- initialised from the same $F_i$ and $\Sigma$ (Fisher label seed 20261014).

Per model, `G<G>.jsonl` has one row per evaluation point: KL to the teacher and to the original model, with standard
errors and per-document values, CE, the task battery at selected points, and gradient norms. Beside it are the run's
`manifest.json` and `reference.json`, `init/` (each t = 0 checkpoint's record) and `models/` (each saved model's
config). `NAMES.txt` gives each model's G per budget and the original file names. The same models scored on the full
task splits are in `results/kl_standin_eval/`.
