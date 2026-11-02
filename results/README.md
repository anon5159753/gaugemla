# results: the 50M training runs and GaugeMLA's task evaluation

**Only `kl_standin_50m/` is GaugeMLA, our method.** The `mla_w{64,16}_ours_unrope_cost` runs in `<model>_gc/` are
GaugeMLA without the KL stand-in: the ablation $A_p^{\mathrm{ramp}}$, the earlier version of the method.

| folder | what it is |
|---|---|
| `kl_standin_50m/<model>/` | **GaugeMLA, the method of the paper**: arm `mla_w64_ours_unrope_cost_kl` on the multi-head models and `mla_w16_ours_unrope_cost_kl` on the grouped-query ones, trained for 50M tokens at 25, 12.5 and 6.25% of the KV cache. |
| `<model>_gc/` | The methods it is compared with, trained the same way: TransMLA, MHA2MLA, CARE (multi-head models only), and **$A_p^{\mathrm{ramp}}$, GaugeMLA's un-roping ablation** (arm `mla_w{64,16}_ours_unrope_cost`). |
| `kl_standin_eval/<model>/` | GaugeMLA's t = 0 initialisations and saved 50M models, and the original model, scored on the full splits of the ten tasks. |

## GaugeMLA and $A_p^{\mathrm{ramp}}$

Both un-rope the rotary planes that do not fit in the rotary width. They differ only in what replaces each plane's
rotation:

- **GaugeMLA** (`kl_standin_50m/`, the `_kl` in the arm name) uses the KL stand-in
  $A^{\mathrm{KL}}_{ip} = (F^{aa}_{ip})^{-1} F^{ab}_{ip}$, the fixed map that minimises an approximation of the KL
  objective for plane $p$ of head $i$.
- **$A_p^{\mathrm{ramp}}$** (`<model>_gc/`, arm `mla_w{64,16}_ours_unrope_cost`) uses the plane's average rotation over
  the causal offsets, counting every query–key pair equally. It is the earlier version of GaugeMLA, now an ablation,
  and not the method of the paper.

Every run in both folders was trained the same way:
- 50M tokens of knowledge distillation toward the same teacher, with the gradient clipped at 1.0;
- the same evaluation documents and training stream;
- GaugeMLA and $A_p^{\mathrm{ramp}}$ initialised from the same $F_i$ and $\Sigma$ (Fisher label seed 20261014).

## Files

- `kl_standin_50m/<model>/G<G>.jsonl` and `<model>_gc/runs/<stamp>/kd_*_<branch>_s42.jsonl` have one row per
  evaluation point: KL to the teacher and to the original model with standard errors and per-document values, CE,
  the task battery at selected points, and gradient norms.
- Beside them are each run's manifest and reference, the t = 0 checkpoints' records (`init/`) and the saved models'
  configs (`models/`).
- `kl_standin_50m/NAMES.txt` gives each model's G per budget (G = 4, 2, 1 is 25, 12.5, 6.25% on Pythia-410m) and the
  original file names. `kl_standin_eval/NAMES.txt` does the same for the evaluation files.
