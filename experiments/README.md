# Experiments: what the names mean

Each `<experiment>/<tag>/` holds two scripts. `<tag>_init.py` builds the t = 0 checkpoints, and `<tag>_run.py`
trains each branch by knowledge distillation (in `true_fisher_x1_seeds/`, `kl_standin_ablation/` and `regauge_test/`
it only measures the checkpoints). `colab/launch.py` runs either one.

`results/` holds the logged 50M training runs and GaugeMLA's full-split evaluation:

- `grad_clip/<model>_gc/` → `results/<model>_gc/`: TransMLA, MHA2MLA, CARE and $A_p^{\mathrm{mean}}$.
- `kl_standin_50m/<model>_klgc/` → `results/kl_standin_50m/<model>/`: GaugeMLA.
- `kl_standin_50m/kl_full_eval/` → `results/kl_standin_eval/<model>/`: GaugeMLA's t = 0 initialisations and saved
  50M models scored on the full task splits.

The other experiments are here as code: `grad_clip_5m/` and `kl_standin_5m/` (Pythia-410m, 5M tokens),
`true_fisher_x1_seeds/` and `kl_standin_ablation/` (the t = 0 seed study and the Σ = I ablation), `regauge_test/`
(MHA2MLA, TransMLA, CARE and GaugeMLA built from a regauged Pythia-410m in `p410m_rgkv/`, and GQA's mean pool from
the same regauged checkpoints in `p410m_rgkv_gqa/`, which has not been run yet; its own
[README](regauge_test/README.md) defines the gauges and gives the paper's tables), and the evaluation scripts
`grad_clip/full_eval/` and `kl_standin_ablation/kl_abl_eval/`. The paper reports their results, except the GQA
regauge rows, which are pending.

## Names

- A branch is `<arm>_G<G>_s42` and its t = 0 checkpoint is `<arm>_B<B>_t0`. B is the width cached per token per
  layer, keys and values together, and G = B / (2 d_k). The share of the original KV cache is B / (2 H_kv d_k), so
  G = 4, 2, 1 is 25, 12.5, 6.25% on Pythia-410m. `s42` is the training seed.
- `w64` and `w16` give the rotary width d_h^R of the MLA conversions: 64 on the multi-head models, 16 on the
  grouped-query ones.
- GaugeMLA's KL objective has two measured factors: the output metric F_i, fitted as the true Fisher with each fit
  document's top 1% of tokens (by gradient) left out, and the input metric Σ, measured on the calibration documents.
- `FISHER_LABEL_SEED` picks the draw of the next tokens the Fisher is fitted on. Every 50M run trains from the draw at
  20261014; `true_fisher_x1_seeds/` ($A_p^{\mathrm{mean}}$) and the `_kls` kits (GaugeMLA) take 20262014–20265014, and
  the `_klsi` kits all five.

## Arms

| arm | name in the paper | what it is | used in |
|---|---|---|---|
| `mla_w64_ours_unrope_cost_kl`, `mla_w16_ours_unrope_cost_kl` | **GaugeMLA (our method)** | MLA initialised under the KL objective, with the rotary key cached after RoPE as in MLA. The rotary planes that do not fit are un-roped, each read out by its head's KL stand-in $A^{\mathrm{KL}} = (F^{aa})^{-1} F^{ab}$, and the un-roped rows enter the latent weighted by $F^{aa}$. | `kl_standin_50m/`, `kl_standin_5m/`, `kl_standin_ablation/` (`_kls`) |
| `mla_w64_ours_unrope_cost`, `mla_w16_ours_unrope_cost` | $A_p^{\mathrm{mean}}$ (ablation) | GaugeMLA with each un-roped plane read out by its mean rotation over the causal offsets instead (the earlier GaugeMLA). | `grad_clip/`, `true_fisher_x1_seeds/` |
| `mla_w64_ours_unrope_cost_klsi`, `mla_w16_ours_unrope_cost_klsi` | Σ = I (ablation) | GaugeMLA with the input metric set to the identity, at five Fisher label seeds. | `kl_standin_ablation/` (`_klsi`) |
| `mla_w64_transmla_sigma`, `mla_w16_transmla_sigma` | TransMLA | TransMLA's conversion (Meng et al., 2025), built from the same measured Σ. | `grad_clip/` |
| `mla_w64_mha2mla_s2norm`, `mla_w16_mha2mla_s2norm` | MHA2MLA | MHA2MLA's conversion (Ji et al., 2025), which un-ropes the least-contributing key dimensions, chosen by its 2-norm criterion. | `grad_clip/` |
| `mla_w64_care_sigma` | CARE | CARE's covariance-aware conversion (Zhou et al., 2026), built from the same measured Σ. Not built on the grouped-query models. | `grad_clip/` |
| `decoupled_optwide_fisher_sigma_dkr64` | Decoupled, KL objective, d_h^R = 64 | One key–value latent solved under the KL objective, with a rotary block of width 64 decoupled, cached before RoPE and read out per head. | `grad_clip_5m/` |
| `pool` | GQA | Average the key and value maps of each group of consecutive heads, with the gauges at the identity. | `grad_clip_5m/` |
| `gauge3B` | GQA + optimal gauge | Keep the group averages and solve each head's gauge against them. | `grad_clip_5m/` |
| `procrustes_simk` | Procrustes alignment | Rotate each group's heads into alignment before averaging (Jin et al., 2024), with each layer's groups chosen by their combinatorial search over a similarity criterion on calibration activations. | `grad_clip_5m/` |
| `procrustes` | Procrustes alignment, one KV head | The same alignment with every head in one group (6.25%), where there are no groups to choose; it serves as `procrustes_simk` at that budget. | `grad_clip_5m/` |
| `optimal` | Per-group subspace | Keep GQA's consecutive groups and solve for each group's optimal key and value subspaces. | `grad_clip_5m/` |
| `optwide` | Joint key, joint value subspace | One key subspace and one value subspace, each shared by every head of the layer; at one KV head it is the per-group subspace. | `grad_clip_5m/` |
| `mha` | MHA (uncompressed) | The original model, distilled under the same recipe, as a reference (branch `mha_G0`). | `grad_clip_5m/` |

`grad_clip_5m/`'s registry also holds two arms the paper no longer reports: `mla_w64_ours_unrope_cost`
($A_p^{\mathrm{mean}}$ at 5M; the 5M GaugeMLA rows are `kl_standin_5m/`'s) and `mla_ours_unrope_cost` ($A_p^{\mathrm{mean}}$
at the rotary width that minimises the KL objective). The scripts keep them as they ran; the Colab notebooks neither
build nor train them.
