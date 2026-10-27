# Experiments: what the names mean

Each `<experiment>/<tag>/` holds two scripts. `<tag>_init.py` builds the t = 0 checkpoints, and `<tag>_run.py`
trains each branch by knowledge distillation (in `init_ablation/` and `true_fisher_x1_seeds/` it only measures the
checkpoints). `colab/launch.py` runs either one. The logged results are under the same tag: `results/<tag>/` for
`grad_clip/` and `grad_clip_5m/`, `results/init_ablation/<tag>/`, and `results/tfx1_seeds/<tag>/` for
`true_fisher_x1_seeds/`. `grad_clip/full_eval/` scores the saved 50M models on the full task splits
(`results/full_eval/`).

## Names

- A branch is `<arm>_G<G>_s42` and its t = 0 checkpoint is `<arm>_B<B>_t0`. B is the width cached per token per
  layer, keys and values together, and G = B / (2 d_k). The share of the original KV cache is B / (2 H_kv d_k), so
  G = 4, 2, 1 is 25, 12.5, 6.25% on Pythia-410m. `s42` is the training seed.
- `w64` and `w16` give the rotary width d_h^R of the MLA conversions: 64 on the multi-head models, 16 on the
  grouped-query ones.
- GaugeMLA's KL objective has two measured factors: the output metric F_i, fitted as the true Fisher with each fit
  document's top 1% of tokens (by gradient) left out, and the input metric Σ, measured on the calibration documents.

## Arms

| arm | name in the paper | what it is | used in |
|---|---|---|---|
| `mla_w64_ours_unrope_cost`, `mla_w16_ours_unrope_cost` | GaugeMLA | MLA initialised under the KL objective, with the rotary key cached after RoPE as in MLA; the rotary rows that do not fit are un-roped. | `grad_clip/`, `grad_clip_5m/`, `true_fisher_x1_seeds/` |
| `mla_ours_unrope_cost` | GaugeMLA, KL width | As above, at the rotary width that minimises the KL objective. | `grad_clip_5m/` |
| `mla_w64_transmla_sigma`, `mla_w16_transmla_sigma` | TransMLA | TransMLA's conversion (Meng et al., 2025), built from the same measured Σ. | `grad_clip/` |
| `mla_w64_mha2mla_s2norm`, `mla_w16_mha2mla_s2norm` | MHA2MLA | MHA2MLA's conversion (Ji et al., 2025), which un-ropes the least-contributing key dimensions, chosen by its 2-norm criterion. | `grad_clip/` |
| `mla_w64_care_sigma` | CARE | CARE's covariance-aware conversion (Zhou et al., 2026), built from the same measured Σ. Not built on the grouped-query models. | `grad_clip/` |
| `mla_w64_ours_unrope_cost_fi`, `mla_w16_ours_unrope_cost_fi` | F_i = I | GaugeMLA with the output metric set to the identity. | `init_ablation/` (`_fi`) |
| `mla_w64_ours_unrope_cost_si`, `mla_w16_ours_unrope_cost_si` | Σ = I | GaugeMLA with the input metric set to the identity, at five Fisher label seeds. | `init_ablation/` (`_si`) |
| `mla_w64_ours_unrope_cost_ii`, `mla_w16_ours_unrope_cost_ii` | Both = I | GaugeMLA with both metrics set to the identity. | `init_ablation/` (`_ii`) |
| `decoupled_optwide_fisher_sigma_dkr64` | Decoupled, KL objective, d_h^R = 64 | One key–value latent solved under the KL objective, with a rotary block of width 64 decoupled, cached before RoPE and read out per head. | `grad_clip_5m/` |
| `pool` | GQA | Average the key and value maps of each group of consecutive heads, with the gauges at the identity. | `grad_clip_5m/` |
| `gauge3B` | GQA + optimal gauge | Keep the group averages and solve each head's gauge against them. | `grad_clip_5m/` |
| `procrustes_simk` | Procrustes alignment | Rotate each group's heads into alignment before averaging (Jin et al., 2024), with each layer's groups chosen by their combinatorial search over a similarity criterion on calibration activations. | `grad_clip_5m/` |
| `procrustes` | Procrustes alignment, one KV head | The same alignment with every head in one group (6.25%), where there are no groups to choose; it serves as `procrustes_simk` at that budget. | `grad_clip_5m/` |
| `optimal` | Per-group subspace | Keep GQA's consecutive groups and solve for each group's optimal key and value subspaces. | `grad_clip_5m/` |
| `optwide` | Joint key, joint value subspace | One key subspace and one value subspace, each shared by every head of the layer; at one KV head it is the per-group subspace. | `grad_clip_5m/` |
| `mha` | MHA (uncompressed) | The original model, distilled under the same recipe, as a reference (branch `mha_G0`). | `grad_clip_5m/` |
