# Experiments: what the names mean

Each `<experiment>/<tag>/` holds `<tag>_init.py`, which builds the t = 0 checkpoints, and `<tag>_run.py`, which
trains each branch by knowledge distillation (in `true_fisher_x1_seeds/`, `kl_standin_ablation/` and `regauge_test/`
it only measures the checkpoints).

| experiment | what it holds | data in `results/` |
|---|---|---|
| `kl_standin_50m/` (7 models) | GaugeMLA, 50M tokens | `kl_standin_50m/<model>/` |
| `grad_clip/` (7 models) | TransMLA, MHA2MLA, CARE and $A_p^{\mathrm{mean}}$, 50M tokens | `<model>_gc/` |
| `kl_standin_50m/kl_full_eval/` | GaugeMLA's t = 0 initialisations and saved 50M models on the full task splits | `kl_standin_eval/<model>/` |
| `kl_standin_5m/`, `grad_clip_5m/` (Pythia-410m) | GaugeMLA and every other construction, 5M tokens | code only |
| `kl_standin_ablation/` (14 kits), `true_fisher_x1_seeds/` (7 kits) | t = 0 only: GaugeMLA at four more draws of F_i (`_kls`), Σ = I at all five (`_klsi`), $A_p^{\mathrm{mean}}$ at four more (`_seeds`) | code only |
| `grad_clip/full_eval/`, `kl_standin_ablation/kl_abl_eval/` | the other full-split evaluations, as the batch scripts they ran as | code only |
| `regauge_test/` (Pythia-410m) | t = 0 only: the regauge test, described in its own [README](regauge_test/README.md) | code only |

## Running a kit

`colab/launch.py` runs any kit: `launch.run("<tag>", "init")` on a High-RAM CPU runtime, then
`launch.run("<tag>", "run", BRANCH=..., TEST_TOKENS=...)` on a GPU. Models gated on the Hugging Face Hub need the
Colab secret `HF_TOKEN`. The paper's runs used 48 GB and 96 GB GPUs; larger teachers need more GPU memory
(StableLM-2-1.6B's 12B teacher alone takes 48 GB in fp32).

A training kit writes under `MyDrive/gaugemla/<experiment>/<tag>/`: `init/` (the t = 0 checkpoints), the fits
(`fisher/`, `sigma/`, ...) and `runs/<stamp>/` (the rows, and `models/`). In the Colab notebooks every kit copies
GaugeMLA's fits of F_i and Σ, so every method is built on the same fits, as in the paper. Finished checkpoints and
branches are skipped, so a rerun continues where the last one stopped.

## Names

- A branch is `<arm>_G<G>_s42` and its t = 0 checkpoint is `<arm>_B<B>_t0`. B is the width cached per token per
  layer, keys and values together, and G = B / (2 d_k). The share of the original KV cache is B / (2 H_kv d_k), so
  G = 4, 2, 1 is 25, 12.5, 6.25% on Pythia-410m. `s42` is the training seed.
- `w64` and `w16` give the rotary width d_h^R: 64 on the multi-head models, 16 on the grouped-query ones.
- GaugeMLA's KL objective has two measured factors: the output metric F_i, the true Fisher with each fit document's
  top 1% of tokens (by gradient) left out, and the input metric Σ, measured on the calibration documents.
- `FISHER_LABEL_SEED` picks the draw of next tokens the Fisher is fitted on: 20261014 for every 50M run, 20262014–20265014
  for `true_fisher_x1_seeds/` ($A_p^{\mathrm{mean}}$) and the `_kls` kits (GaugeMLA), and all five for the `_klsi` kits.

## Arms

| arm | name in the paper | what it is | used in |
|---|---|---|---|
| `mla_w64_ours_unrope_cost_kl`, `mla_w16_ours_unrope_cost_kl` | **GaugeMLA (our method)** | MLA initialised under the KL objective, with the rotary key cached after RoPE as in MLA. The rotary planes that do not fit are un-roped, each read out by its head's KL stand-in $A^{\mathrm{KL}} = (F^{aa})^{-1} F^{ab}$, and the un-roped rows enter the latent weighted by $F^{aa}$. | `kl_standin_50m/`, `kl_standin_5m/`, `kl_standin_ablation/` (`_kls`) |
| `mla_w64_ours_unrope_cost`, `mla_w16_ours_unrope_cost` | $A_p^{\mathrm{mean}}$ (ablation) | GaugeMLA with each un-roped plane read out by its mean rotation over the causal offsets (the earlier GaugeMLA). | `grad_clip/`, `true_fisher_x1_seeds/` |
| `mla_w64_ours_unrope_cost_klsi`, `mla_w16_ours_unrope_cost_klsi` | Σ = I (ablation) | GaugeMLA with the input metric set to the identity, at five Fisher label seeds. | `kl_standin_ablation/` (`_klsi`) |
| `mla_w64_transmla_sigma`, `mla_w16_transmla_native_sigma` | TransMLA | TransMLA's conversion (Meng et al., 2025), from the same measured Σ. On the grouped-query models (`w16`) it follows TransMLA's released code: it compresses the stored key and value heads (each once, not its copies per query head), then is written into the per-head layout the other arms use. | `grad_clip/` |
| `mla_w64_mha2mla_s2norm`, `mla_w16_mha2mla_native_s2norm` | MHA2MLA | MHA2MLA's conversion (Ji et al., 2025), which un-ropes the least-contributing key dimensions by its 2-norm criterion. On the grouped-query models (`w16`) it follows MHA2MLA's released code: dimensions are ranked per stored KV head, each kept one is cached once, and the SVD runs on the stored key and value rows. | `grad_clip/` |
| `mla_w64_care_sigma`, `mla_w16_care_native_sigma` | CARE | CARE's covariance-aware conversion (Zhou et al., 2026), from the same measured Σ, with TransMLA's rotary path. On the grouped-query models (`w16`) the solve runs on the stored key and value heads, as CARE's released code does, then is written into the per-head layout. | `grad_clip/` |
| `decoupled_optwide_fisher_sigma_dkr64` | Decoupled RoPE, KL objective, d_h^R = 64 | One key–value latent solved under the KL objective, with a rotary block of width 64 decoupled, cached before RoPE and read out per head. | `grad_clip_5m/` |
| `pool` | Mean pool (GQA) | Average the key and value maps of each group of consecutive heads, with the gauges at the identity. | `grad_clip_5m/` |
| `gauge3B` | Mean pool + optimal gauge | Keep the group averages and solve each head's gauge against them. | `grad_clip_5m/` |
| `procrustes_simk` | Procrustes-aligned mean pool | Rotate each group's heads into alignment before averaging (Jin et al., 2025), with each layer's groups chosen by their combinatorial search over a similarity criterion. | `grad_clip_5m/` |
| `procrustes` | Procrustes-aligned mean pool, one KV head | The same alignment with every head in one group (6.25%), where there are no groups to choose. | `grad_clip_5m/` |
| `optimal` | Optimal subspace per group | Keep GQA's consecutive groups and solve for each group's optimal key and value subspaces. | `grad_clip_5m/` |
| `optwide` | Optimal subspace per layer | One key subspace and one value subspace shared by every head of the layer; at one KV head it is the optimal subspace per group. | `grad_clip_5m/` |
| `mha` | MHA (uncompressed) | The original model, distilled under the same recipe, as a reference (branch `mha_G0`). | `grad_clip_5m/` |

The registry also names the earlier `w16` builds of the three baselines (`mla_w16_transmla_sigma`,
`mla_w16_mha2mla_s2norm`, `mla_w16_care_sigma`), on a copy of each KV head for every query head that reads it. The
`_native` arms replace them; no script here builds or trains them, and `results/` holds none of their runs.
`grad_clip_5m/`'s registry also holds two arms the paper no longer reports, `mla_w64_ours_unrope_cost` and
`mla_ours_unrope_cost` ($A_p^{\mathrm{mean}}$ at 5M, and at the width that minimises the KL objective); the Colab
notebooks neither build nor train them.
