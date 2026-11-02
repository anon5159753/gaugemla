# GaugeMLA: code, data and Colab notebooks

The attention mechanism has free parameters that do not change its output: a trained model's key and value weights
are one of many bases, related by a gauge symmetry, that give the same query–key and output–value circuits.
Converting a model to a smaller KV cache is an initialisation followed by uptraining, and these free parameters can
be used to limit the damage the initialisation does to attention. Earlier work on tying key and value heads uses
only orthogonal matrices as the gauge, the rotations of the query–key inner product (Jin et al., 2024; TransMLA).
GaugeMLA is the initialisation that naturally arises from acknowledging the whole symmetry, and it fully leverages
the gauge freedom for a superior initialisation. It converts a model's attention to Multi-head Latent Attention
(MLA) by minimising the KL divergence to the original model over every change of basis the gauge allows; rotary
positional embeddings restrict the gauge, and decoupling the rotary key, as MLA does, restores it.

## The gauge symmetry of attention

Write $x \in \mathbb{R}^{1 \times d_m}$ for a token and $X \in \mathbb{R}^{T \times d_m}$ for a sequence of $T$ tokens.
Head $i$ forms queries, keys and values $Q_i = X W_{Q_i}^\top$, $K_i = X W_{K_i}^\top$ and $V_i = X W_{V_i}^\top$, and
its output projection $W_{O_i}$ writes the result back into the residual stream. With the query–key (QK) and
output–value (OV) circuits

$$M_i = W_{Q_i}^\top W_{K_i}, \qquad N_i = W_{O_i} W_{V_i},$$

attention reads

$$\mathrm{Attn}(X) = \sum_{i=1}^{H} \mathrm{softmax}\left(\frac{X M_i X^\top}{\sqrt{d_k}}\right) X N_i^\top .$$

The weights enter only through the two circuits. For any invertible $\Omega_i \in \mathrm{GL}(d_k)$ and
$\Gamma_i \in \mathrm{GL}(d_v)$, the change of basis

$$W_{K_i} \mapsto \Omega_i W_{K_i}, \quad W_{Q_i} \mapsto \Omega_i^{-\top} W_{Q_i}, \qquad W_{V_i} \mapsto \Gamma_i W_{V_i}, \quad W_{O_i} \mapsto W_{O_i} \Gamma_i^{-1}$$

leaves both circuits, and so the output of attention, unchanged:

$$M_i \mapsto \left(\Omega_i^{-\top} W_{Q_i}\right)^\top \Omega_i W_{K_i} = W_{Q_i}^\top \Omega_i^{-1} \Omega_i W_{K_i} = M_i, \qquad N_i \mapsto W_{O_i} \Gamma_i^{-1} \Gamma_i W_{V_i} = N_i .$$

The gauges $\Omega_i$ and $\Gamma_i$ are free parameters: the original model does not depend on them, but once heads
share a compressed cache they decide how much each head is damaged (Figure 1). Orthogonal gauges, for which
$\Omega_i^{-\top} = \Omega_i$, are the rotations of the query–key inner product that earlier work uses; GaugeMLA uses
the whole of $\mathrm{GL}$. With rotary positional embeddings the logit between positions $m$ and $n$ is
$x_m W_{Q_i}^\top R_\delta W_{K_i} x_n^\top$ with $\delta = n - m$, so the key gauge must commute with every rotation,
$\Omega_i R_\delta = R_\delta \Omega_i$; decoupling the rotary key, as MLA does, leaves the rest of the key's gauge free.

![Figure 1: A single attention layer for MHA, GQA, and Gauged GQA](docs/figure1.png)

Figure 1: A single attention layer for MHA, GQA, and Gauged GQA. (a) MHA: H query, key and value heads, each caching
its own key and value. (b) GQA: the H key and value heads are replaced by group maps $\bar{W}$ and $\bar{U}$.
(c) Gauged GQA: the cache is GQA's, and each head reads it through its own gauges Ω<sub>i</sub><sup>-1</sup> and
Γ<sub>i</sub><sup>-1</sup>, which are never cached and help recover the original model's QK and OV circuits.

The figure shows Gauged GQA, not GaugeMLA: it is the basic motivation of GaugeMLA, which fully leverages this gauge
freedom, choosing every change of basis to limit the damage to attention, for a superior initialisation.

## The KL objective

What initialisation is for is the converted model's output, so GaugeMLA minimises the KL divergence from the original
model,

$$\mathrm{KL}\left(p_O \,\|\, p_S\right) = \mathbb{E}_{v \sim p_O}\left[\log p_O(v) - \log p_S(v)\right],$$

where $p_O$ is the original model's next-token distribution and $p_S$ the converted model's. With the rotary key
decoupled, keys and values both read a single content latent $\bar{C}$, and head $i$'s damage is

$$\Delta_i^{K} = W_{K_i}^{\mathrm{nr}} - \Omega_i^{-1} \bar{C}, \qquad \Delta_i^{V} = W_{V_i} - \Gamma_i^{-1} \bar{C},$$

where $W_{K_i}^{\mathrm{nr}}$ is the non-rotary part of the key map. Expanding to second order in $\Delta_i$,

$$\mathrm{KL}\left(p_O \,\|\, p_S\right) \approx \tfrac{1}{2} \sum_i \mathrm{tr}\left(\Delta_i^\top F_i \Delta_i \Sigma\right),$$

where the sum runs over keys and values ($\Delta_i$ is $\Delta_i^{K}$ or $\Delta_i^{V}$). The two factors say which
directions of the damage matter:

- $\Sigma = \mathbb{E}[x^\top x]$, **the input metric** (one per layer), is the second moment of the attention input
  (after the layer norm), from one forward pass. Damage along input directions the layer rarely sees costs little;
  $\Sigma = I$ counts every input direction alike.
- $F_i = \mathbb{E}[g^\top g]$, **the output metric** (one per head, for keys and for values), is the second moment of
  the loss gradient $g$ at that head's key or value output, from one forward and one backward pass with the next
  token at each position drawn from the model's own prediction: the second-order term of the KL is the Fisher
  information of the original model. Damage along key or value directions the next-token distribution is
  insensitive to costs little; $F_i = I$ counts every output direction alike.

Both are measured on a calibration set of 64 passages of 512 tokens from The Pile, disjoint from the evaluation
passages; TransMLA, MHA2MLA and CARE measure their statistics on the same passages. For each passage and layer, the
1% of tokens with the largest gradient at the key and value projections are left out of $F_i$'s average: a few tokens
dominate it, chiefly each passage's first token, which becomes an attention sink with an extreme gradient. $\Sigma$
keeps every token.

GaugeMLA chooses the latent, the gauges and which rotary rows to un-rope by this objective. The rotary planes that do
not fit in the shared rotary width $d_h^{R}$ are un-roped: plane $p$ of head $i$ has its rotation, which depends on
the offset $\delta$, replaced by the fixed map that minimises an approximation of the KL objective,

$$R(\theta_p \delta) \;\longmapsto\; A^{\mathrm{KL}}_{ip} = \left(F^{aa}_{ip}\right)^{-1} F^{ab}_{ip},$$

where $F^{aa}_{ip}$ is the second moment of the gradient at the plane's keys with the rotation removed, and
$F^{ab}_{ip}$ its cross moment with the gradient itself. The approximation keeps the KL objective to second order and
drops the cross terms between keys at different positions. The un-roped rows are initialised as
$(W_{K_i}^{\mathrm{nr}})_p = A^{\mathrm{KL}}_{ip} (W_{K_i}^{\mathrm{rot}})_p$, with $(\cdot)_p$ plane $p$'s two rows.

The paper ablates two parts of this initialisation at conversion. The un-roping is replaced by $A_p^{\mathrm{ramp}}$,
the mean rotation of plane $p$ over the causal offsets (the earlier GaugeMLA), and $\Sigma$ is set to $I$. The next
token drawn for $F_i$ is random, so the paper reports each t = 0 result of GaugeMLA and its ablations as the mean over
five draws.

## Results

GaugeMLA almost always initialises the compressed model so that its attention mechanism is better preserved, an
advantage which persists through 50M tokens of knowledge distillation. Only attention is converted, so the KL
divergence to the original model measures what the conversion loses; the KL divergence to the teacher is what
distillation minimises.

<!-- generated by make_readme_tables.py from the paper's tables; every number re-derived from results/ -->

GaugeMLA has the lowest KL of the MLA conversions in 21 of 21 model–budget pairs after 50M tokens of knowledge distillation, and in 21 of 21 at t = 0, to the original model and to the teacher alike.

**KL to the teacher and to the original model, at t = 0 and after 50M tokens of knowledge distillation.**

| model | method | KL to the | 6.25%, t = 0 | 6.25%, 50M | 12.5%, t = 0 | 12.5%, 50M | 25%, t = 0 | 25%, 50M |
|---|---|---|---|---|---|---|---|---|
| Pythia-410m | GaugeMLA | teacher | **3.384** | **0.662** | **1.236** | **0.461** | **0.500** | **0.379** |
|  | GaugeMLA | original | **3.083** | **0.367** | **0.927** | **0.168** | **0.177** | **0.086** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 3.421 | 0.669 | 1.371 | 0.464 | 0.621 | 0.379 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 3.119 | 0.375 | 1.065 | 0.170 | 0.302 | 0.087 |
|  | TransMLA | teacher | 4.561 | 0.681 | 2.990 | 0.481 | 1.134 | 0.395 |
|  | TransMLA | original | 4.253 | 0.387 | 2.684 | 0.185 | 0.823 | 0.101 |
|  | MHA2MLA | teacher | 4.768 | 0.774 | 3.930 | 0.541 | 2.862 | 0.431 |
|  | MHA2MLA | original | 4.465 | 0.481 | 3.626 | 0.247 | 2.556 | 0.140 |
|  | CARE | teacher | 4.567 | 0.681 | 2.919 | 0.482 | 1.090 | 0.412 |
|  | CARE | original | 4.259 | 0.387 | 2.613 | 0.185 | 0.778 | 0.118 |
| StableLM-2-1.6B | GaugeMLA | teacher | **2.099** | **0.534** | **0.856** | **0.432** | **0.449** | **0.376** |
|  | GaugeMLA | original | **1.797** | **0.241** | **0.540** | **0.139** | **0.119** | **0.081** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 2.117 | 0.537 | 0.912 | 0.434 | 0.534 | 0.379 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 1.815 | 0.245 | 0.598 | 0.141 | 0.208 | 0.083 |
|  | TransMLA | teacher | 2.976 | 0.541 | 0.975 | 0.441 | 0.497 | 0.383 |
|  | TransMLA | original | 2.680 | 0.250 | 0.660 | 0.146 | 0.167 | 0.088 |
|  | MHA2MLA | teacher | 7.901 | 0.698 | 4.534 | 0.561 | 3.857 | 0.480 |
|  | MHA2MLA | original | 7.601 | 0.408 | 4.237 | 0.270 | 3.556 | 0.185 |
|  | CARE | teacher | 2.913 | 0.542 | 0.976 | 0.442 | 0.497 | 0.384 |
|  | CARE | original | 2.616 | 0.251 | 0.662 | 0.147 | 0.168 | 0.088 |
| Qwen1.5-0.5B | GaugeMLA | teacher | **3.919** | **0.746** | **1.515** | **0.540** | **0.693** | **0.463** |
|  | GaugeMLA | original | **3.501** | **0.399** | **1.075** | **0.198** | **0.224** | **0.123** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 4.037 | 0.755 | 1.697 | 0.545 | 0.877 | 0.469 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 3.618 | 0.408 | 1.264 | 0.203 | 0.420 | 0.129 |
|  | TransMLA | teacher | 5.222 | 0.765 | 4.766 | 0.565 | 2.442 | 0.482 |
|  | TransMLA | original | 4.812 | 0.417 | 4.354 | 0.220 | 2.012 | 0.141 |
|  | MHA2MLA | teacher | 6.512 | 1.097 | 4.888 | 0.829 | 3.977 | 0.688 |
|  | MHA2MLA | original | 6.094 | 0.741 | 4.468 | 0.480 | 3.556 | 0.341 |
|  | CARE | teacher | 5.247 | 0.766 | 4.686 | 0.566 | 2.424 | 0.482 |
|  | CARE | original | 4.836 | 0.418 | 4.273 | 0.221 | 1.994 | 0.141 |
| OLMo-1B | GaugeMLA | teacher | **2.651** | **0.463** | **1.160** | **0.388** | **0.651** | **0.348** |
|  | GaugeMLA | original | **2.418** | **0.282** | **0.909** | **0.208** | **0.378** | **0.168** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 3.091 | 0.469 | 2.304 | 0.397 | 2.142 | 0.358 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 2.861 | 0.289 | 2.071 | 0.215 | 1.903 | 0.175 |
|  | TransMLA | teacher | 5.011 | 0.484 | 4.173 | 0.408 | 3.329 | 0.380 |
|  | TransMLA | original | 4.785 | 0.301 | 3.946 | 0.224 | 3.098 | 0.194 |
|  | MHA2MLA | teacher | 7.210 | 0.581 | 7.762 | 0.487 | 4.762 | 0.439 |
|  | MHA2MLA | original | 6.985 | 0.394 | 7.541 | 0.301 | 4.536 | 0.252 |
|  | CARE | teacher | 5.043 | 0.486 | 4.058 | 0.409 | 3.314 | 0.387 |
|  | CARE | original | 4.817 | 0.303 | 3.831 | 0.224 | 3.083 | 0.201 |
| SmolLM-360M | GaugeMLA | teacher | **3.989** | **1.305** | **2.802** | **0.763** | **1.686** | **0.544** |
|  | GaugeMLA | original | **3.598** | **0.944** | **2.405** | **0.397** | **1.282** | **0.175** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 4.279 | 1.328 | 3.235 | 0.798 | 2.457 | 0.580 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 3.887 | 0.966 | 2.841 | 0.433 | 2.060 | 0.210 |
|  | TransMLA | teacher | 4.319 | 1.696 | 3.954 | 1.034 | 3.605 | 0.744 |
|  | TransMLA | original | 3.928 | 1.338 | 3.564 | 0.671 | 3.211 | 0.379 |
|  | MHA2MLA | teacher | 7.819 | 2.312 | 10.701 | 1.763 | 10.772 | 1.262 |
|  | MHA2MLA | original | 7.438 | 1.949 | 10.301 | 1.395 | 10.375 | 0.895 |
| TinyLlama-1.1B | GaugeMLA | teacher | **4.401** | **0.859** | **3.576** | **0.523** | **2.931** | **0.371** |
|  | GaugeMLA | original | **4.166** | **0.674** | **3.343** | **0.339** | **2.696** | **0.189** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 4.237 | 0.860 | 3.701 | 0.528 | 3.318 | 0.379 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 4.001 | 0.674 | 3.464 | 0.345 | 3.080 | 0.199 |
|  | TransMLA | teacher | 4.898 | 1.022 | 4.299 | 0.620 | 4.172 | 0.452 |
|  | TransMLA | original | 4.664 | 0.834 | 4.061 | 0.436 | 3.935 | 0.269 |
|  | MHA2MLA | teacher | 9.657 | 1.634 | 9.897 | 1.186 | 8.887 | 0.838 |
|  | MHA2MLA | original | 9.421 | 1.441 | 9.665 | 0.996 | 8.646 | 0.648 |
| Llama-3.2-1B | GaugeMLA | teacher | **5.837** | **0.789** | **3.678** | **0.590** | **2.180** | **0.479** |
|  | GaugeMLA | original | **5.484** | **0.481** | **3.323** | **0.285** | **1.824** | **0.176** |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | teacher | 5.711 | 0.802 | 4.833 | 0.601 | 3.974 | 0.491 |
|  | $A_p^{\mathrm{ramp}}$ (ablation) | original | 5.359 | 0.493 | 4.483 | 0.295 | 3.623 | 0.190 |
|  | TransMLA | teacher | 5.840 | 1.010 | 5.279 | 0.773 | 5.023 | 0.640 |
|  | TransMLA | original | 5.487 | 0.694 | 4.925 | 0.461 | 4.673 | 0.330 |
|  | MHA2MLA | teacher | 7.533 | 1.381 | 6.560 | 1.086 | 5.328 | 0.864 |
|  | MHA2MLA | original | 7.175 | 1.061 | 6.207 | 0.767 | 4.972 | 0.551 |

GaugeMLA is our method; $A_p^{\mathrm{ramp}}$ is its un-roping ablation (the earlier GaugeMLA, which un-ropes by the mean rotation over the offsets), shown beside it and left out of the bolding. Bold is the lowest of GaugeMLA, TransMLA, MHA2MLA and CARE for that model, budget, time and KL. At t = 0, GaugeMLA and $A_p^{\mathrm{ramp}}$ are the initialisation each run trained from; the paper reports the mean over five draws of $F_i$. CARE is not performed for the grouped-query models.

## About this repository

Code and logged results for GaugeMLA experiments. Each model's multi-head/grouped-query attention is converted to MLA
by GaugeMLA and by the methods it is compared with (TransMLA and MHA2MLA on all seven models, CARE on the four
multi-head ones), then trained by knowledge distillation toward a larger teacher of its family, with the gradient
norm clipped at 1.0 every step. GaugeMLA's initialisation fits the true Fisher with each fit document's top 1% of
tokens (by gradient) left out.

## Requirements

- **A Google Colab Pro+ account.**
- **A Hugging Face token**, added as the Colab secret `HF_TOKEN` with notebook access.
- **Initialisation: a High-RAM CPU runtime** (Runtime → Change runtime type → CPU, High-RAM). It uses no GPU.
- **Training: an A100 GPU runtime with High-RAM.**
- **Google Drive space.** A Pythia-410m checkpoint is about 1.6 GB in fp32. GaugeMLA alone builds 3 initial
  checkpoints; the full 50M grid has 15 and the 5M grid 21 more (3 of them copies of GaugeMLA's), and each finished
  50M branch saves one more model.

## Run it on Colab (Pythia-410m)

We provide notebooks on Colab for easy access to GPU resources. They have been tested, so they run without change.
Upload a notebook from `colab/` to Colab (File → Upload notebook), set its runtime, and run all cells. Run the
initialisation first.

**GaugeMLA, our method, is the arm `mla_w64_ours_unrope_cost_kl`** (kits `p410m_klgc` at 50M and `p410m_kl5m` at
5M). The arm `mla_w64_ours_unrope_cost` is its un-roping ablation $A_p^{\mathrm{ramp}}$ (the earlier GaugeMLA), which
is built and trained with the baselines.

**To run GaugeMLA, use `colab/pythia410m_gaugemla.ipynb` only.** It builds GaugeMLA's three t = 0 checkpoints and
trains them, and nothing else. The other three notebooks are for the complete experiments, GaugeMLA beside every
method it is compared with: they build every method's checkpoints (15 for the 50M grid and 21 more for the 5M grid,
against GaugeMLA's 3), so they take much longer to initialise.

| notebook | runtime | what it does |
|---|---|---|
| **GaugeMLA only** | | |
| `colab/pythia410m_gaugemla.ipynb` | A100, High-RAM | GaugeMLA's t = 0 checkpoints, then 50M-token training of its three branches (25, 12.5, 6.25%) |
| **The full experiments** | | |
| `colab/pythia410m_init.ipynb` | CPU, High-RAM | the t = 0 checkpoints of every method, in four cells: GaugeMLA, then TransMLA, MHA2MLA, CARE and $A_p^{\mathrm{ramp}}$ (the 50M grid); GaugeMLA, then the other constructions (the 5M grid) |
| `colab/pythia410m_train_50M.ipynb` | A100, High-RAM | 50M-token training: GaugeMLA's 3 branches in one cell, the 12 of TransMLA, MHA2MLA, CARE and $A_p^{\mathrm{ramp}}$ in the next; saves each final model |
| `colab/pythia410m_train_5M.ipynb` | A100, High-RAM | 5M-token training: GaugeMLA's 3 branches in one cell, the other constructions' 18 (the MHA control included) in the next |

Each notebook mounts Google Drive, clones only the parts of the repository it uses, and starts that experiment's
script. Everything is written under `MyDrive/gaugemla/`: `kl_standin_50m/p410m_klgc/` (GaugeMLA) and
`grad_clip/p410m_gc/` (the rest) for the 50M grid, and `kl_standin_5m/p410m_kl5m/` and `grad_clip_5m/p410m_gc5m/`
for the 5M grid. Each has `init/` (checkpoints), the fits (`fisher/`, `sigma/`, ...) and `runs/<stamp>/` (rows, and
`models/` for saved models). GaugeMLA's initialisation fits $F_i$ and $\Sigma$, and every other kit copies them, so
every method is built on the same fits, as in the paper. Training reads its t = 0 checkpoints from the same Drive
folders, so mount the same Google account in every notebook; the original model, the teacher and the text are
downloaded from the Hugging Face Hub in each session. Finished checkpoints and branches are skipped, so a rerun
continues where the last session stopped.

Set `TEST_TOKENS = 50000` for a quick test. The experiments in the paper trained on 50M tokens (5M for the 5M grid),
with more powerful computing resources.

## Data (`results/`)

The logged 50M training runs of every method, and GaugeMLA's evaluation on the full task splits, are here as JSON.
Weights and fitted matrices (`.npz`) are not included, but the fits and t = 0 checkpoints can be initialised quickly
(`colab/pythia410m_init.ipynb`). The paper's other experiments (the 5M comparison of constructions, the five draws of
$F_i$ and the ablations) are included as code in `experiments/`, not as data.

| folder | contents |
|---|---|
| `results/kl_standin_50m/<model>/` | **GaugeMLA's 50M training runs**, 7 models, at 25 / 12.5 / 6.25% of the KV cache. `G<G>.jsonl` has one row per evaluation point (KL to the teacher and to the original model with their standard errors and per-document values, the task battery at selected points, gradient norms), beside the stamp's `manifest.json` and `reference.json`, `init/` (each t = 0 checkpoint's record) and `models/` (each saved model's config and conversion record). `NAMES.txt` maps these names to the original run names and gives each model's G per budget. `README.md` says how GaugeMLA differs from its ablation $A_p^{\mathrm{ramp}}$ in `results/<model>_gc/`. |
| `results/<model>_gc/` | **The other methods' 50M training runs**, 7 models, the same recipe, evaluation documents and training stream: TransMLA and MHA2MLA on every model, CARE on the four multi-head ones, and the ablation $A_p^{\mathrm{ramp}}$ (arm `mla_w*_ours_unrope_cost`). `runs/<stamp>/kd_*_<branch>_s42.jsonl` has the same rows, beside the stamp's `_manifest.json`, reference and summary rows, and `models/*/`. `init/` holds each t = 0 checkpoint's record and `_cells.json`. |
| `results/kl_standin_eval/<model>/` | **GaugeMLA on the full task splits**: its t = 0 initialisations (`G<G>_t0.json`) and saved 50M models (`G<G>_50M.json`), and the original model (`original.json`), each on the ten tasks. `NAMES.txt` maps the names to the original ones. |

The task battery logged in the run rows has 13 tasks. Ten are the paper's: LAMBADA, ARC-Easy, ARC-Challenge, PIQA,
HellaSwag, OpenBookQA, WinoGrande, BoolQ, SIQA and MMLU. Three are synthetic probes that are logged but not reported
in the paper:

- `induction`: a random 48-token sequence followed by its first 4 tokens again; the model must continue it with the
  next 8 tokens by copying.
- `kv_synth_N50`: fifty lines of the form "The code for ⟨word⟩ is ⟨number⟩.", then one of the words again; the model
  must recall its number.
- `arith_1d`: eight worked examples of one-digit addition, then a new pair of digits to add.

| tag | model | teacher |
|---|---|---|
| `olmo1b_gc` | allenai/OLMo-1B-hf | allenai/OLMo-7B-hf (float32) |
| `p410m_gc` | EleutherAI/pythia-410m | EleutherAI/pythia-2.8b (float32) |
| `slm1p6b_gc` | stabilityai/stablelm-2-1_6b | stabilityai/stablelm-2-12b (float32) |
| `q0p5b_gc` | Qwen/Qwen1.5-0.5B | Qwen/Qwen1.5-4B (float32) |
| `l3_1b_gc` | meta-llama/Llama-3.2-1B | meta-llama/Llama-3.1-8B (float32) |
| `tl1b_gc` | TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | meta-llama/Llama-2-7b-hf (float32) |
| `sm360m_gc` | HuggingFaceTB/SmolLM-360M | HuggingFaceTB/SmolLM2-1.7B (float32) |

## Our method in the code

GaugeMLA's own code, apart from the baselines':

- **Initialisation.** `src/dt_mla_ours.py` solves each layer's latent and read-out under the KL objective
  (`build_layer`, `build_weights`). `src/dt_mla_unrope.py` builds on it with the rotary rows that do not fit
  un-roped, and `src/dt_mla_unrope_width.py` gives their widths. `src/dt_mla_kl_standin.py` gives each un-roped
  plane its read-out $A^{\mathrm{KL}}_{ip}$ (`measure`, `standin`, `install`). `src/dt_mla_unrope_arms.py` (`build`)
  and `src/dt_mla_w64_ours_unrope_cost_kl.py` make it the arm `mla_w64_ours_unrope_cost_kl`.
- **The KL objective's two factors** are fitted in the initialisation script,
  `experiments/kl_standin_50m/p410m_klgc/p410m_klgc_init.py`: `fit_fisher` fits F<sub>i</sub>, with the 1% token
  exclusion, and `fit_sigma` fits Σ. The KL stand-in is measured on the Fisher's own pass, in the same script.
  `compute_mla_width` gives the width table.
- **Training.** `experiments/kl_standin_50m/p410m_klgc/p410m_klgc_run.py` distils GaugeMLA (`train_branch`,
  `kd_loss`, gradient clipped at 1.0), and every other arm's run script trains the same way; `load_ckpt` loads a
  saved model.

[`src/README.md`](src/README.md) lists every module the initialisation imports.

## Code

- `src/`: the method library (conversions, fits, architectures, and the registry `dt_grid_models.py`).
  [`src/README.md`](src/README.md) lists the modules GaugeMLA's initialisation needs; training needs none of them.
- `experiments/<experiment>/<tag>/<tag>_init.py`, `<tag>_run.py`: one pair per model and experiment, all in the
  same form. [`experiments/README.md`](experiments/README.md) says what each arm name means. `colab/launch.py` runs any of them: `launch.run("<tag>", "init")` on a High-RAM CPU runtime, then
  `launch.run("<tag>", "run", BRANCH=..., TEST_TOKENS=...)` on a GPU. The `_seeds`, `_kls` and `_klsi` kits also take
  `FISHER_LABEL_SEED` (20262014, 20263014, 20264014 or 20265014, and 20261014 for `_klsi`). Models gated on the
  Hugging Face Hub need a Colab secret `HF_TOKEN` with access. The paper's runs used 48 GB and 96 GB GPUs; larger
  teachers need more GPU memory (StableLM-2-1.6B's 12B teacher alone takes 48 GB in fp32).
  - `kl_standin_50m/` (7 models): GaugeMLA, 50M tokens, gradient clipping 1.0, final models saved.
  - `grad_clip/` (7 models): TransMLA, MHA2MLA, CARE and $A_p^{\mathrm{ramp}}$, the same training.
  - `kl_standin_5m/` and `grad_clip_5m/` (Pythia-410m): GaugeMLA and every other construction, 5M tokens, the same
    training.
  - `kl_standin_ablation/` (14 kits) and `true_fisher_x1_seeds/` (7 kits), t = 0 only: GaugeMLA at four more draws of
    $F_i$ (`_kls`), the ablation Σ = I at all five (`_klsi`), and $A_p^{\mathrm{ramp}}$ at four more (`_seeds`).
  - `kl_standin_50m/kl_full_eval/`, `grad_clip/full_eval/` and `kl_standin_ablation/kl_abl_eval/`: the full-split task
    evaluations, in the cluster form they ran in.

For anonymity, comments and docstrings have been removed from the code, and names and absolute paths have been
redacted from strings in `results/`. All numbers are as logged. Hashes recorded in `results/` refer to the original
files.
