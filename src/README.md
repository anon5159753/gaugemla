# src: the method library

To run GaugeMLA on Pythia-410m (w = 64, as in the Colab notebooks):

- **Training needs nothing from `src/`.** Each run script (`experiments/<experiment>/<tag>/<tag>_run.py`) carries its
  own MLA architecture and loss, and reads the t = 0 checkpoints from Google Drive.
- **Initialisation imports the 35 modules below**, plus `requirements.txt`, which it installs. The other 25 build the
  other arms. The initialisation's module check expects every file it pins to be present and unchanged in the clone,
  so keep `src/` whole.

## The 35 modules GaugeMLA's initialisation imports

### GaugeMLA

| module | role |
|---|---|
| `dt_mla_w64_ours_unrope_cost` | The arm `mla_w64_ours_unrope_cost`, GaugeMLA at rotary width w = 64: a thin wrapper that names its settings. |
| `dt_mla_unrope_arms` | Builds every un-roped GaugeMLA arm from one recipe. |
| `dt_mla_unrope` | GaugeMLA with the rotary rows that do not fit in the rotary width un-roped rather than dropped. |
| `dt_mla_unrope_width` | The rotary width rule when those rows are un-roped. |
| `dt_mla_ours` | GaugeMLA's initialiser: the latent and gauges that minimise the KL objective. |
| `dt_mla_transmla` | TransMLA's initialiser; `dt_mla_unrope` uses its rotary-plane helpers (pairing, allocation, rotation matrices, eigendecomposition). |
| `dt_mla_w64`, `dt_mla_w16` | The shared definitions of the fixed-width MLA arms: w = 64, and w = 16 for the grouped-query models. |

### The MLA model, its checkpoints and the registry

| module | role |
|---|---|
| `dt_mla_arch` | The MLA architecture the converted model runs on (GPT-NeoX host, as Pythia). |
| `dt_mla_arch_llama` | The same on a LLaMA-family host. |
| `dt_checkpoint` | Saves and reloads a converted model as a checkpoint with its conversion record. |
| `dt_feasibility_arms` | The dispatcher from an arm name to its initialiser. |
| `dt_feasibility_train` | Model setup, the build driver (`init_cells`) and the digests of the fits. |
| `dt_grid_models` | The registry: every model and its cell table. |
| `dt_util` | Small shared helpers. |

### Fits and the width table

| module | role |
|---|---|
| `kvlab` | Loads the model and gives access to its layers. |
| `gam_core` | Reads each layer's attention weights and geometry (rotary frequencies). |
| `gam_core_llama` | The same for LLaMA-family checkpoints. |
| `dt_decoupled`, `dt_optimal` | The initialisation's width functions (`compute_mla_width`, `compute_dstar`) build on their metrics and offset weights. |
| `gam_solve`, `gam_apply` | Closed-form conversions and their application, imported alongside. |

### Imported with the other constructions

`dt_arms`, `dt_arch_llama`, `dt_gqa_arch`, `dt_optimal_arch`, `dt_decoupled_arch`, `dt_procrustes`, `dt_groupsearch`,
`dt_calib`, `dt_data`, `dt_eval_tasks`, `dt_score`, `dt_logs`, `dt_smoltalk`: the modules above import them. They
serve the other constructions (grouped-query pooling, Procrustes alignment, the subspace and decoupled arms) and
the task battery, scoring and logging.

## The other 25 modules

| modules | what they build |
|---|---|
| `dt_mla_w64_transmla`, `dt_mla_w64_mha2mla`, `dt_mla_w64_care`, `dt_mla_w16_transmla`, `dt_mla_w16_mha2mla`, `dt_mla_mha2mla`, `dt_mla_care` | The baselines: TransMLA, MHA2MLA and CARE at w = 64 (and the first two at w = 16). |
| `dt_mla_w64_ours_unrope_cost_fi`, `dt_mla_w64_ours_unrope_cost_si`, `dt_mla_w64_ours_unrope_cost_ii`, `dt_mla_w16_ours_unrope_cost_fi`, `dt_mla_w16_ours_unrope_cost_si`, `dt_mla_w16_ours_unrope_cost_ii` | The ablations: F_i = I, Σ = I, both. |
| `dt_mla_w16_ours_unrope_cost` | GaugeMLA at w = 16: the grouped-query models' initialisation imports it in place of the w = 64 arm. |
| `dt_mla_ours_unrope_cost` | GaugeMLA at the KL width (the 5M comparison). |
| `dt_mla_fit_llama` | Σ, the Fisher and MHA2MLA's statistics on LLaMA-family models. |
| `gam_record` | Records the gauges of the "GQA + optimal gauge" construction. |
| `dt_mla_ours_unrope_energy`, `dt_mla_w64_ours_unrope_energy`, `dt_mla_w16_ours_unrope_energy`, `dt_mla_w64_ours_unrope_tmcopy`, `dt_mla_w64_ours_unrope_tmcopy_fi`, `dt_mla_w64_ours_unrope_tmdamped`, `dt_mla_ours_eig`, `dt_mla_w64_ours` | Registered GaugeMLA variants that none of this repository's experiments runs. |
