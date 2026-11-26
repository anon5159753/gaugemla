# src: the method library

Training needs nothing from `src/`: each run script carries its own MLA architecture and loss and reads the t = 0
checkpoints from Drive. GaugeMLA's initialisation (Pythia-410m, w = 64, as in the Colab notebooks)
imports the 36 modules below, plus `requirements.txt`, which it installs; the other 35 build the other arms. Its
module check expects every file it pins to be present and unchanged, so keep `src/` whole.

## GaugeMLA in the code

- Initialisation: `dt_mla_ours` solves each layer's latent and read-out under the KL objective (`build_layer`,
  `build_weights`); `dt_mla_unrope` un-ropes the rotary rows that do not fit, at the widths `dt_mla_unrope_width`
  gives; `dt_mla_kl_standin` gives each un-roped plane its read-out $A^{\mathrm{KL}}_{ip}$ (`measure`, `standin`,
  `install`); `dt_mla_unrope_arms` (`build`) and `dt_mla_w64_ours_unrope_cost_kl` make it the arm.
- Fits: in `experiments/kl_standin_50m/p410m_klgc/p410m_klgc_init.py`, `fit_fisher` fits F<sub>i</sub> with the 1%
  token exclusion and `fit_sigma` fits Σ; the KL stand-in is measured on the Fisher's own pass, and
  `compute_mla_width` gives the width table.
- Training: `p410m_klgc_run.py` beside it (`train_branch`, `kd_loss`, the gradient clipped at 1.0; `load_ckpt` loads
  a saved model). Every other arm trains the same way.

## The 36 modules GaugeMLA's initialisation imports

### GaugeMLA

| module | role |
|---|---|
| `dt_mla_w64_ours_unrope_cost_kl` | GaugeMLA's arm at w = 64: a thin wrapper that names its settings and applies the KL stand-in. |
| `dt_mla_kl_standin` | The KL stand-in: `measure` gathers its statistics on the Fisher fit's pass, `standin` gives each un-roped plane's read-out (F^aa)^-1 F^ab and the metric F^aa, and `install` hands them to the builder. |
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
`dt_calib`, `dt_data`, `dt_eval_tasks`, `dt_score`, `dt_logs`, `dt_smoltalk`: imported by the modules above, for the
other constructions, the task battery, scoring and logging.

## The other 35 modules

| modules | what they build |
|---|---|
| `dt_mla_w16_ours_unrope_cost_kl` | GaugeMLA at w = 16: the grouped-query models' initialisation imports it in place of the w = 64 arm. |
| `dt_mla_w64_ours_unrope_cost`, `dt_mla_w16_ours_unrope_cost` | $A_p^{\mathrm{mean}}$, the un-roping ablation: GaugeMLA with each un-roped plane read out by its mean rotation over the offsets (the earlier GaugeMLA). |
| `dt_mla_w64_ours_unrope_cost_klsi`, `dt_mla_w16_ours_unrope_cost_klsi` | The ablation Σ = I: GaugeMLA with the input metric set to the identity. |
| `dt_mla_w64_transmla`, `dt_mla_w64_mha2mla`, `dt_mla_w64_care`, `dt_mla_mha2mla`, `dt_mla_care` | The baselines on the multi-head models: TransMLA, MHA2MLA and CARE at w = 64. |
| `dt_mla_gqa_native`, `dt_mla_w16_transmla_native`, `dt_mla_w16_care_native`, `dt_mla_w16_mha2mla_native` | The baselines on the grouped-query models (w = 16), each following its authors' released code: `dt_mla_gqa_native` builds all three on the stored key and value heads; the other three are their arms. |
| `dt_mla_care_gqa` | CARE's solve on a grouped-query model's own key and value rows, called by `dt_mla_gqa_native`. |
| `dt_mla_w16_transmla`, `dt_mla_w16_mha2mla`, `dt_mla_w16_care` | The earlier w = 16 builds of the three baselines, replaced by the native arms and run by no script here. |
| `dt_mla_fit_llama` | Σ, the Fisher and MHA2MLA's statistics on LLaMA-family models. |
| `gam_record` | Records the gauges of the "Mean pool + optimal gauge" construction. |
| `dt_mla_ours_unrope_cost` | $A_p^{\mathrm{mean}}$ at the KL width: in the 5M kit's registry, but neither built by the notebooks nor reported in the paper. |
| `dt_mla_w64_ours_unrope_cost_fi`, `dt_mla_w64_ours_unrope_cost_si`, `dt_mla_w64_ours_unrope_cost_ii`, `dt_mla_w16_ours_unrope_cost_fi`, `dt_mla_w16_ours_unrope_cost_si`, `dt_mla_w16_ours_unrope_cost_ii`, `dt_mla_ours_unrope_energy`, `dt_mla_w64_ours_unrope_energy`, `dt_mla_w16_ours_unrope_energy`, `dt_mla_w64_ours_unrope_tmcopy`, `dt_mla_w64_ours_unrope_tmcopy_fi`, `dt_mla_w64_ours_unrope_tmdamped`, `dt_mla_ours_eig`, `dt_mla_w64_ours` | Registered variants of the earlier GaugeMLA that none of this repository's experiments runs. |
