# slm1p6b_urx1_fi init: builds the t = 0 checkpoints of every cell in the registry's table and writes them to Drive.
# Run by colab/launch.py from a Colab notebook, which mounts Drive, clones the repository to /content/repo
# and passes DRY_RUN as environment variables.

import os

DRIVE_ROOT = "/content/drive"
CKPT_ROOT  = DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/init/ckpt"
GRID_INIT_DIR = DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/init"
DATA_DIR   = DRIVE_ROOT + "/MyDrive/gaugemla/data"

if not os.path.isdir(DRIVE_ROOT + "/MyDrive"):
    raise RuntimeError(
        f"{DRIVE_ROOT}/MyDrive does not exist after drive.mount(). The mount did not complete -- "
        "usually the authorisation popup was dismissed or a different account was chosen. "
        "Re-run this cell and complete the prompt. ⛔ There is no alternative save path.")

for _name, _d in (("CKPT_ROOT", CKPT_ROOT), ("GRID_INIT_DIR", GRID_INIT_DIR), ("DATA_DIR", DATA_DIR)):
    if not _d.startswith(DRIVE_ROOT + "/"):
        raise RuntimeError(f"{_name} = {_d!r} is not under {DRIVE_ROOT}. ⛔ Drive is the only "
                           f"permitted location; fix the path, do not work around it.")

for _d in (CKPT_ROOT, GRID_INIT_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

for _d in (CKPT_ROOT, GRID_INIT_DIR, DATA_DIR):
    _probe = os.path.join(_d, ".write_probe")
    try:
        with open(_probe, "w") as _f:
            _f.write("probe")
        with open(_probe) as _f:
            assert _f.read() == "probe", "Drive read back something other than what was written"
        os.remove(_probe)
    except Exception as _e:
        raise RuntimeError(
            f"{_d} is mounted but NOT WRITABLE ({type(_e).__name__}: {_e}). Check the Drive quota and "
            f"that the mount is still authorised. ⛔ Do not redirect the output elsewhere -- Drive is "
            f"the only save path; fix Drive.") from _e
print(f"Drive writable: probe written, read back and removed under both output paths")
print(f"checkpoints <- {CKPT_ROOT}")
print(f"accounting  -> {GRID_INIT_DIR}/_cells.json")
print(f"task caches -> {DATA_DIR}")

# %%
import os
import subprocess

REPO_DIR = "/content/repo"
REPO_BRANCH = "main"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    raise RuntimeError(f"no clone of the repository at {REPO_DIR}: start this script from a colab/ notebook, "
                       "which clones it first")
_pipeline = os.path.join(REPO_DIR, "src")
if not os.path.isdir(_pipeline):
    raise RuntimeError(f"{_pipeline} is not in the clone at {REPO_DIR}")
_head = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
print(f"Repo ready at {REPO_DIR}  (branch {REPO_BRANCH}, {_head})")
print(f"  pipeline: {_pipeline}")

# %%
import sys

FEASIBILITY_DIR = os.path.join(REPO_DIR, "src")
assert os.path.isdir(FEASIBILITY_DIR), f"{FEASIBILITY_DIR} not found -- did the clone succeed?"
if FEASIBILITY_DIR not in sys.path:
    sys.path.insert(0, FEASIBILITY_DIR)
print(f"On sys.path: {FEASIBILITY_DIR}")

# %%
MODULE_MANIFEST = {'src/dt_arch_llama.py': '51d567d483b254779344b86d642d2c503df865ce',
 'src/dt_arms.py': 'cf744b5c44ee453d14c787aa8458b9c5250a5bb0',
 'src/dt_calib.py': '5327a03293f3c8f9cb052feef5d3eabb05dbebfc',
 'src/dt_checkpoint.py': '86994a5b7675ef790b9650e43604f71c27ac68a0',
 'src/dt_data.py': 'f217484eefa9d0fe4461a094f85b96fcb064bec9',
 'src/dt_decoupled.py': 'd8361a42d5a81e5bde4c29574f4d2770726520b4',
 'src/dt_decoupled_arch.py': '9ce54c9fc162cea408ff99abab7b7ff2b15a89b3',
 'src/dt_eval_tasks.py': 'e588eb6a89e5208d7caaf30b9cfd54a1756a472a',
 'src/dt_feasibility_arms.py': 'd0af1202c0f9dd5bea9129bdb4dec1c5599ea316',
 'src/dt_feasibility_train.py': '4c54fec049c8f7dccddf1131115ad8cde3ac2bdc',
 'src/dt_gqa_arch.py': '2dfeaef57063f9838ce3b7e87c2f4fa73401017d',
 'src/dt_grid_models.py': '1a79fa2302cd077a3b1c0b8cd594f7fb369b332c',
 'src/dt_groupsearch.py': '2119e0a8a588b68bfbcb88cad95759a27d95a3a7',
 'src/dt_logs.py': 'ebeabd8a5cf762a54547e54bd46a8c10d899c7de',
 'src/dt_mla_arch.py': 'de2fe52f01729e5f77a56d7e53bf464fdd847446',
 'src/dt_mla_arch_llama.py': '481f489058549802812fcd8693a80e494d67f672',
 'src/dt_mla_care.py': 'deb4a6006442ef60321fac9528088931c65a3cb7',
 'src/dt_mla_fit_llama.py': 'e5caef6bb99470360a09618cfd04a3fb0936bc19',
 'src/dt_mla_mha2mla.py': '13dd2a320f286eeca986eb3a6b09df4ddef80b40',
 'src/dt_mla_ours.py': '50e7f4da94bd19416f9c0ef0a103f0c9f380b9fc',
 'src/dt_mla_ours_eig.py': '2ba2d4824f919ded65a0575400470d0d113f15df',
 'src/dt_mla_ours_unrope_cost.py': '8a10c9c152db01cac46063d04f7708eea3a39054',
 'src/dt_mla_ours_unrope_energy.py': 'd44e1fdd3b2df48bfc0268d96064d3fb9a87068d',
 'src/dt_mla_transmla.py': 'ef13fb6a18d34314d77d5f7a27afcf21b6dff903',
 'src/dt_mla_unrope.py': '261e2ba08dc7c4c62f360a07c6828d4d5a0acd5c',
 'src/dt_mla_unrope_arms.py': '2084c568ac54d86ab343f93629f34ebed8cb3e6a',
 'src/dt_mla_unrope_width.py': '767380608dddafb663c3abbbc5738f2fffd47084',
 'src/dt_mla_w16.py': 'b1a7f06c2aefbd6f8a9b2e4d5b2c92414fae834a',
 'src/dt_mla_w16_mha2mla.py': '7ffe337a78ab3cf5d8505ab78e9fd625b0557597',
 'src/dt_mla_w16_ours_unrope_cost.py': 'd9dda39b3576978bb8ea3377a93b56d1d8eaae9d',
 'src/dt_mla_w16_ours_unrope_cost_fi.py': '8b32c3fd037bc72f3248b0273557e91a200ce8ef',
 'src/dt_mla_w16_ours_unrope_cost_ii.py': '73e8b7899b8889e49584754e941910eadc609320',
 'src/dt_mla_w16_ours_unrope_cost_si.py': '9d678e700870ec8c026a1ef599ca77ce9a8e8d68',
 'src/dt_mla_w16_ours_unrope_energy.py': 'd528694581cf25741ea81a1e434f72c4f13882d2',
 'src/dt_mla_w16_transmla.py': '5e835fb38967e10dcbb7d6a4fbdeda11e31eff34',
 'src/dt_mla_w64.py': 'c8701c836571604326a168f2bcc17e56e1c8fdfe',
 'src/dt_mla_w64_care.py': '9a65f0777a2a49ccdb874f134443de69f2ae0e0e',
 'src/dt_mla_w64_mha2mla.py': 'f74c4fcec3bfaff8d71ac807f495f49e59ff0620',
 'src/dt_mla_w64_ours.py': '37290db79cc7b7c0a0d8bd0da73df2e809fdcb8c',
 'src/dt_mla_w64_ours_unrope_cost.py': '357ba144ca238780cf8e9e0c1e48251963d82134',
 'src/dt_mla_w64_ours_unrope_cost_fi.py': 'c4877a57d2e90517d0ce0f242a09e5bf7d486cc5',
 'src/dt_mla_w64_ours_unrope_cost_ii.py': 'e508250ec9f70f71e6acfbe8499876ad518f000a',
 'src/dt_mla_w64_ours_unrope_cost_si.py': '17b94301408a05b8f2e4b987dfd82a182940a111',
 'src/dt_mla_w64_ours_unrope_energy.py': 'e591ed1bad1cd8fcaae57b8a8e925bc795a468a2',
 'src/dt_mla_w64_ours_unrope_tmcopy.py': '010acf94a469dcd1dfc2a5c092c1d62278064479',
 'src/dt_mla_w64_ours_unrope_tmcopy_fi.py': '40dbb618c5995230374b9c682e3880a9acc41b81',
 'src/dt_mla_w64_ours_unrope_tmdamped.py': '5f0768c0318a15ad5a2dfc957a0b3393e0baf2e5',
 'src/dt_mla_w64_transmla.py': '43331dbfa7f15d60a0279ae7c5d7b27226d5ae1d',
 'src/dt_optimal.py': '3dfbd207310ddfdb7d9b5a33856fdaa57b0d7259',
 'src/dt_optimal_arch.py': 'dd426e1ef5a6102cb876ed9ce6b0bf19139e1ff8',
 'src/dt_procrustes.py': '77bbc071685b25c08656bfd7ce7f809647c6243d',
 'src/dt_score.py': 'fc37d92141de17e6031a2eeb096afbf2e26080f9',
 'src/dt_smoltalk.py': 'c90404571d412025002d20e60a8010e9b1fa2c5e',
 'src/dt_util.py': 'e35e1231d6184573fcc30b655ad116c7a76fb27c',
 'src/gam_apply.py': '9902a2a31b558cebfff0f788749976c282209c80',
 'src/gam_core.py': '0b9db693ddd1728cf59da7f3ef3d4380b1e7f9ae',
 'src/gam_core_llama.py': 'ab95ffdf219bbc0d5d76518c47f575857887ebc7',
 'src/gam_record.py': 'c76d0c1ae71769b5ca6500bab7e46fd135e73b2e',
 'src/gam_solve.py': '199240015fcd9c7564ecc07b8f00e77503cc6f41',
 'src/kvlab.py': 'c7a4a436ce4a35e8a5193dca514cdf25e3321a50',
 'src/requirements.txt': '5237af0603076565244bd7595137036d93f7e731'}

_bad = []
for _rel, _want in sorted(MODULE_MANIFEST.items()):
    _r = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", f"HEAD:{_rel}"], capture_output=True, text=True)
    _got = _r.stdout.strip() if _r.returncode == 0 else None
    if _got is None:
        _bad.append(f"  {_rel}: NOT IN THE CLONE (never pushed?)")
    elif _got != _want:
        _bad.append(f"  {_rel}: blob {_got[:12]} in the clone, {_want[:12]} when generated")
_dirty = subprocess.run(["git", "-C", REPO_DIR, "status", "--porcelain", "--", *sorted(MODULE_MANIFEST)],
                        capture_output=True, text=True, check=True).stdout.strip()
if _dirty:
    _bad.append(f"  locally modified in the clone:\n{_dirty}")
if _bad:
    raise RuntimeError(
        f"the clone at {REPO_DIR} (branch {REPO_BRANCH}, {_head}) is not the code this notebook was "
        f"generated against:\n" + "\n".join(_bad) + "\nPush those files as they were at generation, or "
        f"regenerate the notebook (scratch/grid/make_init_nb_b.py). ⛔ Do not edit MODULE_MANIFEST.")
print(f"module check: all {len(MODULE_MANIFEST)} pipeline files are the generated-against blobs ({_head})")

# %%
import subprocess

def _torch_version_cuda():
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
        capture_output=True, text=True, check=True)
    v, c = out.stdout.strip().splitlines()
    return v, c

_before = _torch_version_cuda()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{FEASIBILITY_DIR}/requirements.txt"],
               check=True)
_after = _torch_version_cuda()
assert _before == _after, f"pip install disturbed torch: {_before} -> {_after}"
print(f"torch unchanged: {_after[0]} (cuda {_after[1]})")

# %%
GAUGE_DIR_ON_DRIVE = DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/gauges"
DATA_DIR_ON_DRIVE  = DATA_DIR
os.makedirs(GAUGE_DIR_ON_DRIVE, exist_ok=True)
os.environ["KVTYING_GAUGE_DIR"] = GAUGE_DIR_ON_DRIVE
os.environ["KVTYING_DATA_DIR"]  = DATA_DIR_ON_DRIVE

import dt_arms
import dt_checkpoint
import dt_feasibility_arms
import dt_feasibility_train
import dt_mla_arch

assert dt_arms.GAUGE_DIR == GAUGE_DIR_ON_DRIVE, (
    f"dt_arms read its gauge dir as {dt_arms.GAUGE_DIR!r}, not {GAUGE_DIR_ON_DRIVE!r}. "
    "That means dt_arms was already imported before the env var was set -- restart the "
    "runtime and run this block from the top. ⛔ Do not reassign dt_arms.GAUGE_DIR; the "
    "module has already resolved paths from it.")
print(f"pipeline imported. gauges -> {dt_arms.GAUGE_DIR}")
print(f"arms available: {list(dt_feasibility_arms.ARM_NAMES)}")
print(f"  MLA arms ({dt_mla_arch.MLA_SCHEMA}): {sorted(dt_feasibility_arms.MLA_ARMS)}")
print(f"  default ARMS ({len(dt_feasibility_train.ARMS)} of {len(dt_feasibility_arms.ARM_NAMES)}) "
      f"excludes them: {sorted(set(dt_feasibility_arms.ARM_NAMES) - set(dt_feasibility_train.ARMS))}")

# %%
MODEL_TAG    = 'slm1p6b_urx1_fi'
REGISTRY_SHA = '8079af3365be2b293dfac6f01ec940a33788e3f0ccebe167888a69cc7831f476'
GRID = {'H': 32,
 'Hkv': 32,
 'L': 24,
 'arms_only': ('mla_w64_ours_unrope_cost_fi',),
 'base': 'slm1p6b',
 'd_k': 64,
 'dm': 2048,
 'extra_arms': ('mla_w64_ours_unrope_cost_fi',),
 'family': 'llama_layout',
 'fisher_exclude_top': 0.01,
 'fisher_labels': 'model',
 'hf_id': 'stabilityai/stablelm-2-1_6b',
 'init_budgets': 'trained',
 'init_layout': 'B',
 'init_plan': 'mla_only',
 'kl_width_rule': 'argmin',
 'max_ctx': 4096,
 'model_type': 'stablelm',
 'norm_kind': 'ln_affine',
 'o_bias': False,
 'partial_unrope': True,
 'qkv_bias': True,
 'r': 16,
 'registry_sha': '8079af3365be2b293dfac6f01ec940a33788e3f0ccebe167888a69cc7831f476',
 'registry_version': 'grid-2026-10-04.3',
 'revision': 'f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef',
 'role': 'trained',
 'rope_base': 10000,
 't0_only': True,
 'tag': 'slm1p6b_urx1_fi',
 'teacher': {'dtype': 'float32',
             'hf_id': 'stabilityai/stablelm-2-12b',
             'revision': '74dd3f536b71d18aa122a11af0d1b83105cb2d38',
             'tokenizer_equal': 'functional',
             'tokenizer_oid': 'ab53dcfb233a308e5045eed71cfca31afdf32d74',
             'vocab': 100352},
 'tie_word_embeddings': False,
 'tokenizer_oid': '1143a0b36a2a29335a0387736659842689db19f0',
 'train_plan': 'mla50',
 'vocab': 100352}
GRID_CELLS = [{'arm': 'mla_w64_ours_fisher_sigma',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_transmla_sigma',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_mha2mla_s2norm',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_care_sigma',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_fisher_sigma',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'pool',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'gauge3B',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'procrustes_simk',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optimal',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optwide',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_median',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_energy',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_energy',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_cost',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost_fi',
  'label': '50%',
  'B': 2048,
  'G': 16,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so many '
         'arms"; 50% is not trained; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_fisher_sigma',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_transmla_sigma',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_mha2mla_s2norm',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_care_sigma',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_fisher_sigma',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'pool',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'gauge3B',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'procrustes_simk',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optimal',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optwide',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_median',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_energy',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this '
         'rotary allocation is t = 0 only under D-A1; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_energy',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this '
         'rotary allocation is t = 0 only under D-A1; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_cost',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost_fi',
  'label': '25%',
  'B': 1024,
  'G': 8,
  'star': False,
  'init': 'BUILD',
  'why': '',
  'train': [],
  'train_why': "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)",
  'degenerate': None,
  'rot_rule': 'fixed:64',
  'rot_w': 64},
 {'arm': 'mla_w64_ours_fisher_sigma',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_transmla_sigma',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_mha2mla_s2norm',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_care_sigma',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_fisher_sigma',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'pool',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'gauge3B',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'procrustes_simk',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optimal',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optwide',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_median',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_energy',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this '
         'rotary allocation is t = 0 only under D-A1; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_energy',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this '
         'rotary allocation is t = 0 only under D-A1; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_cost',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost_fi',
  'label': '12.5%',
  'B': 512,
  'G': 4,
  'star': False,
  'init': 'BUILD',
  'why': '',
  'train': [],
  'train_why': "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)",
  'degenerate': None,
  'rot_rule': 'fixed:64',
  'rot_w': 64},
 {'arm': 'mla_w64_ours_fisher_sigma',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_transmla_sigma',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_mha2mla_s2norm',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_care_sigma',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_fisher_sigma',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops the '
         'rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'pool',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'gauge3B',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'procrustes_simk',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optimal',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'optwide',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'decoupled_optwide_median',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:MLA_ONLY',
  'why': 'MLA_ONLY decided 2026-10-08 (the authors): "we\'re not running the other arms, just the MLA arms"; '
         'would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_energy',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this '
         'rotary allocation is t = 0 only under D-A1; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_energy',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:TRAINED_ONLY',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this '
         'rotary allocation is t = 0 only under D-A1; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_ours_unrope_cost',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'OMITTED:ARMS_ONLY',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'mla_w64_ours_unrope_cost_fi; would be BUILD',
  'train': [],
  'train_why': 'not built',
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None},
 {'arm': 'mla_w64_ours_unrope_cost_fi',
  'label': '6.25%',
  'B': 256,
  'G': 2,
  'star': False,
  'init': 'BUILD',
  'why': '',
  'train': [],
  'train_why': "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)",
  'degenerate': None,
  'rot_rule': 'fixed:64',
  'rot_w': 64},
 {'arm': 'mha',
  'label': 'src',
  'B': 4096,
  'G': 32,
  'star': False,
  'init': 'BUILD',
  'why': 'the unconverted model (no checkpoint: built in the run cell)',
  'train': [],
  'train_why': 'NOT_TRAINED: MLA50 decided 2026-10-06 (the authors): only p410m trains the control; its t = 0 '
               "is the run's reference (KL of the teacher to the original)",
  'degenerate': None,
  'rot_rule': None,
  'rot_w': None}]

import shutil
import dt_grid_models
if dt_grid_models.registry_sha(MODEL_TAG) != REGISTRY_SHA:
    raise RuntimeError(f"the cloned registry gives {MODEL_TAG} sha {dt_grid_models.registry_sha(MODEL_TAG)[:16]}, "
                       f"this notebook was generated at {REGISTRY_SHA[:16]}. Regenerate the notebook.")
_live = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}
         for c in dt_grid_models.cells(MODEL_TAG)]
if _live != GRID_CELLS:
    raise RuntimeError("the cloned registry's cell table differs from the one embedded here")
dt_feasibility_train.use_model(MODEL_TAG)
if dt_feasibility_train.MODEL != GRID["hf_id"]:
    raise RuntimeError(f"dt_feasibility_train.MODEL is {dt_feasibility_train.MODEL}, the registry says {GRID['hf_id']}")
from huggingface_hub import model_info as _model_info
_main = _model_info(GRID["hf_id"]).sha
if _main != GRID["revision"]:
    raise RuntimeError(f"{GRID['hf_id']}'s main is now {_main}, not the registry's pinned "
                       f"{GRID['revision']}. Every load in the pipeline reads main, so the inits would be "
                       f"built from different weights. Put this to the authors; do not re-pin silently.")

import torch as _torch
from transformers import AutoConfig as _AutoConfig, AutoModelForCausalLM as _AutoModel
_cfg = _AutoConfig.from_pretrained(GRID["hf_id"], revision=GRID["revision"])
if GRID["Hkv"] != GRID["H"]:
    _cfg.num_key_value_heads = _cfg.num_attention_heads
with _torch.device("meta"):
    _meta_model = _AutoModel.from_config(_cfg)
N_PARAMS_EXPECTED = sum(_p.numel() for _p in _meta_model.parameters())
del _meta_model
_ckpt_bytes = 4 * N_PARAMS_EXPECTED
N_INIT_DIRS = 3
_need_drive = int(1.05 * N_INIT_DIRS * _ckpt_bytes)
_ram_min = int(1.9 * _ckpt_bytes)
import psutil as _psutil
_ram = _psutil.virtual_memory().total
_du = shutil.disk_usage(DRIVE_ROOT + "/MyDrive")
print(f"{MODEL_TAG}: {GRID['hf_id']} @ {GRID['revision'][:12]} (= main), registry {REGISTRY_SHA[:16]}, "
      f"{len(GRID_CELLS)} cells")
print(f"preflight: {N_PARAMS_EXPECTED / 1e9:.2f} B parameters = {_ckpt_bytes / 1e9:.1f} GB per fp32 init; "
      f"{N_INIT_DIRS} inits need {_need_drive / 1e9:.0f} GB of Drive (free {_du.free / 1e9:.0f} GB of "
      f"{_du.total / 1e9:.0f}); host RAM {_ram / 1e9:.0f} GB (minimum {_ram_min / 1e9:.0f})")
if _ram < _ram_min:
    raise RuntimeError(f"host RAM {_ram / 1e9:.0f} GB is below the {_ram_min / 1e9:.0f} GB this model's builds "
                       f"need. Switch to a larger runtime; never run a cut-down grid.")
if _du.free < _need_drive:
    raise RuntimeError(f"Drive has {_du.free / 1e9:.0f} GB free; keeping every init (the authors, 2026-10-04) needs "
                       f"{_need_drive / 1e9:.0f} GB. Free the space first; there is no local fallback.")

# %%
import hashlib
import os

import numpy as np
import torch
from datasets import load_dataset

import gam_core as gc
import kvlab

FISHER_N_DOC = 64
FISHER_SEQ_LEN = 512
FISHER_SEED_EVAL = 20260912
FISHER_DOC_OFFSET = 2000
FISHER_LABEL_SEED = 20261014
FISHER_EXCLUDE_TOP = 0.01
FISHER_EXCLUDED = []
FISHER_DIR = DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/fisher"
os.makedirs(FISHER_DIR, exist_ok=True)

def _pile_split(tok, n_doc, skip):
    ev = load_dataset("NeelNanda/pile-10k", split="train")
    rng = np.random.default_rng(FISHER_SEED_EVAL)
    ids, pick, taken = [], [], 0
    for j in rng.permutation(len(ev)):
        if j < FISHER_DOC_OFFSET:
            continue
        e = tok(ev[int(j)]["text"])["input_ids"]
        if len(e) < FISHER_SEQ_LEN + 1:
            continue
        taken += 1
        if taken <= skip:
            continue
        ids.append(e[:FISHER_SEQ_LEN])
        pick.append(int(j))
        if len(ids) == n_doc:
            break
    if len(ids) != n_doc:
        raise RuntimeError(f"only {len(ids)} documents were long enough; need {n_doc}")
    h = {hashlib.sha256(ev[int(j)]["text"].encode()).hexdigest() for j in pick}
    return torch.tensor(ids, dtype=torch.long), pick, h

def fit_fisher(model, ids):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    layers = kvlab.get_layers(model)
    GK = np.zeros((L, H, d_k, d_k), dtype=np.float64)
    GV = np.zeros((L, H, d_v, d_v), dtype=np.float64)
    grabbed = {}

    def _hook(li, slot):
        def fn(_m, _gi, go):
            g = go[0].detach().reshape(go[0].shape[0], go[0].shape[1], H, -1).double()
            grabbed.setdefault(li, [None, None])[slot] = g.clone()
        return fn

    handles = [p.register_full_backward_hook(_hook(li, s))
               for li, lay in enumerate(layers)
               for s, p in enumerate((lay.self_attn.k_proj, lay.self_attn.v_proj))]
    handles.append(model.get_input_embeddings().register_forward_hook(
        lambda _m, _i, o: o.requires_grad_(True)))
    model.eval()
    n_tok = 0
    try:
        for d in range(ids.shape[0]):
            chunk = ids[d:d + 1]
            model.zero_grad(set_to_none=True)
            _lg = model(chunk).logits[:, :-1].float()
            with torch.no_grad():
                _v = torch.multinomial(torch.softmax(_lg.reshape(-1, _lg.shape[-1]), dim=-1), 1,
                                       generator=torch.Generator().manual_seed(FISHER_LABEL_SEED + d))
            torch.nn.functional.cross_entropy(_lg.reshape(-1, _lg.shape[-1]), _v.view(-1)).backward()
            if len(grabbed) != len(layers) or any(x is None for v in grabbed.values() for x in v):
                raise RuntimeError(f"only {len(grabbed)}/{len(layers)} layers' k/v hooks fired")
            _nx = int(round(FISHER_EXCLUDE_TOP * chunk.shape[1]))
            FISHER_EXCLUDED.append([])
            for li in range(L):
                gk, gv = grabbed[li]
                _sc = (gk ** 2).sum((2, 3))[0] + (gv ** 2).sum((2, 3))[0]
                _top = torch.topk(_sc, _nx).indices
                _keep = torch.ones_like(_sc)
                _keep[_top] = 0.0
                FISHER_EXCLUDED[-1].append(sorted(_top.tolist()))
                gk, gv = gk * _keep[None, :, None, None], gv * _keep[None, :, None, None]
                GK[li] += np.einsum("bthi,bthj->hij", gk.numpy(), gk.numpy())
                GV[li] += np.einsum("bthi,bthj->hij", gv.numpy(), gv.numpy())
            n_tok += chunk.shape[1] - _nx
            grabbed.clear()
            if (d + 1) % 16 == 0:
                print(f"    {d + 1}/{ids.shape[0]} docs")
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
    return GK / n_tok, GV / n_tok

_fm, _ftok = kvlab.load_model(dt_feasibility_train.MODEL)
_, _epick, _ehash = _pile_split(_ftok, FISHER_N_DOC, skip=0)
_fids, _fpick, _fhash = _pile_split(_ftok, FISHER_N_DOC, skip=FISHER_N_DOC)
if _ehash & _fhash or set(_epick) & set(_fpick):
    raise RuntimeError("the Fisher fit corpus overlaps the eval corpus; the control is void")
print(f"fit corpus: {FISHER_N_DOC} docs, picks {_fpick[:4]} ... -- disjoint from eval "
      f"({len(_ehash & _fhash)} shared hashes, {len(set(_epick) & set(_fpick))} shared indices)")

FISHER_CACHE = os.path.join(FISHER_DIR, f"fisher_{dt_feasibility_train.MODEL.split('/')[-1]}_"
                                        f"s{FISHER_SEED_EVAL}_n{FISHER_N_DOC}"
                                        f"_L{FISHER_SEQ_LEN}_o{FISHER_DOC_OFFSET}"
                                        f"_truefisher_r{FISHER_LABEL_SEED}_x{FISHER_EXCLUDE_TOP:g}"
                                        f"_heldout.npz")
if os.path.exists(FISHER_CACHE):
    _z = np.load(FISHER_CACHE)
    _GK, _GV = _z["GK"], _z["GV"]
    _cached_e = set(_z["eval_sha256"].tolist())
    if _cached_e != _ehash:
        raise RuntimeError(
            f"the cached Fisher at {FISHER_CACHE} was fitted against a different corpus than the "
            f"one loaded now: {len(_cached_e - _ehash)} of its eval documents are absent from this "
            f"run's eval split, and {len(_ehash - _cached_e)} of this run's are absent from its. "
            f"Every constant in the filename matches, so the dataset itself has changed underneath "
            f"the key. Delete the cache and refit -- reusing it would build every Fisher arm "
            f"against a metric fitted on text this run does not contain.")
    print(f"loaded cached Fisher <- {FISHER_CACHE}")
else:
    print(f"fitting the TRUE Fisher (next tokens drawn from the model, @@TRUEF@@), each layer without each document's top {FISHER_EXCLUDE_TOP:.0%} tokens by gradient (@@TRUEX@@), on {FISHER_N_DOC} x {FISHER_SEQ_LEN} = "
          f"{FISHER_N_DOC * FISHER_SEQ_LEN} tokens (CPU, ~5 min) ...")
    _GK, _GV = fit_fisher(_fm, _fids)
    np.savez_compressed(FISHER_CACHE, GK=_GK, GV=_GV,
                        excluded_pos=np.array(FISHER_EXCLUDED, dtype=np.int16),
                        eval_sha256=np.array(sorted(_ehash)),
                        fit_sha256=np.array(sorted(_fhash)))
    print(f"saved -> {FISHER_CACHE}")
del _fm

FISHER = (_GK, _GV)
FISHER_EVAL_HASHES = _ehash
print(f"FISHER ready: GK {_GK.shape}, GV {_GV.shape}, "
      f"max|GK| {np.abs(_GK).max():.3e}, max|GV| {np.abs(_GV).max():.3e}")
print(f"  eval-split record stored in the cache ({len(_ehash)} sha256) -- the corpus cell "
      f"re-reads it from there, so the check survives the CPU -> GPU runtime switch")

# %%
import hashlib
import os

import numpy as np
import torch

import gam_core as gc
import kvlab

for _need in ("_pile_split", "FISHER_N_DOC", "FISHER_SEQ_LEN", "FISHER_SEED_EVAL",
              "FISHER_DOC_OFFSET", "_ehash", "_fhash", "_fids"):
    if _need not in globals():
        raise RuntimeError(
            f"{_need} is not defined -- run the @@FISHER@@ cell above first. Sigma is fitted on "
            f"exactly the disjoint split that cell built, by calling its own code, so that the two "
            f"metrics cannot end up fitted on different corpora while both filenames look correct.")

SIGMA_DIR = DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/sigma"
os.makedirs(SIGMA_DIR, exist_ok=True)

def verify_sigma_input_space(model, tol=1e-6):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    grab = {}
    h1 = layers[0].input_layernorm.register_forward_hook(
        lambda _m, _i, o: grab.__setitem__("ln", o.detach().double()))
    h2 = layers[0].self_attn.k_proj.register_forward_hook(
        lambda _m, _i, o: grab.__setitem__("qkv", o.detach().double()))
    try:
        with torch.no_grad():
            model(torch.arange(16, dtype=torch.long).reshape(1, 16))
    finally:
        h1.remove()
        h2.remove()
    if len(grab) != 2:
        raise RuntimeError("the verification hooks did not both fire")
    x = grab["ln"].reshape(-1, dm).numpy()
    xa = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
    K_ref = grab["qkv"].reshape(-1, H, d_k).numpy()
    _, Kt, _, _ = gc.read_layer(layers[0], H, d_k, d_v, dm)
    dev = max(float(np.abs(xa @ Kt[i].T - K_ref[:, i, :]).max()) for i in range(H))
    scale = float(np.abs(K_ref).max())
    if dev > tol * max(scale, 1.0):
        raise RuntimeError(
            f"the hooked activation does NOT reproduce the module's keys (max dev {dev:.3e}, key "
            f"scale {scale:.3e}). Sigma would be the second moment of a different input space than "
            f"gc.read_layer's maps consume. Check that the hook is on input_layernorm and that the "
            f"constant column is appended LAST.")
    return dev, scale

def fit_sigma(model, ids, bs=2):
    layers = kvlab.get_layers(model)
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    n = dm + 1
    SIG = [np.zeros((n, n), dtype=np.float64) for _ in range(L)]
    ntok = [0]

    def _hook(li):
        def fn(_m, _i, o):
            x = o.detach().double().reshape(-1, dm).numpy()
            xa = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
            SIG[li] += xa.T @ xa
            if li == 0:
                ntok[0] += xa.shape[0]
        return fn

    handles = [lay.input_layernorm.register_forward_hook(_hook(li))
               for li, lay in enumerate(layers)]
    model.eval()
    try:
        with torch.no_grad():
            for d in range(0, ids.shape[0], bs):
                model(ids[d:d + bs])
                if (d + bs) % 16 == 0:
                    print(f"    {min(d + bs, ids.shape[0])}/{ids.shape[0]} docs", flush=True)
    finally:
        for h in handles:
            h.remove()
    if ntok[0] == 0:
        raise RuntimeError("no tokens were seen -- the layer-0 hook never fired")
    for li in range(L):
        SIG[li] /= float(ntok[0])
    c11 = SIG[0][-1, -1]
    if abs(c11 - 1.0) > 1e-9:
        raise RuntimeError(
            f"Sigma[-1, -1] = {c11:.6g}, must be exactly 1.0. That entry is E[1*1] for the "
            f"augmented constant coordinate, so the constant column was not appended and this is "
            f"the second moment of an un-augmented input space.")
    return SIG, ntok[0]

def sigma_shape_report(SIG):
    off, rel = [], []
    for S in SIG:
        S = (S + S.T) / 2.0
        fro = np.linalg.norm(S, "fro")
        c = np.trace(S) / S.shape[0]
        off.append(np.linalg.norm(S - np.diag(np.diag(S)), "fro") / fro)
        rel.append(np.linalg.norm(S - c * np.eye(S.shape[0]), "fro") / fro)
    return np.array(off), np.array(rel)

SIGMA_CACHE = os.path.join(SIGMA_DIR, f"sigma_{dt_feasibility_train.MODEL.split('/')[-1]}_"
                                      f"s{FISHER_SEED_EVAL}_n{FISHER_N_DOC}"
                                      f"_L{FISHER_SEQ_LEN}_o{FISHER_DOC_OFFSET}_heldout.npz")
if os.path.exists(SIGMA_CACHE):
    _z = np.load(SIGMA_CACHE)
    _SIG = [np.ascontiguousarray(s) for s in _z["SIG"]]
    _cached_e = set(_z["eval_sha256"].tolist())
    if _cached_e != _ehash:
        raise RuntimeError(
            f"the cached Sigma at {SIGMA_CACHE} was fitted against a different corpus than the one "
            f"loaded now: {len(_cached_e - _ehash)} of its eval documents are absent from this "
            f"run's eval split, and {len(_ehash - _cached_e)} of this run's are absent from its. "
            f"Every constant in the filename matches, so the dataset changed underneath the key. "
            f"Delete the cache and refit.")
    print(f"loaded cached Sigma <- {SIGMA_CACHE}")
    _sm, _ = kvlab.load_model(dt_feasibility_train.MODEL)
    _dev, _scale = verify_sigma_input_space(_sm)
    del _sm
else:
    _sm, _ = kvlab.load_model(dt_feasibility_train.MODEL)
    _sm.config.use_cache = False
    _dev, _scale = verify_sigma_input_space(_sm)
    print(f"input space verified: [LN(x), 1] rebuilds layer 0's keys to {_dev:.2e} "
          f"(key scale {_scale:.2e})")
    print(f"fitting Sigma on the DISJOINT split, {FISHER_N_DOC} x {FISHER_SEQ_LEN} = "
          f"{FISHER_N_DOC * FISHER_SEQ_LEN} tokens (CPU, ~3 min) ...")
    _SIG, _sntok = fit_sigma(_sm, _fids)
    np.savez_compressed(SIGMA_CACHE, SIG=np.stack(_SIG),
                        eval_sha256=np.array(sorted(_ehash)),
                        fit_sha256=np.array(sorted(_fhash)))
    print(f"saved -> {SIGMA_CACHE}  ({os.path.getsize(SIGMA_CACHE) / 1e6:.0f} MB)")
    del _sm

SIGMA = _SIG
SIGMA_EVAL_HASHES = _ehash
_off, _rel = sigma_shape_report(SIGMA)
print(f"SIGMA ready: {len(SIGMA)} layers x {SIGMA[0].shape}, "
      f"Sigma[-1,-1] = {SIGMA[0][-1, -1]:.9f}")
print(f"  off-diagonal mass   median {np.median(_off):.4f}  range [{_off.min():.4f}, {_off.max():.4f}]")
print(f"  rel. |Sigma - c*I|  median {np.median(_rel):.4f}  range [{_rel.min():.4f}, {_rel.max():.4f}]")
print(f"  (both near 0 would mean Sigma IS a multiple of I and this cell is dead weight -- "
      f"they are not, and the ~{100 * (1 - np.median(_off)):.0f}% that is diagonal is all the "
      f"gamma^2 proxy ever modelled)")
print(f"  eval-split record stored in the cache ({len(_ehash)} sha256) -- the corpus cell re-reads "
      f"it from there, so the check survives the CPU -> GPU runtime switch")

# %%
import hashlib
import numpy as np
import gam_core as gc
import kvlab
import dt_decoupled
import dt_optimal

def _spectra(model, w=1.0, GK=None, GV=None, SIGMA=None):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0:
        raise ValueError("no rotary rows: there is no rotary width to choose")
    theta, wD = gc.inv_freq(model), dt_optimal._offset_weights(dt_optimal.T_EVAL)
    rot, kv = [], []
    for li, lay in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        A_free, A_V, A_rot = dt_decoupled._metrics(Qt, Wo, H, r, theta, wD,
                                                   None if GK is None else GK[li],
                                                   None if GV is None else GV[li])
        S_h = None if SIGMA is None else dt_decoupled._sigma_pair(SIGMA[li], dm + 1)[0]
        for stack, out in ((dt_decoupled._rot_stack(Kt, A_rot, H, r), rot),
                           (dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H, r, w), kv)):
            if S_h is not None:
                stack = stack @ S_h
            s = np.linalg.svd(stack, compute_uv=False)
            out.append(np.sort(s ** 2)[::-1])
    return rot, kv, (L, H, d_k, d_v, dm, r)

def _crossing(lam_rot, lam_kv, B):
    def _at(lam, idx1):
        return lam[idx1 - 1] if 1 <= idx1 <= len(lam) else 0.0
    d = 0
    while d < B and _at(lam_rot, d + 1) >= _at(lam_kv, B - d):
        d += 1
    return d

def compute_dstar(model, group_sizes, w=1.0, GK=None, GV=None, SIGMA=None):
    rot, kv, (L, H, d_k, d_v, dm, r) = _spectra(model, w, GK, GV, SIGMA)
    out = {}
    for G in group_sizes:
        B = G
        by_layer = [_crossing(rot[li], kv[li], B) for li in range(L)]
        if max(by_layer) > H * r:
            raise AssertionError(f"d*={max(by_layer)} exceeds rank Xi_rot <= H*r = {H * r}")
        raw = float(np.median(by_layer))
        out[G] = {"by_layer": by_layer, "median": int(round(raw)), "median_raw": raw}
    return out

def _mla_layer(Kt, Vt, Qt, Wo, GK_l, GV_l, SIGMA_l, H, r, dm, theta, wD):
    P = r // 2
    Kt = np.asarray(Kt)
    A_free, A_V, _ = dt_decoupled._metrics(Qt, Wo, H, r, theta, wD, GK_l, GV_l)
    S_h = dt_decoupled._sigma_pair(SIGMA_l, dm + 1)[0]
    s = np.linalg.svd(dt_decoupled._joint_stack(Kt, Vt, A_free, A_V, H, r, 1.0) @ S_h,
                      compute_uv=False)
    lam = np.sort(s ** 2)[::-1]
    G_l = np.asarray(GK_l, dtype=np.float64)
    dg = np.diagonal(G_l, axis1=1, axis2=2)
    cz = np.stack([gc.to_complex(dg[i][:, None], r)[:, 0] for i in range(H)])
    varpi = 0.5 * (cz.real + cz.imag)
    if (varpi < 0).any():
        raise ValueError("a negative Fisher diagonal -- GK is not a Fisher")
    Ks = np.stack([gc.to_complex(Kt[i], r) for i in range(H)]) @ S_h
    Vh, mu_p = [], []
    for p in range(P):
        _, sp, vh = np.linalg.svd(np.sqrt(varpi[:, p])[:, None] * Ks[:, p, :], full_matrices=False)
        Vh.append(vh)
        mu_p.append(sp ** 2)
    flat = np.concatenate(mu_p)
    order = np.argsort(-flat, kind="stable")
    mu = flat[order]
    KS = Kt[:, :r, :] @ S_h
    e_rot = (KS ** 2).sum(-1)
    varpi_x = 0.5 * (dg[:, :P] + dg[:, P:r])
    want = float((varpi_x * (e_rot[:, :P] + e_rot[:, P:r])).sum())
    if abs(float(mu.sum()) - want) > 1e-9 * max(want, 1e-300):
        raise AssertionError(f"sum(mu) = {mu.sum():.9e} but the (p, p+r/2)-paired rotary energy is "
                             f"{want:.9e} -- gc.to_complex pairs the rotary rows differently")
    F = G_l[:, :r, :r]
    Rc = Ks.copy()
    nxt = [0] * P
    D = np.empty(len(order) + 1)
    for m in range(len(order) + 1):
        if m:
            p = int(order[m - 1]) // H
            v = Vh[p][nxt[p]]
            nxt[p] += 1
            Rc[:, p, :] -= np.outer(Rc[:, p, :] @ v.conj(), v)
        R = np.concatenate([Rc.real, Rc.imag], axis=1)
        D[m] = float(np.einsum("hjk,hjk->", F, R @ R.transpose(0, 2, 1)))
    if D[-1] > 1e-9 * max(D[0], 1e-300):
        raise AssertionError(f"all {len(order)} directions leave rotary damage {D[-1]:.3e} of {D[0]:.3e}")
    ratio = want / D[0] if D[0] > 0 else float("nan")
    return lam, mu, D, ratio

def _mla_proxy_crossing(mu, lam, B, cap):
    def _at(a, k):
        return a[k - 1] if 1 <= k <= len(a) else 0.0
    m = 0
    while (2 * (m + 1) <= B and m + 1 <= cap
           and _at(mu, m + 1) >= _at(lam, B - 2 * m - 1) + _at(lam, B - 2 * m)):
        m += 1
    return m

def compute_mla_width(model, group_sizes, GK, GV, SIGMA):
    L, H, d_k, d_v, dm, r = gc.geometry(model)
    if r == 0 or r % 2:
        raise ValueError(f"the MLA width needs an even rotary width; this model has r = {r}")
    if L % 2:
        raise ValueError(f"the half-up median rule assumes an even layer count; L = {L}")
    theta, wD = gc.inv_freq(model), dt_optimal._offset_weights(dt_optimal.T_EVAL)
    per = []
    for li, lay in enumerate(kvlab.get_layers(model)):
        Qt, Kt, Vt, Wo = gc.read_layer(lay, H, d_k, d_v, dm)
        per.append(_mla_layer(Kt, Vt, Qt, Wo, GK[li], GV[li], SIGMA[li], H, r, dm, theta, wD))
    cap = H * (r // 2)
    by_G = {}
    for G in group_sizes:
        B = G
        m_by, m_px, mg_in, mg_out, ru_m, ru_rel = [], [], [], [], [], []
        for lam, mu, D, _ in per:
            ms = np.arange(0, min(cap, B // 2) + 1)
            tot = D[ms] + np.array([lam[B - 2 * m:].sum() for m in ms])
            k = int(np.flatnonzero(tot == tot.min())[-1])
            m_by.append(int(ms[k]))
            mg_in.append(float(tot[k - 1] - tot[k]) if k > 0 else None)
            mg_out.append(float(tot[k + 1] - tot[k]) if k + 1 < len(ms) else None)
            rest = np.delete(np.arange(len(ms)), k)
            if len(rest):
                j = int(rest[np.argmin(tot[rest])])
                ru_m.append(int(ms[j]))
                ru_rel.append(float((tot[j] - tot[k]) / tot[k]) if tot[k] > 0 else None)
            else:
                ru_m.append(None)
                ru_rel.append(None)
            m_px.append(_mla_proxy_crossing(mu, lam, B, cap))
        srt, spx = sorted(m_by), sorted(m_px)
        lo, hi = srt[L // 2 - 1], srt[L // 2]
        m_med = (lo + hi + 1) // 2
        blocked = m_med == 0 or 2 * m_med >= B
        by_G[G] = {"B": B, "m_by_layer": m_by, "m_median_raw": (lo + hi) / 2, "m": m_med,
                   "half_tie": (lo + hi) % 2 == 1, "m_half_even": int(round((lo + hi) / 2)),
                   "w": None if blocked else 2 * m_med, "d_c": None if blocked else B - 2 * m_med,
                   "blocked": blocked, "at_cap": [m == cap for m in m_by],
                   "at_budget": [m == B // 2 for m in m_by],
                   "margin_in": mg_in, "margin_out": mg_out,
                   "runner_up_m": ru_m, "runner_up_rel_gap": ru_rel,
                   "proxy_m_by_layer": m_px,
                   "proxy_w": 2 * ((spx[L // 2 - 1] + spx[L // 2] + 1) // 2),
                   "crosscheck_fisher_sigma_median": DSTAR["fisher_sigma"][G]["median"]}
    return by_G, [float(x[3]) for x in per]

DSTAR_BUDGETS = [256, 512, 1024]
KLW_BUDGETS = []
_base, _ = kvlab.load_model(dt_feasibility_train.MODEL)
_L = gc.geometry(_base)[0]
DSTAR = {"budget_key": "B", "model_tag": MODEL_TAG, "base_model": dt_feasibility_train.MODEL,
         "base_fingerprint": dt_checkpoint._FINGERPRINT["mha"](_base), "w": 1.0,
         "guessed": compute_dstar(_base, DSTAR_BUDGETS, w=1.0),
         "fisher_sigma": compute_dstar(_base, DSTAR_BUDGETS, w=1.0, GK=FISHER[0], GV=FISHER[1],
                                       SIGMA=SIGMA)}
for _B in DSTAR_BUDGETS:
    _d, _s = DSTAR["guessed"][_B]["by_layer"], DSTAR["fisher_sigma"][_B]["by_layer"]
    print(f"B={_B:>5}   guessed d* median {DSTAR['guessed'][_B]['median']:>5} [{min(_d)}, {max(_d)}]   "
          f"Fisher+Sigma d* median {DSTAR['fisher_sigma'][_B]['median']:>5} [{min(_s)}, {max(_s)}]")
if dt_checkpoint._FINGERPRINT["mha"](_base) != DSTAR["base_fingerprint"]:
    raise RuntimeError("the base fingerprint moved between two reads of one model")
_mla_by_B, _mla_ratio = compute_mla_width(_base, DSTAR_BUDGETS, FISHER[0], FISHER[1], SIGMA)
del _base
DSTAR["mla"] = {"schema": "mla-width/v3",
                "rule": 'm* = argmin_m D_rot_fullFisher(m) + lam-tail(B-2m) (largest m on a tie), per layer, direction order by mu; w = 2m',
                "rounding": 'median over layers of m*, half up in integers',
                "base_fingerprint": DSTAR["base_fingerprint"],
                "fisher_sha256": dt_feasibility_train.fisher_digest(FISHER),
                "sigma_sha256": dt_feasibility_train.sigma_digest(SIGMA),
                "crosscheck_ok": None,
                "crosscheck_note": f"{MODEL_TAG} has no registry regression block: no built reference",
                "model_tag": MODEL_TAG,
                "proj_trace_ratio_by_layer": _mla_ratio,
                "by_B": _mla_by_B}
print(f"\n  the MLA KL-objective width (w = 2m, d_c = B - w), per budget:")
for _B in DSTAR_BUDGETS:
    _r = _mla_by_B[_B]
    _flag = "  <- BUILT (a KL-width cell)" if _B in KLW_BUDGETS else ""
    print(f"    B={_B:>5}   m* [{min(_r['m_by_layer'])}, {max(_r['m_by_layer'])}]   m {_r['m']:>4}   "
          f"w {_r['w']}   d_c {_r['d_c']}   blocked {_r['blocked']}{_flag}")
print(f"  fisher {DSTAR['mla']['fisher_sha256']}   sigma {DSTAR['mla']['sigma_sha256']}   "
      f"base {DSTAR['base_fingerprint']}")

# %%
assert CKPT_ROOT == DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/init/ckpt", (
    f"CKPT_ROOT is {CKPT_ROOT!r}. slm1p6b_urx1_fi's inits live in /MyDrive/gaugemla/grid/slm1p6b_urx1_fi/init/ckpt and its t = 0 "
    "notebook reads that path; a branch written anywhere else is invisible to it.")

INIT_CELLS = [['mla_w64_ours_unrope_cost_fi', 8, 1024],
              ['mla_w64_ours_unrope_cost_fi', 4, 512],
              ['mla_w64_ours_unrope_cost_fi', 2, 256]]
INIT_ARMS = list(dict.fromkeys(a for a, _, _ in INIT_CELLS))
_reg_cells = sorted([c["arm"], c["G"] if c["G"] is not None else -1, c["B"]] for c in GRID_CELLS
                    if c["arm"] != "mha" and (c["init"] == "BUILD" or c["init"].startswith("DEDUP:")))
if sorted([a, -1 if g is None else g, b] for a, g, b in INIT_CELLS) != _reg_cells:
    raise RuntimeError("INIT_CELLS is not the registry's BUILD + DEDUP cells")
INIT_ONLY = [a for a in os.environ.get("INIT_ARMS", "").split(",") if a]
if INIT_ONLY:
    if sorted(set(INIT_ONLY) - set(INIT_ARMS)):
        raise RuntimeError(f"INIT_ARMS names {sorted(set(INIT_ONLY) - set(INIT_ARMS))}, which this grid does not "
                           f"build: {INIT_ARMS}")
    INIT_CELLS = [c for c in INIT_CELLS if c[0] in INIT_ONLY]
    INIT_ARMS = [a for a in INIT_ARMS if a in INIT_ONLY]
    print(f"INIT_ARMS: building only {INIT_ARMS} ({len(INIT_CELLS)} cells); the registry's other cells stay unbuilt")

DRY_RUN = {"True": True, "False": False}.get(os.environ.get("DRY_RUN", ""))
if DRY_RUN is None:
    raise RuntimeError(f"DRY_RUN must be exactly True or False, got {os.environ.get('DRY_RUN')!r}")

_sigma_wanted = sorted({a for a in INIT_ARMS if "_sigma" in a or a.startswith("mla_")})
if _sigma_wanted and "SIGMA" not in globals():
    raise RuntimeError(
        f"{_sigma_wanted} are in INIT_ARMS but SIGMA is not defined. Run the @@SIGMA@@ cell above "
        "(it fits Sigma on the same disjoint split the Fisher uses and caches it to Drive). "
        "Without it the decoupled_*_sigma arms ARE decoupled_optwide_fisher under other names, "
        "and an mla_* arm cannot be built at all -- its width table is keyed on this Sigma's hash.")

_unknown = [a for a in INIT_ARMS if a not in dt_feasibility_arms.ARM_NAMES]
if _unknown:
    raise RuntimeError(
        f"unknown arm(s) {_unknown}. Known: {list(dt_feasibility_arms.ARM_NAMES)}. "
        "A name that is not in ARM_NAMES would be dispatched to nothing and the branch "
        "would simply never appear -- check the spelling against the list above.")

_mla_wanted = sorted(a for a in INIT_ARMS if a.startswith("mla_"))
if _mla_wanted:
    if "DSTAR" not in globals() or "mla" not in DSTAR:
        raise RuntimeError(
            f"{_mla_wanted} are in INIT_ARMS but DSTAR['mla'] is not defined. Run the d* cell above to "
            f"the END -- its @@MLA@@ block computes the MLA rotary width table (w and d_c per G) and "
            f"prints it beside the cross-check. DSTAR keys present: "
            f"{sorted(globals().get('DSTAR') or {})}.")
    if DSTAR.get("budget_key") != "B" or DSTAR["mla"].get("model_tag") != MODEL_TAG:
        raise RuntimeError("DSTAR is not this model's budget-keyed table: run the width cell above")
    if not (DSTAR["mla"].get("crosscheck_ok") is True or (DSTAR["mla"].get("crosscheck_ok") is None
                                                         and "regression" not in GRID)):
        raise RuntimeError(
            "DSTAR['mla']['crosscheck_ok'] is False. The d* cell's own cross-check -- this session's "
            "Fisher+Sigma crossing against the BUILT decoupled_optwide_fisher_sigma_dstar arm -- did "
            "not agree, so the FISHER/SIGMA in this runtime are not demonstrably the arrays the rest "
            "of the grid was built from, and a width chosen from them would not be comparable to it. "
            "_mla_widths refuses the table too; this says so before the model loads. Read that cell's "
            "⛔ lines rather than overriding this.")
    _mla_blocked = sorted(B for B in KLW_BUDGETS
                          if DSTAR["mla"]["by_B"].get(B, {"blocked": True})["blocked"])
    if _mla_blocked:
        raise RuntimeError(
            f"DSTAR['mla'] has no usable row for the KL-width cells at B = {_mla_blocked} (blocked, or absent from the "
            f"table -- D4). Either the per-layer median m* came out 0, or the whole budget went to "
            f"the rotary key; w = 0 is outside all four methods and TransMLA's own code asserts "
            f"against it. Those cells are the registry's: put the m >= 1 floor (or recording them as not "
            f"representable) to the authors -- it is "
            f"never applied automatically.")

S2NORM = None
_s2_wanted = sorted(a for a in INIT_ARMS if a in dt_feasibility_train.S2NORM_ARMS)
if _s2_wanted:
    import glob as _glob
    import dt_mla_mha2mla
    import kvlab as _kvlab

    for _need in ("_fids", "_fhash", "_ehash"):
        if _need not in globals():
            raise RuntimeError(
                f"{_s2_wanted} are in INIT_ARMS but {_need} is not defined -- run the @@FISHER@@ cell "
                f"above first. S2 is fitted on exactly the disjoint split that cell built, by using "
                f"its own ids, so the three measured objects in this notebook cannot end up fitted on "
                f"different corpora while every filename still looks right.")

    S2NORM_DIR = DRIVE_ROOT + "/MyDrive/gaugemla/grid/slm1p6b_urx1_fi/s2norm"
    os.makedirs(S2NORM_DIR, exist_ok=True)
    _s2_glob = os.path.join(S2NORM_DIR, f"s2norm_*_fit_{dt_mla_mha2mla.ids_sha256(_fids)[:16]}.npz")
    _s2_cands = sorted(_glob.glob(_s2_glob))
    if len(_s2_cands) > 1:
        raise RuntimeError(
            f"{len(_s2_cands)} S2 caches match {os.path.basename(_s2_glob)}: "
            f"{[os.path.basename(c) for c in _s2_cands]}. They share the fit ids but not the model "
            f"tag, so nothing here can say which teacher each belongs to. Remove the stale one(s).")
    if _s2_cands:
        _s2m, _ = _kvlab.load_model(dt_feasibility_train.MODEL)
        _s2m.config.use_cache = False
        _s2m.eval()
        S2NORM_CACHE = _s2_cands[0]
        S2NORM = dt_mla_mha2mla.load_s2norm_cache(S2NORM_CACHE, _s2m, _fids,
                                                  fit_doc_sha256=_fhash, eval_doc_sha256=_ehash)
        del _s2m
    elif DRY_RUN:
        S2NORM_CACHE = None
        print(f"⚠ @@S2@@ {_s2_wanted} requested and NO S2 cache matches "
              f"{os.path.basename(_s2_glob)}.")
        print(f"  DRY_RUN reads the cache and never fits: the fit is a full forward pass over "
              f"{tuple(_fids.shape)} tokens and it writes to Drive.")
        print(f"  S2NORM stays None. Nothing on a dry run consumes it. Set DRY_RUN = False and re-run "
              f"this cell to fit it into {S2NORM_DIR}.")
    else:
        _s2m, _ = _kvlab.load_model(dt_feasibility_train.MODEL)
        _s2m.config.use_cache = False
        _s2m.eval()
        print(f"@@S2@@ fitting the S2 norms on the DISJOINT fit split, {tuple(_fids.shape)} tokens "
              f"(CPU, one forward pass) ...")
        import dt_mla_fit_llama
        S2NORM = dt_mla_fit_llama.fit_s2norm(_s2m, _fids)
        print(f"  S2 layout cross-check: max relative deviation {S2NORM.pop('layout_rel_max'):.3e} "
              f"(bar {dt_mla_mha2mla.QKV_LAYOUT_TOL:g})")
        S2NORM_CACHE = os.path.join(S2NORM_DIR, dt_mla_mha2mla.s2norm_cache_name(_s2m, _fids))
        dt_mla_mha2mla.save_s2norm_cache(S2NORM_CACHE, S2NORM, _s2m, _fids,
                                         fit_doc_sha256=_fhash, eval_doc_sha256=_ehash)
        del _s2m
    if S2NORM is not None:
        S2NORM_EVAL_HASHES = _ehash
        print(f"@@S2@@ S2NORM ready for {_s2_wanted}: q/k {S2NORM['q'].shape}, "
              f"n_tok {S2NORM['n_tok']}, digest {dt_mla_mha2mla.s2norm_digest(S2NORM)[:16]} "
              f"-> {S2NORM_CACHE}")
        print(f"  ⚠ ON THIS CHECKPOINT S2 IS DOMINATED BY ONE PLANE, and that is reported rather than "
              f"fixed (§4c.4): plane 6 turns at ~1e-3 rad/token and scores ~85 against 3-13 elsewhere, "
              f"so at G <= 4 almost every kept pair is plane 6 and the fast planes lose RoPE. It is "
              f"their method's own selection, and the uniform per-head rule would pick plane 6 too.")
else:
    print(f"@@S2@@ no MHA2MLA arm in INIT_ARMS ({dt_feasibility_train.S2NORM_ARMS}), so S2NORM stays "
          f"None -- init_branch filters it by S2NORM_ARMS and nothing else can receive one.")

import subprocess
GAUGE_GS = []
_gauge_files = {G: dt_arms.npz_path(dt_feasibility_train.MODEL, "arm3B", G) for G in GAUGE_GS}
_gauge_missing = sorted(G for G, p in _gauge_files.items() if not os.path.exists(p))
if _gauge_missing and DRY_RUN:
    print(f"gauge3B needs arm3B gauges at G = {_gauge_missing} for {dt_feasibility_train.MODEL}: the real "
          f"build records them first (gam_record.py, closed-form). Not on a dry run.")
elif _gauge_missing:
    subprocess.run([sys.executable, os.path.join(REPO_DIR, "src",
                                                 "gam_record.py"),
                    "--model", dt_feasibility_train.MODEL, "--arms", "arm3B", "--cases", "B",
                    "--groups", ",".join(str(G) for G in _gauge_missing), "--outdir", dt_arms.GAUGE_DIR],
                   check=True)
    _still = [G for G in _gauge_missing if not os.path.exists(_gauge_files[G])]
    if _still:
        raise RuntimeError(f"gam_record.py ran but the gauges for G = {_still} are not at "
                           f"{[_gauge_files[G] for G in _still]}")

print(f"GRID: {len(INIT_CELLS)} cells over budgets {sorted({b for _, _, b in INIT_CELLS})} "
      f"({len(INIT_ARMS)} arms) -> {CKPT_ROOT}")

if DRY_RUN:
    print("\n★ DRY RUN -- plumbing checked, NOTHING BUILT and nothing overwritten.")
    print("  Set DRY_RUN = False above to actually convert.")
else:
    dt_feasibility_train.init_cells(
        cells=INIT_CELLS,
        init_ckpt_dir=CKPT_ROOT,
        dstar=DSTAR,
        metric=FISHER,
        sigma=SIGMA,
        s2norm=S2NORM,
    )

print(f"\ncontents of {CKPT_ROOT}:")
for _n in sorted(os.listdir(CKPT_ROOT)):
    _meta = os.path.join(CKPT_ROOT, _n, "kv_tying_meta.json")
    print(f"   {_n:<34} meta: {'yes' if os.path.exists(_meta) else 'MISSING'}")
print("\nNext: the accounting cell below; then the run notebook.")

# %%
import hashlib
import json
_acc, _missing = [], []
for _c in GRID_CELLS:
    _row = {k: _c[k] for k in ("arm", "label", "B", "G", "init", "why", "train", "train_why", "rot_rule", "rot_w")}
    if _c["arm"] == "mha":
        _row.update(state="CONTROL", path=None)
    elif (_c["init"] == "BUILD" or _c["init"].startswith("DEDUP:")) and _c["arm"] not in INIT_ARMS:
        _row.update(state="NOT_BUILT", path=None)
    elif _c["init"] == "BUILD" or _c["init"].startswith("DEDUP:"):
        _d = os.path.join(CKPT_ROOT, f"{_c['arm']}_B{_c['B']}_t0")
        if dt_feasibility_train._checkpoint_valid(_d):
            _m = dt_checkpoint.read_meta(_d)
            if (_m.get("base_model") != GRID["hf_id"] or _m.get("arm") != _c["arm"] or _m.get("B") != _c["B"]
                    or _m.get("G") != _c["G"]):
                raise RuntimeError(f"{_d}: its sidecar says base {_m.get('base_model')!r} arm {_m.get('arm')!r} "
                                   f"B {_m.get('B')!r} G {_m.get('G')!r}; the registry cell is {GRID['hf_id']} "
                                   f"{_c['arm']} B {_c['B']} G {_c['G']}. Not trusted as either.")
            if _c.get("rot_rule") is not None:
                if _m.get("arch") == "mla":
                    _rule, _w = (_m.get("mla_width") or {}).get("width_rule", "argmin"), _m.get("w")
                    if (_m.get("mla_provenance") or {}).get("arm") != _m.get("arm"):
                        raise RuntimeError(f"{_d}: its build record says arm "
                                           f"{(_m.get('mla_provenance') or {}).get('arm')!r}, the sidecar "
                                           f"{_m.get('arm')!r}. Relabelled by hand; not trusted.")
                elif _m.get("arch") == "decoupled":
                    _rule, _w = _c["rot_rule"], _m.get("d_k_R")
                else:
                    raise RuntimeError(f"{_d}: arch {_m.get('arch')!r}, but the registry expects a rotary "
                                       f"width ({_c['rot_rule']}, {_c['rot_w']}) of {_c['arm']}")
                if _rule != _c["rot_rule"] or (_c["rot_w"] is not None and _w != _c["rot_w"]):
                    raise RuntimeError(f"{_d}: built at rotary width {_w} (rule {_rule}), but the registry's "
                                       f"cell {_c['arm']} B {_c['B']} expects {_c['rot_w']} (rule "
                                       f"{_c['rot_rule']}). A valid checkpoint at the wrong width; not used.")
            with open(os.path.join(_d, dt_checkpoint.META_NAME), "rb") as _f:
                _msha = hashlib.sha256(_f.read()).hexdigest()
            _row.update(state="ON_DISK", path=_d, fingerprint=_m["fingerprint"], arch=_m.get("arch"),
                        meta_sha256=_msha)
        else:
            _row.update(state="MISSING", path=_d)
            _missing.append(f"{_c['arm']}_B{_c['B']}")
    else:
        _row.update(state="RECORDED", path=None)
    _acc.append(_row)
_known = {f"{_c['arm']}_B{_c['B']}_t0" for _c in GRID_CELLS}
_extra = sorted(n for n in os.listdir(CKPT_ROOT) if n.endswith("_t0") and n not in _known)

print(f"{MODEL_TAG}: {len(_acc)} registry cells")
for _st in ("ON_DISK", "MISSING", "NOT_BUILT", "RECORDED", "CONTROL"):
    _rs = [r for r in _acc if r["state"] == _st]
    print(f"  {_st:<9} {len(_rs):>3}" + ("" if _st in ("ON_DISK",) else
          "".join(f"\n      {r['arm']:<40} {r['label']:>6} B={r['B']:<5} {r['init']}" for r in _rs)))
if _extra:
    print(f"  in {CKPT_ROOT} but in no registry cell (left alone; e.g. the one-KV-head dedup bases): {_extra}")

if DRY_RUN:
    print("\n★ DRY RUN -- _cells.json NOT written. Set DRY_RUN = False in the init cell, run it, then this.")
else:
    _out = {"model_tag": MODEL_TAG, "registry_sha": REGISTRY_SHA, "hf_id": GRID["hf_id"],
            "revision": GRID["revision"], "ckpt_root": CKPT_ROOT, "module_manifest": MODULE_MANIFEST,
            "cells": _acc, "missing": _missing, "extra_dirs": _extra, "init_arms": INIT_ARMS}
    _p = os.path.join(GRID_INIT_DIR, "_cells.json")
    with open(_p, "w", encoding="utf-8") as _f:
        json.dump(_out, _f, indent=1)
    print(f"\nwrote {_p}")
    if _missing:
        raise RuntimeError(f"{len(_missing)} BUILD cells are not on disk after the init cell: {_missing}. "
                           f"Re-run the init cell (it resumes); a cell that keeps failing is a bug to fix, "
                           f"never a cell to drop.")
    print(f"✔ every BUILD and DEDUP cell of {INIT_ARMS} is on disk and reloads; next: {MODEL_TAG}_run")
