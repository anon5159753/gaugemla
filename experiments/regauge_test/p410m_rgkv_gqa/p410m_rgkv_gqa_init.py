# p410m_rgkv_gqa init: regauges the original model by one condition (REGAUGE), checks it computes the same function and builds its GQA mean-pool t = 0 checkpoints on Drive, closed-form from the weights (nothing is fitted; no GPU).
# Run by colab/launch.py from a Colab notebook, which mounts Drive, clones the repository to /content/repo
# and passes DRY_RUN, REGAUGE as environment variables.

import os

DRIVE_ROOT = "/content/drive"
import sys as _sys
KIT_DIR = os.path.dirname(os.path.abspath(__file__))
if KIT_DIR not in _sys.path:
    _sys.path.insert(0, KIT_DIR)
import regauge_gauges as RG
REGAUGE = os.environ.get("REGAUGE", "")
if REGAUGE not in RG.CONDITIONS:
    raise RuntimeError(f"REGAUGE={REGAUGE!r}: this tag runs one condition per process, one of {list(RG.CONDITIONS)}, "
                       f"and the launcher sets it")
RG_ROOT    = DRIVE_ROOT + "/MyDrive/gaugemla/regauge_test/p410m_rgkv_gqa"
COND_ROOT  = RG_ROOT + "/" + REGAUGE
CKPT_ROOT  = COND_ROOT + "/init/ckpt"
FISHER_LABEL_SEEDS = (20261014,)
FISHER_LABEL_SEED = 20261014
print(f"@@REGAUGE@@ condition {REGAUGE}: {RG.describe(REGAUGE)}")
print(f"@@REGAUGE@@ checkpoints -> {CKPT_ROOT}")
GRID_INIT_DIR = COND_ROOT + "/init"
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
 'src/dt_feasibility_arms.py': '2a22c2b174a5c0943950975acc5a73cc0cf2d49c',
 'src/dt_feasibility_train.py': '4c54fec049c8f7dccddf1131115ad8cde3ac2bdc',
 'src/dt_gqa_arch.py': '2dfeaef57063f9838ce3b7e87c2f4fa73401017d',
 'src/dt_grid_models.py': '6f5902d3b25c73ac5cfddb4ede687a883cdd73dd',
 'src/dt_groupsearch.py': '2119e0a8a588b68bfbcb88cad95759a27d95a3a7',
 'src/dt_logs.py': 'ebeabd8a5cf762a54547e54bd46a8c10d899c7de',
 'src/dt_mla_arch.py': 'de2fe52f01729e5f77a56d7e53bf464fdd847446',
 'src/dt_mla_arch_llama.py': '481f489058549802812fcd8693a80e494d67f672',
 'src/dt_mla_care.py': 'deb4a6006442ef60321fac9528088931c65a3cb7',
 'src/dt_mla_kl_standin.py': '6efc5e8dfd122f5767c00be4e4e8bd17ffcb6d8f',
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
 'src/dt_mla_w16_ours_unrope_cost_kl.py': '8cf2ac876d3d6a8b92af558db90b202c046382f7',
 'src/dt_mla_w16_ours_unrope_cost_klsi.py': '12e11182470cb17ae263bdc07ee8b1fb5a64d14b',
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
 'src/dt_mla_w64_ours_unrope_cost_kl.py': 'fdbe8b41b282b4b4d3f84802631fcc030f1949cb',
 'src/dt_mla_w64_ours_unrope_cost_klsi.py': '650613b40df52093e2f2b07d01850679ad802480',
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
GAUGE_DIR_ON_DRIVE = COND_ROOT + "/gauges"
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
MODEL_TAG    = 'p410m_rgkv_gqa'
REGISTRY_SHA = '9bb5271002fb809e97e3a496d99c380eeb234b7c7d0a0a379ec289a63a80c9a1'
GRID = {'H': 16,
 'Hkv': 16,
 'L': 24,
 'arms_only': ('pool',),
 'base': 'p410m',
 'd_k': 64,
 'dm': 1024,
 'family': 'neox',
 'fisher_exclude_top': 0.01,
 'fisher_labels': 'model',
 'hf_id': 'EleutherAI/pythia-410m',
 'init_budgets': 'trained',
 'init_layout': 'B',
 'kl_width_rule': 'argmin',
 'max_ctx': 2048,
 'model_type': 'gpt_neox',
 'norm_kind': 'ln_affine',
 'o_bias': True,
 'partial_unrope': True,
 'qkv_bias': True,
 'r': 16,
 'regauge': ('identity', 'kvs', 'kvd'),
 'registry_sha': '9bb5271002fb809e97e3a496d99c380eeb234b7c7d0a0a379ec289a63a80c9a1',
 'registry_version': 'grid-2026-10-04.3',
 'revision': '9879c9b5f8bea9051dcb0e68dff21493d67e9d4f',
 'role': 'trained',
 'rope_base': 10000,
 't0_only': True,
 'tag': 'p410m_rgkv_gqa',
 'teacher': {'dtype': 'float32',
             'hf_id': 'EleutherAI/pythia-2.8b',
             'revision': '2a259cdd96a4beb1cdf467512e3904197345f6a9',
             'tokenizer_oid': 'f74dfbfab8f97770a87769c739fb080c21c8bacc',
             'vocab': 50304},
 'tie_word_embeddings': False,
 'tokenizer_oid': 'f74dfbfab8f97770a87769c739fb080c21c8bacc',
 'train_plan': 'mla50',
 'vocab': 50304}
GRID_CELLS = [{'B': 1024,
  'G': 8,
  'arm': 'mla_w64_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_w64_transmla_sigma',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_w64_mha2mla_s2norm',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_w64_care_sigma',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 1024,
  'G': 8,
  'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'pool',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'gauge3B',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'procrustes_simk',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'optimal',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'optwide',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'decoupled_optwide_median',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_w64_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_w64_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 1024,
  'G': 8,
  'arm': 'mla_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '50%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY decided 2026-10-11 (the authors): "the init needs to reflect it and not build so '
         'many arms"; 50% is not trained; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_w64_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_w64_transmla_sigma',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_w64_mha2mla_s2norm',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_w64_care_sigma',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 512,
  'G': 4,
  'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'pool',
  'degenerate': None,
  'init': 'BUILD',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)",
  'why': ''},
 {'B': 512,
  'G': 4,
  'arm': 'gauge3B',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'procrustes_simk',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'optimal',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'optwide',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'decoupled_optwide_median',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_w64_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; '
         'this rotary allocation is t = 0 only under D-A1; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_w64_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; '
         'this rotary allocation is t = 0 only under D-A1; would be BUILD'},
 {'B': 512,
  'G': 4,
  'arm': 'mla_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_w64_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_w64_transmla_sigma',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_w64_mha2mla_s2norm',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_w64_care_sigma',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 256,
  'G': 2,
  'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'pool',
  'degenerate': None,
  'init': 'BUILD',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)",
  'why': ''},
 {'B': 256,
  'G': 2,
  'arm': 'gauge3B',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'procrustes_simk',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'optimal',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'optwide',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'decoupled_optwide_median',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_w64_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; '
         'this rotary allocation is t = 0 only under D-A1; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_w64_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; '
         'this rotary allocation is t = 0 only under D-A1; would be BUILD'},
 {'B': 256,
  'G': 2,
  'arm': 'mla_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '12.5%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_w64_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_w64_ours_unrope_energy|mla_w64_ours_unrope_cost',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_w64_transmla_sigma',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_w64_mha2mla_s2norm',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_w64_care_sigma',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_ours_fisher_sigma',
  'degenerate': None,
  'init': 'SUBSTITUTED:mla_ours_unrope_energy|mla_ours_unrope_cost',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops '
         'the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them'},
 {'B': 128,
  'G': 1,
  'arm': 'decoupled_optwide_fisher_sigma_dkr64',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'pool',
  'degenerate': None,
  'init': 'BUILD',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)",
  'why': ''},
 {'B': 128,
  'G': 1,
  'arm': 'gauge3B',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'procrustes_simk',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'optimal',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'optwide',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be DEDUP:optimal'},
 {'B': 128,
  'G': 1,
  'arm': 'decoupled_optwide_median',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_w64_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; '
         'this rotary allocation is t = 0 only under D-A1; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_w64_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_ours_unrope_energy',
  'degenerate': None,
  'init': 'OMITTED:TRAINED_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; '
         'this rotary allocation is t = 0 only under D-A1; would be BUILD'},
 {'B': 128,
  'G': 1,
  'arm': 'mla_ours_unrope_cost',
  'degenerate': None,
  'init': 'OMITTED:ARMS_ONLY',
  'label': '6.25%',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'not built',
  'why': 'ARMS_ONLY (the authors 2026-10-12, "so that should be 6 runs in total"): this tag builds only '
         'pool; would be BUILD'},
 {'B': 2048,
  'G': 16,
  'arm': 'mha',
  'degenerate': None,
  'init': 'BUILD',
  'label': 'src',
  'rot_rule': None,
  'rot_w': None,
  'star': False,
  'train': [],
  'train_why': 'NOT_TRAINED: MLA50 decided 2026-10-06 (the authors): only p410m trains the control; its t '
               "= 0 is the run's reference (KL of the teacher to the original)",
  'why': 'the unconverted model (no checkpoint: built in the run cell)'}]

import shutil
import dt_grid_models
if dt_grid_models.registry_sha(MODEL_TAG) != REGISTRY_SHA:
    raise RuntimeError(f"the cloned registry gives {MODEL_TAG} sha {dt_grid_models.registry_sha(MODEL_TAG)[:16]}, "
                       f"this notebook was generated at {REGISTRY_SHA[:16]}. Regenerate the notebook.")
_live = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}
         for c in dt_grid_models.cells(MODEL_TAG)]
if _live != GRID_CELLS:
    raise RuntimeError("the cloned registry's cell table differs from the one embedded here")
RG.check_registry(GRID["regauge"])
dt_feasibility_train.use_model(MODEL_TAG)
if dt_feasibility_train.MODEL != GRID["hf_id"]:
    raise RuntimeError(f"dt_feasibility_train.MODEL is {dt_feasibility_train.MODEL}, the registry says {GRID['hf_id']}")
from huggingface_hub import model_info as _model_info
_main = _model_info(GRID["hf_id"]).sha
if _main != GRID["revision"]:
    raise RuntimeError(f"{GRID['hf_id']}'s main is now {_main}, not the registry's pinned "
                       f"{GRID['revision']}. Every load in the pipeline reads main, so the inits would be "
                       f"built from different weights. Put this to the authors; do not re-pin silently.")

from transformers import AutoConfig as _AutoConfig
_cfg = _AutoConfig.from_pretrained(GRID["hf_id"], revision=GRID["revision"])
_d, _f, _V, _L = _cfg.hidden_size, _cfg.intermediate_size, _cfg.vocab_size, _cfg.num_hidden_layers
N_PARAMS_EXPECTED = (2 * _V * _d + _L * (3 * _d * _d + 3 * _d + _d * _d + _d + 2 * _d * _f + _f + _d + 4 * _d)
                     + 2 * _d)
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
assert FISHER_LABEL_SEED in FISHER_LABEL_SEEDS

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

_fm, _ftok = kvlab.load_model(dt_feasibility_train.MODEL)
_, _epick, _ehash = _pile_split(_ftok, FISHER_N_DOC, skip=0)
_fids, _fpick, _fhash = _pile_split(_ftok, FISHER_N_DOC, skip=FISHER_N_DOC)
if _ehash & _fhash or set(_epick) & set(_fpick):
    raise RuntimeError("the Fisher fit corpus overlaps the eval corpus; the control is void")
print(f"fit corpus: {FISHER_N_DOC} docs, picks {_fpick[:4]} ... -- disjoint from eval "
      f"({len(_ehash & _fhash)} shared hashes, {len(set(_epick) & set(_fpick))} shared indices)")

RG.check_constants(FISHER_N_DOC, FISHER_SEQ_LEN, FISHER_SEED_EVAL, FISHER_DOC_OFFSET)
GATE = RG.gate_and_install(REGAUGE, _fm, _fids, COND_ROOT, RG_ROOT, kvlab, dt_feasibility_train.MODEL, log=print)

del _fm
print(f"@@REGAUGE@@ gate {GATE['verdict']} for {REGAUGE}; nothing is fitted in this script (no Fisher, Sigma, width "
      f"table or KL stand-in: the pool reads none). Next: the pool builds.")

# %%
assert CKPT_ROOT == DRIVE_ROOT + "/MyDrive/gaugemla/regauge_test/p410m_rgkv_gqa/" + REGAUGE + "/init/ckpt", (
    f"CKPT_ROOT is {CKPT_ROOT!r}. p410m_rgkv_gqa's inits live in /MyDrive/gaugemla/regauge_test/p410m_rgkv_gqa/<condition>/init/ckpt and its run "
    "script reads that path; a branch written anywhere else is invisible to it.")

INIT_CELLS_ALL = [['pool', 4, 512], ['pool', 2, 256], ['pool', 1, 128]]
INIT_CELLS = RG.cells_for(REGAUGE, INIT_CELLS_ALL)
INIT_ARMS = list(dict.fromkeys(a for a, _, _ in INIT_CELLS))
_reg_cells = sorted([c["arm"], c["G"] if c["G"] is not None else -1, c["B"]] for c in GRID_CELLS
                    if c["arm"] != "mha" and (c["init"] == "BUILD" or c["init"].startswith("DEDUP:")))
if sorted([a, -1 if g is None else g, b] for a, g, b in INIT_CELLS_ALL) != _reg_cells:
    raise RuntimeError("INIT_CELLS_ALL is not the registry's BUILD + DEDUP cells")
if INIT_CELLS != INIT_CELLS_ALL or INIT_ARMS != [RG.POOL_ARM]:
    raise RuntimeError(f"@@REGAUGE@@ {REGAUGE} would build {INIT_CELLS}, not the registry's {INIT_CELLS_ALL} (the pool alone)")
print(f"@@REGAUGE@@ {REGAUGE} builds {len(INIT_CELLS)} of the registry's {len(INIT_CELLS_ALL)} cells: {INIT_ARMS}")

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

    S2NORM_DIR = COND_ROOT + "/s2norm"
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
        dt_mla_mha2mla.verify_attn_input(_s2m, _fids[:2])
        print(f"@@S2@@ fitting the S2 norms on the DISJOINT fit split, {tuple(_fids.shape)} tokens "
              f"(CPU, one forward pass) ...")
        S2NORM = dt_mla_mha2mla.fit_s2norm(_s2m, _fids)
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
        dstar=None,
        metric=None,
        sigma=None,
        s2norm=None,
    )
    RG.record_builds(REGAUGE, CKPT_ROOT, INIT_CELLS, GATE, os.path.join(RG_ROOT, "identity", "init", "ckpt"), log=print)

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
    elif (_c["init"] == "BUILD" or _c["init"].startswith("DEDUP:")) and not RG.builds(REGAUGE, _c["arm"]):
        _row.update(state="NOT_IN_CONDITION", path=None)
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
for _st in ("ON_DISK", "MISSING", "NOT_IN_CONDITION", "RECORDED", "CONTROL"):
    _rs = [r for r in _acc if r["state"] == _st]
    print(f"  {_st:<9} {len(_rs):>3}" + ("" if _st in ("ON_DISK",) else
          "".join(f"\n      {r['arm']:<40} {r['label']:>6} B={r['B']:<5} {r['init']}" for r in _rs)))
if _extra:
    print(f"  in {CKPT_ROOT} but in no registry cell (left alone; e.g. the one-KV-head dedup bases): {_extra}")

if DRY_RUN:
    print("\n★ DRY RUN -- _cells.json NOT written. Set DRY_RUN = False in the init cell, run it, then this.")
else:
    _out = {"model_tag": MODEL_TAG, "fisher_label_seed": FISHER_LABEL_SEED, "registry_sha": REGISTRY_SHA, "hf_id": GRID["hf_id"],
            "regauge": REGAUGE, "regauge_gauge": RG.CONDITIONS[REGAUGE]["gauge"],
            "regauge_fisher": RG.CONDITIONS[REGAUGE]["fisher"], "gate": {k: GATE[k] for k in ("verdict", "bar", "fp_original", "fp_regauged", "omega_digest")},
            "gate_regauged": GATE["regauged"],
            "revision": GRID["revision"], "ckpt_root": CKPT_ROOT, "module_manifest": MODULE_MANIFEST,
            "cells": _acc, "missing": _missing, "extra_dirs": _extra}
    _p = os.path.join(GRID_INIT_DIR, "_cells.json")
    with open(_p, "w", encoding="utf-8") as _f:
        json.dump(_out, _f, indent=1)
    print(f"\nwrote {_p}")
    if _missing:
        raise RuntimeError(f"{len(_missing)} BUILD cells are not on disk after the init cell: {_missing}. "
                           f"Re-run the init cell (it resumes); a cell that keeps failing is a bug to fix, "
                           f"never a cell to drop.")
    print(f"✔ every BUILD and DEDUP cell is on disk and reloads; next: {MODEL_TAG}_run.ipynb")
