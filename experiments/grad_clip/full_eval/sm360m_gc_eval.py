# sm360m_gc full evaluation: re-scores the saved models of sm360m_gc's 50M branches on the full task splits.
# Cluster form (a PBS job; DRIVE_ROOT is the cluster's storage), kept as the code behind results/full_eval/.

import os
if not os.environ.get("PBS_JOBID"):
    raise SystemExit("sm360m_gc_eval.py runs only inside a PBS job on cluster (bash ~/kvtying/full_eval/submit_all.sh, or "
                     "qsub -v TAG=sm360m_gc jobs/eval.pbs from ~/kvtying/full_eval/logs), never on a login node.")

# %%
import os

DRIVE_ROOT = "/home/user/kvtying/drive"
CKPT_ROOT  = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/sm360m_gc/init/ckpt"
RUNS_ROOT  = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/sm360m_gc/runs"
DATA_DIR   = DRIVE_ROOT + "/MyDrive/gaugemla/grid/_tok/f922b1797f0c/battery13"

if not os.path.isdir(DRIVE_ROOT + "/MyDrive"):
    raise RuntimeError(
        f"{DRIVE_ROOT}/MyDrive does not exist. On cluster it is made once, on the login node, by "
        "setup_cluster.sh; if it was, this node cannot see the home volume: report the node to the admins. "
        "⛔ There is no alternative save path.")

for _name, _d in (("CKPT_ROOT", CKPT_ROOT), ("RUNS_ROOT", RUNS_ROOT), ("DATA_DIR", DATA_DIR)):
    if not _d.startswith(DRIVE_ROOT + "/"):
        raise RuntimeError(f"{_name} = {_d!r} is not under {DRIVE_ROOT}. ⛔ Drive is the only "
                           f"permitted location; fix the path, do not work around it.")

for _d in (CKPT_ROOT, RUNS_ROOT, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

for _d in (CKPT_ROOT, RUNS_ROOT, DATA_DIR):
    _probe = os.path.join(_d, f".write_probe.{os.environ.get('PBS_JOBID', os.getpid())}")
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
print(f"runs        -> {RUNS_ROOT}/<STAMP>")
print(f"task caches -> {DATA_DIR}")

# %%
import sys, json, math, os, time, hashlib, inspect
import numpy as np
import torch

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
torch.backends.cuda.matmul.allow_tf32 = True; print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32); import transformers; print("transformers", transformers.__version__)

try:
    import zstandard
    print("zstandard   ", zstandard.__version__, "(needed for the .zst training shards)")
except ImportError:
    raise RuntimeError("zstandard is missing from the venv: install it on the login node with setup_cluster.sh, "
                       "then resubmit. A job never installs packages.")

if not torch.cuda.is_available():
    raise RuntimeError(
        "NO GPU. This notebook refuses to run on CPU.\n"
        "On cluster, submit it with jobs/run.pbs (-q gpuq -l ngpus=1), then rerun.\n"
        "There is deliberately no CPU fallback: it would produce numbers that look fine, take 100x "
        "longer, and say nothing about it in the output.")

DEV = torch.device("cuda")
_props = torch.cuda.get_device_properties(0)
VRAM_GB = _props.total_memory / 1e9
print("device      ", _props.name)
print("vram        ", f"{VRAM_GB:.1f} GB")
if VRAM_GB < 15:
    raise RuntimeError(f"{VRAM_GB:.1f} GB is not enough: fp32 peak is ~8.7 GB with gradient "
                       f"checkpointing and ~13.9 GB without. On cluster use gpuq (jobs/run.pbs).")
if VRAM_GB < 20:
    print("             NOTE: 16 GB class device -- keep gradient checkpointing ON (peak ~8.7 GB).")

from transformers.models.llama import modeling_llama as _mg
_problems = []
_sig = list(inspect.signature(_mg.eager_attention_forward).parameters)
if _sig[:5] != ["module", "query", "key", "value", "attention_mask"]:
    _problems.append(f"eager_attention_forward's leading params are {_sig[:5]}, expected "
                     f"['module','query','key','value','attention_mask']")
if "scaling" not in _sig:
    _problems.append("eager_attention_forward has no `scaling` keyword")
try:
    _lay_src = inspect.getsource(_mg.LlamaDecoderLayer.forward)
except (OSError, TypeError):
    _lay_src = ""
    _problems.append("cannot read LlamaDecoderLayer.forward source -- the 2-tuple unpack is unverifiable")
if _lay_src and ", _ = self.self_attn(" not in _lay_src:
    _problems.append("LlamaDecoderLayer.forward no longer unpacks a 2-TUPLE from self.self_attn(...); "
                     "our forward returns (output, weights) and would be misread")
for _kw in ("attention_mask=", "past_key_values=", "position_embeddings="):
    if _lay_src and _kw not in _lay_src:
        _problems.append(f"LlamaDecoderLayer.forward no longer passes {_kw} -- our forward's signature "
                         f"would receive it positionally or not at all")
if not hasattr(_mg, "apply_rotary_pos_emb"):
    _problems.append("apply_rotary_pos_emb is gone; rotary must be IMPORTED, never reimplemented "
                     "(only 16 of 64 head dims rotate on pythia-410m)")
if not hasattr(_mg, "rotate_half"):
    _problems.append("rotate_half is gone; the MLA slot layout is split-half BECAUSE that is what "
                     "rotate_half does, and it is checked numerically below")
if _problems:
    raise RuntimeError(
        f"transformers {transformers.__version__} has changed the LlamaDecoderLayer attention contract this "
        f"notebook depends on:\n  - " + "\n  - ".join(_problems) +
        "\n\nThe old tree pinned transformers==5.15.0 for this reason. Either install that version or "
        "adapt the modules in the architecture cell to the new shape. ⛔ Do not skip the check: a "
        "changed call shape is SILENT -- wrong numbers, no traceback.")
print(f"contract    LlamaDecoderLayer call shape OK under transformers {transformers.__version__} "
      f"(2-tuple return, keyword call site, eager_attention_forward(module,q,k,v,mask,scaling=...))")

_w_chk = 10
_hp = _w_chk // 2
_th_chk = torch.arange(1, _hp + 1, dtype=torch.float64) * 0.37
_cs_chk = torch.cat([torch.cos(_th_chk), torch.cos(_th_chk)]).expand(1, 3, _w_chk).contiguous()
_sn_chk = torch.cat([torch.sin(_th_chk), torch.sin(_th_chk)]).expand(1, 3, _w_chk).contiguous()
_g_chk = torch.Generator().manual_seed(20260929)
_q_chk = torch.randn(1, 2, 3, _w_chk, generator=_g_chk, dtype=torch.float64)
_k_chk = torch.randn(1, 1, 3, _w_chk, generator=_g_chk, dtype=torch.float64)

def _split_half_ref(x):
    _a, _b = x[..., :_hp], x[..., _hp:]
    _c, _s = torch.cos(_th_chk), torch.sin(_th_chk)
    return torch.cat([_a * _c - _b * _s, _b * _c + _a * _s], dim=-1)

_qr_chk, _kr_chk = _mg.apply_rotary_pos_emb(_q_chk, _k_chk, _cs_chk, _sn_chk)
_rot_dev = max(float((_qr_chk - _split_half_ref(_q_chk)).abs().max()),
               float((_kr_chk - _split_half_ref(_k_chk)).abs().max()))
_rot_moved = int(((_qr_chk - _q_chk).abs().amax(dim=(0, 1, 2)) > 1e-9).sum())
_rh_ok = torch.equal(_mg.rotate_half(_q_chk),
                     torch.cat([-_q_chk[..., _hp:], _q_chk[..., :_hp]], dim=-1))
if (_qr_chk.shape != _q_chk.shape or _kr_chk.shape != _k_chk.shape
        or _rot_moved != _w_chk or _rot_dev > 1e-12 or not _rh_ok):
    raise RuntimeError(
        f"rotary CONTRACT BROKEN under transformers {transformers.__version__}: with cos/sin of width "
        f"{_w_chk}, apply_rotary_pos_emb rotated {_rot_moved}/{_w_chk} dims (shapes "
        f"{tuple(_qr_chk.shape)}/{tuple(_kr_chk.shape)}) and deviated {_rot_dev:.3e} from the "
        f"split-half rotation; rotate_half split-half = {_rh_ok}.\n"
        f"⛔ The MLA arms gather cos/sin by slot_plane and assume the rotary width IS cos.shape[-1], "
        f"with slot s and slot s + w/2 as one complex coordinate. If either of those stopped holding, "
        f"every mla_* branch is exact at offset 0 and wrong at every other offset -- no traceback, "
        f"just a KL curve that rises with t. Read modeling_gpt_neox.apply_rotary_pos_emb and adapt the "
        f"architecture cell; do not delete the check.")
print(f"rotary      width comes from cos.shape[-1]: at w={_w_chk} all {_rot_moved}/{_w_chk} dims "
      f"rotate, split-half form exact to {_rot_dev:.1e}, rotate_half split-half ✓")

torch.manual_seed(20260914)

# %%
MODEL_TAG      = 'sm360m_gc'
REGISTRY_SHA   = '3e93c8aa484230f6e3f237d0edf4816cc36e6d76798cee8c14674d7c0499fadb'
GRID = {'H': 15,
 'Hkv': 5,
 'L': 32,
 'arms_only': ('mla_w16_ours_unrope_cost', 'mla_w16_transmla_sigma', 'mla_w16_mha2mla_s2norm'),
 'base': 'sm360m',
 'd_k': 64,
 'dm': 960,
 'family': 'llama_layout',
 'fisher_exclude_top': 0.01,
 'fisher_labels': 'model',
 'grad_clip': 1.0,
 'hf_id': 'HuggingFaceTB/SmolLM-360M',
 'init_budgets': 'trained',
 'init_layout': 'B',
 'init_plan': 'mla_only',
 'kl_width_rule': 'unrope_argmin',
 'max_ctx': 2048,
 'model_type': 'llama',
 'norm_kind': 'rms',
 'o_bias': False,
 'qkv_bias': False,
 'r': 64,
 'registry_sha': '3e93c8aa484230f6e3f237d0edf4816cc36e6d76798cee8c14674d7c0499fadb',
 'registry_version': 'grid-2026-10-04.3',
 'revision': '59f7ef243ee09a72cbc14cb054393a3e3b771d41',
 'role': 'trained',
 'rope_base': 10000,
 'save_final': True,
 'tag': 'sm360m_gc',
 'teacher': {'dtype': 'float32',
             'hf_id': 'HuggingFaceTB/SmolLM2-1.7B',
             'revision': 'effd688a12921b4cc83e3312b6feb579f70f9c71',
             'tokenizer_oid': 'f922b1797f0c88e71addc8393787831f2477a4bd',
             'vocab': 49152},
 'tie_word_embeddings': True,
 'tokenizer_oid': 'f922b1797f0c88e71addc8393787831f2477a4bd',
 'train_plan': 'mla50',
 'vocab': 49152,
 'w_compare': 16}
GRID_CELLS = [('mla_w16_transmla_sigma', 1.25, 160, [50000000]),
 ('mla_w16_mha2mla_s2norm', 1.25, 160, [50000000]),
 ('mla_w16_ours_unrope_cost', 1.25, 160, [50000000]),
 ('mla_w16_transmla_sigma', 0.625, 80, [50000000]),
 ('mla_w16_mha2mla_s2norm', 0.625, 80, [50000000]),
 ('mla_w16_ours_unrope_cost', 0.625, 80, [50000000]),
 ('mla_w16_transmla_sigma', 0.3125, 40, [50000000]),
 ('mla_w16_mha2mla_s2norm', 0.3125, 40, [50000000]),
 ('mla_w16_ours_unrope_cost', 0.3125, 40, [50000000])]
CELL_W = {('mla_w16_mha2mla_s2norm', 0.3125): ('fixed:16', 16),
 ('mla_w16_mha2mla_s2norm', 0.625): ('fixed:16', 16),
 ('mla_w16_mha2mla_s2norm', 1.25): ('fixed:16', 16),
 ('mla_w16_ours_unrope_cost', 0.3125): ('fixed:16', 16),
 ('mla_w16_ours_unrope_cost', 0.625): ('fixed:16', 16),
 ('mla_w16_ours_unrope_cost', 1.25): ('fixed:16', 16),
 ('mla_w16_transmla_sigma', 0.3125): ('fixed:16', 16),
 ('mla_w16_transmla_sigma', 0.625): ('fixed:16', 16),
 ('mla_w16_transmla_sigma', 1.25): ('fixed:16', 16)}
CARE_ARMS = ('mla_care_sigma', 'mla_w64_care_sigma')
MAX_TOKENS     = 50_000_000
if os.environ.get("MAX_TOKENS", "50M") != "50M":
    raise RuntimeError(f"this model trains 50M per branch only (qsub -v MAX_TOKENS=50M), "
                       f"got {os.environ.get('MAX_TOKENS')!r}")
LR_PEAK        = 2e-5
MODEL            = GRID["hf_id"]
MODEL_REVISION   = GRID["revision"]
TEACHER          = GRID["teacher"]["hf_id"]
TEACHER_REVISION = GRID["teacher"]["revision"]
TEACHER_DTYPE    = GRID["teacher"]["dtype"]
if MAX_TOKENS != 50_000_000:
    raise RuntimeError(f"MAX_TOKENS = {MAX_TOKENS:,}: this model trains at 50M only (MLA50, the authors 2026-10-06)")
if LR_PEAK != 2e-5:
    raise RuntimeError(f"LR_PEAK = {LR_PEAK}: this model trains at 2e-5 only, with no guard (MLA50, the authors 2026-10-06)")
if TEACHER_DTYPE not in ("float32", "bfloat16"):
    raise RuntimeError(f"TEACHER_DTYPE = {TEACHER_DTYPE!r}: float16 with eager attention overflows")
if TEACHER_REVISION is None:
    raise RuntimeError(f"{TEACHER} has no pinned revision in the registry (a gated repo this account "
                       f"cannot read). Request access on the Hub, pin it, regenerate.")
SEED_TRAIN     = 42
STAMP_SUFFIX   = ""
STAMP          = (f"grid-{MODEL_TAG}-{TEACHER.split('/')[-1]}-{TEACHER_DTYPE}-"
                  f"{MAX_TOKENS // 1_000_000}M-lr{LR_PEAK:.0e}-s{SEED_TRAIN}"
                  + (f"-{STAMP_SUFFIX}" if STAMP_SUFFIX else ""))
GUARD_SEED     = 42
GUARD_STAMP    = (f"grid-{MODEL_TAG}-{TEACHER.split('/')[-1]}-{TEACHER_DTYPE}-5M-lr{2e-5:.0e}-s{GUARD_SEED}"
                  + (f"-{STAMP_SUFFIX}" if STAMP_SUFFIX else ""))
GUARD_RULE     = "none (MLA50, the authors 2026-10-06): the MLA group at 50M, peak 2e-5, with no 5M grid"
CARE_REQUIRED_VERSION = 'care-2026-09-30.1'
RUN_CELLS      = {(a, g) for a, g, _b, _toks in GRID_CELLS if MAX_TOKENS in _toks}
RUN_ARMS       = tuple(dict.fromkeys(a for a, g, _b, _toks in GRID_CELLS if MAX_TOKENS in _toks))
RUN_GS         = tuple(sorted({g for a, g in RUN_CELLS if a != "mha"}))
LATE_DENSE     = True
LOG_TRAIN_LOSS = True
LOG_GRAD_NORM  = True
GRAD_CLIP      = GRID["grad_clip"]

LR         = LR_PEAK
BATCH      = 16
GRAD_CKPT  = True
TIME_PROBE = 50

WARMUP_FRAC    = 0.10
LR_MIN_FRAC    = 0.10

def lr_at(step, total_steps):
    w = max(1, int(WARMUP_FRAC * total_steps))
    lo = LR_MIN_FRAC * LR
    if step < w:
        return LR * (step + 1) / w
    p = min(1.0, (step - w) / max(1, total_steps - w))
    return lo + (LR - lo) * 0.5 * (1.0 + math.cos(math.pi * p))

for _T in (245, 2442):
    _w = max(1, int(WARMUP_FRAC * _T))
    _lr = [lr_at(s, _T) for s in range(_T)]
    assert abs(_lr[_w - 1] - LR) < 1e-15 and abs(_lr[_w] - LR) < 1e-15, "warmup must end AT the peak"
    assert all(a < b for a, b in zip(_lr[:_w - 1], _lr[1:_w])), "warmup must rise strictly"
    assert all(a >= b for a, b in zip(_lr[_w:], _lr[_w + 1:])), "the decay must never rise"
    assert LR_MIN_FRAC * LR <= _lr[-1] < LR_MIN_FRAC * LR * 1.001, f"the last step must sit at the floor: {_lr[-1]}"
del _T, _w, _lr

PRETRAIN_TOKENS = 300_000_000_000
PRETRAIN_FRAC = 100.0 * MAX_TOKENS / PRETRAIN_TOKENS

_EARLY = [0, 50_000, 100_000, 200_000, 400_000, 700_000, 1_000_000, 1_400_000, 2_000_000]
_LATE_SPARSE = [4_000_000, 8_000_000, 14_000_000, 20_000_000]
_LATE_DENSE = [3_000_000, 4_000_000, 6_000_000, 8_000_000,
               11_000_000, 14_000_000, 17_000_000, 20_000_000]

def build_schedule(max_tokens, late_dense=True):
    tail = _LATE_DENSE if late_dense else _LATE_SPARSE
    ev = sorted({t for t in _EARLY + tail if t < max_tokens} | {0, max_tokens})
    picks = {0, max_tokens}
    for _frac in (0.025, 0.10, 0.50):
        _target = _frac * max_tokens
        picks.add(min(ev, key=lambda t: (abs(t - _target), t)))
    bat = sorted(picks)
    phase = {0: "endpoint", max_tokens: "endpoint"}

    if ev != sorted(set(ev)) or ev[0] != 0 or ev[-1] != max_tokens:
        raise RuntimeError(f"schedule is not a strictly increasing 0..{max_tokens} ladder: {ev}")
    if max(ev) != max_tokens:
        raise RuntimeError(f"max(EVAL_AT_TOKENS)={max(ev)} != MAX_TOKENS={max_tokens}. This is the "
                           f"exact drift the run cell's completeness test cannot survive.")
    for _t in bat:
        if _t not in ev:
            raise RuntimeError(f"battery point {_t} is not a KL point -- the battery must land on a "
                               f"KL point so the two instruments are paired")
    return ev, bat, phase

EVAL_AT_TOKENS, BATTERY_AT_TOKENS, BATTERY_PHASE = build_schedule(MAX_TOKENS, LATE_DENSE)
_TOKENS_PER_STEP = BATCH * 512
_TOTAL_STEPS = math.ceil(MAX_TOKENS / _TOKENS_PER_STEP)
WARMUP_END_TOKENS = max(1, int(WARMUP_FRAC * _TOTAL_STEPS)) * _TOKENS_PER_STEP
GUARD_WARMUP_END = max(1, int(WARMUP_FRAC * math.ceil(5_000_000 / _TOKENS_PER_STEP))) * _TOKENS_PER_STEP
LADDERS = {5000000: {'battery': [0, 5000000],
           'kl': [0,
                  12500,
                  17500,
                  25000,
                  35000,
                  50000,
                  70000,
                  100000,
                  137500,
                  175000,
                  250000,
                  350000,
                  500000,
                  750000,
                  1000000,
                  1500000,
                  2000000,
                  2250000,
                  2750000,
                  3500000,
                  4250000,
                  5000000]},
 50000000: {'battery': [0,
                        250000,
                        700000,
                        1750000,
                        5000000,
                        7500000,
                        10000000,
                        15000000,
                        22500000,
                        35000000,
                        50000000],
            'kl': [0,
                   125000,
                   175000,
                   250000,
                   350000,
                   500000,
                   700000,
                   1000000,
                   1375000,
                   1750000,
                   2500000,
                   3500000,
                   5000000,
                   7500000,
                   10000000,
                   15000000,
                   20000000,
                   22500000,
                   27500000,
                   35000000,
                   42500000,
                   50000000]}}
EVAL_AT_TOKENS = sorted(set(LADDERS[MAX_TOKENS]["kl"]) | {WARMUP_END_TOKENS})
BATTERY_AT_TOKENS = list(LADDERS[MAX_TOKENS]["battery"])
BATTERY_PHASE = {0: "endpoint", MAX_TOKENS: "endpoint"}
if EVAL_AT_TOKENS[0] != 0 or EVAL_AT_TOKENS[-1] != MAX_TOKENS or \
        any(a >= b for a, b in zip(EVAL_AT_TOKENS, EVAL_AT_TOKENS[1:])):
    raise RuntimeError(f"the KL ladder is not a strictly increasing 0..{MAX_TOKENS:,}: {EVAL_AT_TOKENS}")
if not set(BATTERY_AT_TOKENS) <= set(EVAL_AT_TOKENS):
    raise RuntimeError("every battery point must be a KL point, so the two instruments are paired")
if MAX_TOKENS == 50_000_000 and sum(t < WARMUP_END_TOKENS for t in BATTERY_AT_TOKENS if t) != 3:
    raise RuntimeError("the 50M battery must have three points inside the warmup")
print(f"warmup end {WARMUP_END_TOKENS:,} tokens ({WARMUP_END_TOKENS // _TOKENS_PER_STEP} of "
      f"{_TOTAL_STEPS} steps) -- a KL point, read by the guard")

print(f"stamp      {STAMP}   (names every log file, endpoint directory and row)")
print(f"budget     {MAX_TOKENS/1e6:.1f}M tokens/arm = {PRETRAIN_FRAC:.5f}% of pretraining")
print(f"seed       {SEED_TRAIN}   (names the branch AND orders the documents)")
print(f"arms       {'every arm found' if RUN_ARMS is None else ', '.join(RUN_ARMS)}")
print(f"G          {'every G found' if RUN_GS is None else RUN_GS}")
print(f"KL points  {len(EVAL_AT_TOKENS)}: {[round(t/1e6, 2) for t in EVAL_AT_TOKENS]}M")
print(f"batteries  {len(BATTERY_AT_TOKENS)}: {[round(t/1e6, 2) for t in BATTERY_AT_TOKENS]}M "
      f"(endpoint n at {[round(t/1e6, 2) for t in sorted(BATTERY_PHASE)]}M)")
print(f"logging    train_loss={LOG_TRAIN_LOSS}  grad_norm={LOG_GRAD_NORM}")
print(f"clip       global grad norm <= {GRAD_CLIP} every step, on the KD loss averaged over every token "
      f"(@@CLIP@@)")
print(f"lr         peak {LR:.1e}; linear warmup over the first {WARMUP_FRAC:.0%} of optimiser steps, "
      f"then cosine down to {LR_MIN_FRAC * LR:.1e} ({LR_MIN_FRAC:.0%} of the peak) at the last step")
print(f"teacher    {TEACHER} @ {TEACHER_REVISION[:12]}  {TEACHER_DTYPE}")
print(f"cells      {len(RUN_CELLS)} branches at {MAX_TOKENS // 1_000_000}M from the registry "
      f"({REGISTRY_SHA[:16]})")
OUT_DIR = os.path.join(RUNS_ROOT, STAMP)
os.makedirs(OUT_DIR, exist_ok=True)
_probe = os.path.join(OUT_DIR, f".write_probe.{os.environ.get('PBS_JOBID', os.getpid())}")
with open(_probe, "w") as _f:
    _f.write("probe")
with open(_probe) as _f:
    assert _f.read() == "probe", "Drive read back something other than what was written"
os.remove(_probe)
print(f"logs       -> {OUT_DIR}  (writable)")
N_TOKENS_MIN = 20_000_000
if MAX_TOKENS < N_TOKENS_MIN:
    print(f"\n⛔ E2 IS UNDERPOWERED, declared BEFORE the run by PREREGISTRATION_2 section 5.2: "
          f"{MAX_TOKENS/1e6:.1f}M is below B-2's N_TOKENS_MIN of {N_TOKENS_MIN/1e6:.0f}M. "
          f"That is automatic, not a judgement made after seeing the numbers.")
else:
    print(f"\n✅ the budget MEETS B-2's N_TOKENS_MIN ({N_TOKENS_MIN/1e6:.0f}M), so section 5.2's "
          f"ex-ante power gate on E2 does NOT fire.")
    print("   ⚠ That closes the BUDGET gate and nothing else: C2 and C4 have never been run, B-4 is "
          "unruled, and E1 needs BOTH pool and gauge3B in RUN_ARMS to exist at all.")

# %%
from huggingface_hub import constants as _hfc
if os.environ.get("HF_HUB_OFFLINE") != "1":
    raise RuntimeError("HF_HUB_OFFLINE is not 1: submit this through jobs/eval.pbs, which sets it. This job reads only "
                       "what is already on cluster and must not download anything.")
_snap = os.path.join(_hfc.HF_HUB_CACHE, "models--" + MODEL.replace("/", "--"), "snapshots", MODEL_REVISION)
_wts = [f for f in (os.listdir(_snap) if os.path.isdir(_snap) else []) if f.endswith((".safetensors", ".bin"))]
if not os.path.isfile(os.path.join(_snap, "config.json")) or not _wts:
    raise RuntimeError(f"{MODEL} @ {MODEL_REVISION} is not in the Hub cache ({_snap}). Every run job of this tag loaded "
                       f"it from there; if the cache was cleared, tell the authors. Do not run this job online instead.")
_TASKS13 = ['lambada', 'piqa', 'arc_easy', 'boolq', 'siqa', 'openbookqa', 'hellaswag', 'induction', 'kv_synth_N50', 'arith_1d', 'arc_challenge', 'winogrande', 'mmlu']
_miss = [t for t in _TASKS13 if not os.path.isfile(os.path.join(DATA_DIR, f"{t}.json"))]
if _miss:
    raise RuntimeError(f"task caches missing from {DATA_DIR}: {_miss}. The run jobs read all 13 there; this job "
                       f"cannot rebuild them offline. Tell the authors.")
if not os.path.isfile(os.path.join(OUT_DIR, "_manifest.json")):
    raise RuntimeError(f"{OUT_DIR}/_manifest.json is missing: this is not the stamp the models were trained in.")
print(f"preflight  offline; {MODEL} @ {MODEL_REVISION[:12]} cached ({len(_wts)} weight file(s)); the {len(_TASKS13)} "
      f"task caches and the stamp's manifest are on Drive")

# %%
import glob

META_NAME = "kv_tying_meta.json"
KNOWN_ARCHES = ("mha", "gqa", "optimal", "decoupled", "mla")
SEARCHED_ARMS = ("optimal_search", "procrustes_search", "procrustes_simk")
ARCH_OF = {"mha": "mha",
           "pool": "gqa", "gauge3B": "gqa", "procrustes": "gqa",
           "procrustes_search": "gqa", "procrustes_simk": "gqa",
           "optimal": "optimal", "optimal_search": "optimal", "optwide": "optimal",
           "decoupled_optwide": "decoupled",
           "decoupled_optwide_median": "decoupled",
           "decoupled_optwide_flexible": "decoupled",
           "decoupled_optwide_fisher": "decoupled",
           "decoupled_optwide_fisher_sigma": "decoupled",
           "decoupled_optwide_fisher_sigma_dstar": "decoupled",
           "mla_ours_fisher_sigma": "mla",
           "mla_transmla_sigma": "mla",
           "mla_mha2mla_s2norm": "mla",
           "mla_care_sigma": "mla",
           "mla_w64_ours_fisher_sigma": "mla", "mla_w64_transmla_sigma": "mla",
           "mla_w64_mha2mla_s2norm": "mla", "mla_w64_care_sigma": "mla",
           "decoupled_optwide_fisher_sigma_dkr64": "decoupled",
           "mla_w16_mha2mla_s2norm": "mla",
           "mla_w16_ours_unrope_cost": "mla",
           "mla_w16_transmla_sigma": "mla",
           "optwide_fisher": "optimal"}
WANTED_ARMS = ("pool", "gauge3B", "optimal", "optimal_search",
               "procrustes", "procrustes_simk", "optwide", "decoupled_optwide",
               "decoupled_optwide_median", "decoupled_optwide_flexible",
               "decoupled_optwide_fisher", "optwide_fisher",
               "decoupled_optwide_fisher_sigma", "decoupled_optwide_fisher_sigma_dstar",
               "mla_ours_fisher_sigma", "mla_transmla_sigma", "mla_mha2mla_s2norm",
               "mla_care_sigma",
               "mla_w64_ours_fisher_sigma", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm",
               "mla_w64_care_sigma", "decoupled_optwide_fisher_sigma_dkr64",
               "mla_w16_mha2mla_s2norm",
               "mla_w16_ours_unrope_cost",
               "mla_w16_transmla_sigma")
WANTED_ARMS = ('mla_w16_transmla_sigma', 'mla_w16_mha2mla_s2norm', 'mla_w16_ours_unrope_cost')
assert {a for a, *_ in GRID_CELLS} <= set(WANTED_ARMS), (GRID_CELLS, WANTED_ARMS)
G1_DEDUP_BASE = {"optimal_search": "optimal", "procrustes_search": "procrustes",
                 "optwide": "optimal"}

metas = sorted(glob.glob(os.path.join(CKPT_ROOT, "**", META_NAME), recursive=True))
if not metas:
    stray = [d for d in sorted(glob.glob(os.path.join(CKPT_ROOT, "*")))
             if os.path.isdir(d) and glob.glob(os.path.join(d, "*.safetensors"))]
    raise RuntimeError(
        f"no {META_NAME} anywhere under {CKPT_ROOT}, but {len(stray)} dir(s) hold safetensors.\n"
        f"  e.g. {stray[:3]}\n"
        f"⛔ A checkpoint WITHOUT its sidecar meta cannot be reloaded: arch, G and the per-layer head "
        f"partition live only there and the skeleton rebuild needs all three. Copy the {META_NAME} "
        f"files across too.")

hdr = (f"{'arm':<28}{'G':>3}{'G_v':>5}  {'arch':<9}{'wide':<6}{'grp/layer':<11}"
       f"{'d_k^R|w':>16}{'GB':>6}  {'shards':<7}dir")
print(f"{len(metas)} checkpoint(s) under {CKPT_ROOT}\n"); print(hdr); print("-" * len(hdr))

CKPTS, problems = {}, []
for m in metas:
    d = os.path.dirname(m)
    rel = os.path.relpath(d, CKPT_ROOT)
    try:
        j = json.load(open(m, encoding="utf-8"))
    except Exception as e:
        problems.append(f"{rel}: meta will not parse ({type(e).__name__}: {e})")
        continue
    arm, arch, G = j.get("arm", "?"), j.get("arch", "?"), j.get("G", "?")
    if G is None and isinstance(j.get("B"), int):
        G = j["B"] / (2 * GRID["d_k"])
    G_v, gbl = j.get("G_v", "-"), j.get("groups_by_layer")
    wide = arm in ("optwide", "optwide_fisher")

    if os.path.exists(os.path.join(d, "model.safetensors")):
        shards = "1"
    elif os.path.exists(os.path.join(d, "model.safetensors.index.json")):
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8"))
        shards = str(len(set(idx["weight_map"].values())))
    else:
        shards = "NONE"
        problems.append(f"{rel}: no model.safetensors and no .index.json -- no weights to load")
    gb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
             if os.path.isfile(os.path.join(d, f))) / 1e9
    gbl_txt = f"{len(gbl)} layers" if gbl else ("--" if G in (1, "?") else "(default)")
    _dkr = j.get("w") if arch == "mla" else j.get("d_k_R")
    if _dkr is None:
        dkr_txt = ""
    elif isinstance(_dkr, list):
        dkr_txt = f"{min(_dkr)}..{max(_dkr)}({len(_dkr)})"
    else:
        dkr_txt = str(_dkr)
    print(f"{arm:<28}{str(G):>3}{str(G_v):>5}  {arch:<9}{'YES' if wide else '':<6}{gbl_txt:<11}"
          f"{dkr_txt:>16}{gb:>6.2f}  {shards:<7}{rel}")

    if gbl is None and arm in SEARCHED_ARMS and isinstance(G, int) and G > 1:
        problems.append(f"{rel}: searched arm {arm!r} at G={G} has NO groups_by_layer -- its partition "
                        f"cannot be re-derived and the loader WILL RAISE. Unusable without re-running "
                        f"the search.")
    if arch not in KNOWN_ARCHES:
        problems.append(f"{rel}: arch {arch!r} is not one of {KNOWN_ARCHES}")
    if arm in ARCH_OF and arch != ARCH_OF[arm]:
        problems.append(f"{rel}: meta says arch={arch!r} but arm {arm!r} restructures into "
                        f"{ARCH_OF[arm]!r} -- one of the two is wrong and the skeleton would mismatch")
    for k in ("arch", "base_model", "fingerprint"):
        if k not in j:
            problems.append(f"{rel}: meta is missing required key {k!r}")
    if j.get("trained"):
        problems.append(
            f"{rel}: this is a TRAINED endpoint checkpoint (run {j.get('run_id')}, "
            f"{j.get('tokens', 0)/1e6:.1f}M tokens), not an initialisation. Training it again would "
            f"start from a model that is already {j.get('tokens', 0)/1e6:.1f}M tokens in, while every "
            f"row logged it as t=0. Move it back out of {os.path.basename(CKPT_ROOT)}/.")
    if isinstance(G, (int, float)) and not isinstance(G, bool) and (arm, G) in CELL_W:
        _rr, _rw = CELL_W[(arm, G)]
        if arch == "mla":
            _hr, _hw = (j.get("mla_width") or {}).get("width_rule", "argmin"), j.get("w")
        else:
            _hr, _hw = _rr, j.get("d_k_R")
        if _hr != _rr or (_rw is not None and _hw != _rw):
            problems.append(f"{rel}: built at rotary width {_hw} (rule {_hr}), but the registry's cell "
                            f"{arm} G={G} expects {_rw} (rule {_rr}) -- a valid checkpoint at the "
                            f"wrong width under this name")
    if isinstance(G, (int, float)) and not isinstance(G, bool):
        if (arm, G) in CKPTS:
            problems.append(f"{rel}: a SECOND checkpoint claims arm={arm!r} G={G} (the first was "
                            f"{os.path.relpath(CKPTS[(arm, G)], CKPT_ROOT)}). One of them would "
                            f"silently win. Remove or rename one.")
        CKPTS[(arm, G)] = d

bases = {json.load(open(m, encoding="utf-8")).get("base_model") for m in metas}
print(f"\nbase_model(s): {sorted(b for b in bases if b)}")
if len([b for b in bases if b]) > 1:
    problems.append(f"checkpoints span MORE THAN ONE base model {sorted(bases)} -- one teacher cannot "
                    f"be the reference for all of them")

G_SEEN = sorted({g for _a, g in CKPTS})
BRANCHES, skipped_dup, missing = [], [], []
for arm in WANTED_ARMS:
    for G in G_SEEN:
        if G == 1 and arm in G1_DEDUP_BASE:
            if (arm, 1) in CKPTS:
                skipped_dup.append(f"{arm}_G1 == {G1_DEDUP_BASE[arm]}_G1")
            continue
        if (arm, G) in CKPTS:
            BRANCHES.append((arm, G))
        else:
            missing.append(f"{arm}_G{G}")

print(f"\nG values on disk: {G_SEEN}")
print(f"arms on disk:     {sorted({a for a, _g in CKPTS})}")
print(f"\n⭐ {len(BRANCHES)} branch(es) to train:")
for arm, G in BRANCHES:
    print(f"     {arm}_G{G}")
if skipped_dup:
    print(f"\n   {len(skipped_dup)} G=1 duplicate(s) SKIPPED -- bit-identical to their base arm, so "
          f"training them would be an unlabelled seed replicate, not a new arm:")
    for s in skipped_dup:
        print(f"     {s}")
if missing:
    print(f"\n   {len(missing)} wanted (arm, G) not on disk: {missing}")
    print(f"   That is not an error -- the grid is whatever is in init/. Drop the folder in and rerun.")

if problems:
    print(f"\n⚠ {len(problems)} PROBLEM(S) -- these are unusable or will fail at load:")
    for p in problems:
        print(f"  - {p}")
    raise RuntimeError(f"{len(problems)} checkpoint problem(s) above. ⛔ Fix or remove them before "
                       f"training: a run that silently drops an arm is worse than one that stops.")
print("\n✔ every meta parses, carries its required keys, names a known arch consistent with its arm,")
print("  and no searched arm has lost its per-layer partition.")
if not BRANCHES:
    raise RuntimeError("no trainable branches were discovered -- check the arm names in the metas "
                       f"against WANTED_ARMS {WANTED_ARMS}")

# %%
def get_layers(model):
    return model.model.layers

def get_attn(layer):
    return layer.self_attn

def _rotary_emb(model):
    for get in (lambda m: m.model.rotary_emb,
                lambda m: get_layers(m)[0].self_attn.rotary_emb):
        try:
            e = get(model)
            if e is not None and hasattr(e, "inv_freq"):
                return e
        except (AttributeError, IndexError):
            continue
    raise RuntimeError(f"could not locate a rotary_emb with inv_freq on {type(model).__name__}")

def is_llama(model):
    try:
        a = get_attn(get_layers(model)[0])
    except (AttributeError, IndexError):
        return False
    return all(hasattr(a, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))

def is_converted(model):
    try:
        a = get_attn(get_layers(model)[0])
    except (AttributeError, IndexError):
        return False
    return type(a).__name__ == "LlamaMLAAttention" or bool(getattr(type(a), "_grid_twin", False))

def geometry(model):
    c = model.config
    layers = get_layers(model)
    L, H, dm = len(layers), int(c.num_attention_heads), int(c.hidden_size)
    d_k = dm // H
    emb = _rotary_emb(model)
    r = 2 * int(emb.inv_freq.shape[0])
    sc = float(getattr(emb, "attention_scaling", 1.0))
    if sc != 1.0:
        raise RuntimeError(
            f"rotary attention_scaling = {sc} != 1: cos/sin carry a factor the slot gather does "
            f"not model, so every rotary slot would be scaled by it.")
    if is_converted(model):
        seen = {(int(get_attn(l).n_heads), int(get_attn(l).head_size)) for l in layers}
        if len(seen) != 1:
            raise RuntimeError(f"converted layers disagree on (n_heads, head_size): {sorted(seen)}")
        H_mod, dh = seen.pop()
        if H_mod != H or dh * H != dm:
            raise RuntimeError(f"converted attention says H={H_mod} dh={dh}; config says H={H} "
                               f"dm={dm}")
        if not 0 < r <= dh or r % 2:
            raise RuntimeError(f"2*len(inv_freq) = {r} is not an even rotary width in (0, {dh}]")
        return L, H, dh, dh, dm, r
    a = get_attn(layers[0])
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not hasattr(a, n):
            raise RuntimeError(f"{type(model).__name__} has no {n} on self_attn -- this is not an "
                               f"unconverted LLaMA-family model")
    hd = getattr(c, "head_dim", None)
    if hd is not None and hd != d_k:
        raise RuntimeError(f"config.head_dim = {hd} but hidden_size/num_attention_heads = {d_k}")
    kv = getattr(c, "num_key_value_heads", H)
    if kv != H and a.k_proj.weight.shape[0] != kv * d_k:
        raise RuntimeError(f"config says {kv} key/value heads but k_proj has {a.k_proj.weight.shape[0]} rows")
    if not 0 < r <= d_k or r % 2:
        raise RuntimeError(f"2*len(inv_freq) = {r} is not an even rotary width in (0, {d_k}]")
    pct = getattr(c, "rotary_pct", None) or getattr(c, "partial_rotary_factor", None)
    if pct is None:
        pct = (getattr(c, "rope_parameters", None) or {}).get("partial_rotary_factor", 1.0)
    if int(round(d_k * pct)) != r:
        raise RuntimeError(f"inv_freq says r = {r} but the config's rotary fraction {pct:.4f} of "
                           f"d_k = {d_k} says {int(round(d_k * pct))}")
    return L, H, d_k, d_k, dm, r

def groups_of(H, G):
    assert H % G == 0, f"H={H} not divisible by G={G}"
    per = H // G
    return [list(range(j * per, (j + 1) * per)) for j in range(G)]

def head_group(groups, H):
    gi = np.zeros(H, dtype=int)
    for g, idx in enumerate(groups):
        for i in idx:
            gi[i] = g
    return gi

# %%
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb, eager_attention_forward)

MLA_SCHEMA = 'mla/v1'
MLA_BIAS_MEMBERS = ('dkv', 'uk', 'uv')
MLA_MODES = ('shared', 'per_head')
MLA_PROBE_SEED = 20260929
MLA_PROBE_TOL = 0.05
MLA_PROBE_VOCAB = 49152
STALE_LLAMA_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')

def pack_index(dC, slot_head, d_h):
    dC = [int(x) for x in dC]
    sh = [int(x) for x in slot_head]
    H, D, w = len(dC), sum(dC), len(sh)
    out, off = [], 0
    for i in range(H):
        own = [D + s for s in range(w) if sh[s] == i]
        if dC[i] < 0 or dC[i] + len(own) != d_h:
            raise ValueError(f"pack_index: head {i} has dC={dC[i]} content rows and {len(own)} slots; "
                             f"they must sum to d_h={d_h} (no padding anywhere, §2.2)")
        out.extend(range(off, off + dC[i]))
        out.extend(own)
        off += dC[i]
    if sorted(out) != list(range(H * d_h)):
        raise ValueError(f"pack_index: not a permutation of range({H * d_h}); a slot_head entry lies "
                         f"outside [0, {H}) or D + w != H*d_h (D={D}, w={w})")
    return out

def attn_param_count(H, d_h, dm, w, d_c, D, rotary_mode, bias_set):
    n = dm * d_c + (d_c if "dkv" in bias_set else 0)
    n += (dm + 1) * w
    n += (dm + 1) * D
    n += (dm + 1) * (H * w if rotary_mode == "shared" else w)
    n += D * d_c + (D if "uk" in bias_set else 0)
    n += H * d_h * d_c + (H * d_h if "uv" in bias_set else 0)
    return int(n)

def _mla_linear(dm, out, bias, dtype, device, requires_grad):
    lin = nn.Linear(dm, out, bias=bool(bias)).to(device=device, dtype=dtype)
    for p in lin.parameters():
        p.requires_grad_(requires_grad)
    return lin

def _mla_param_names(attn):
    bs = attn.bias_set
    names = ["dkv.weight"] + (["dkv.bias"] if "dkv" in bs else [])
    names += ["kr.weight", "kr.bias", "q_content.weight", "q_content.bias", "q_rot.weight", "q_rot.bias"]
    names += ["uk"] + (["uk_bias"] if "uk" in bs else [])
    names += ["uv"] + (["uv_bias"] if "uv" in bs else [])
    names += ["dense.weight"] + (["dense.bias"] if attn.dense.bias is not None else [])
    return names

def _mla_buffer_names(attn):
    return (["slot_plane", "slot_head", "dC"] + (["pack_idx"] if attn.rotary_mode == "per_head" else [])
            + ["mla_scale"])

def _mla_alloc_sha(meta):
    body = {"mla_schema": meta["mla_schema"], "rotary_mode": meta["rotary_mode"],
            "w": int(meta["w"]), "d_c": int(meta["d_c"]), "bias_set": list(meta["bias_set"]),
            "mla_scale": repr(float(meta["mla_scale"])),
            "mla_layers": [{k: [int(x) for x in lay[k]] for k in ("dC", "slot_plane", "slot_head")}
                           for lay in meta["mla_layers"]]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def _mla_check_spec(spec, L, H, d_h, r):
    w, d_c, mode = spec["w"], spec["d_c"], spec["rotary_mode"]
    for nm, v in (("w", w), ("d_c", d_c)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"MLA {nm} must be an int, got {v!r} ({type(v).__name__})")
    if w < 2 or w % 2:
        raise ValueError(f"MLA rotary width w={w} must be even and >= 2 (w = 0 is outside every "
                         f"arm's method, D4)")
    if d_c < 1:
        raise ValueError(f"MLA latent width d_c={d_c} must be >= 1")
    if mode not in MLA_MODES:
        raise ValueError(f"rotary_mode {mode!r} not in {MLA_MODES}")
    bs = list(spec["bias_set"])
    if bs != sorted(set(bs)) or not set(bs) <= set(MLA_BIAS_MEMBERS):
        raise ValueError(f"bias_set {bs!r} must be a sorted, duplicate-free subset of {MLA_BIAS_MEMBERS}")
    sc = spec["mla_scale"]
    if not isinstance(sc, float) or not (0.0 < sc < math.inf):
        raise ValueError(f"mla_scale {sc!r} must be a finite positive float")
    layers = spec["layers"]
    if len(layers) != L:
        raise ValueError(f"allocation covers {len(layers)} layers, the model has {L}")
    half, out = w // 2, []
    for li, lay in enumerate(layers):
        sp = [int(x) for x in lay["slot_plane"]]
        sh = [int(x) for x in lay["slot_head"]]
        dC = [int(x) for x in lay["dC"]]
        if len(sp) != w or len(sh) != w or len(dC) != H:
            raise ValueError(f"layer {li}: slot_plane/slot_head must have w={w} entries and dC H={H} "
                             f"(got {len(sp)}/{len(sh)}/{len(dC)})")
        if any(p < 0 or p >= r // 2 for p in sp):
            raise ValueError(f"layer {li}: slot_plane entries must lie in [0, {r // 2})")
        if sp[:half] != sp[half:]:
            raise ValueError(f"layer {li}: slot s and slot s+w/2 must share one plane -- they are ONE "
                             f"complex coordinate (split-half layout, R1)")
        if any(x < 0 for x in dC):
            raise ValueError(f"layer {li}: negative dC")
        if mode == "shared":
            if any(x != -1 for x in sh):
                raise ValueError(f"layer {li}: slot_head must be -1 everywhere in shared mode")
            if len(set(dC)) != 1:
                raise ValueError(f"layer {li}: shared mode needs one content width for every head, "
                                 f"got {sorted(set(dC))}")
            for p in range(r // 2):
                if sp[:half].count(p) > H:
                    raise ValueError(f"layer {li}: plane {p} holds {sp[:half].count(p)} pairs > H={H}")
        else:
            if any(x < 0 or x >= H for x in sh):
                raise ValueError(f"layer {li}: per_head slot_head entries must lie in [0, {H})")
            if sh[:half] != sh[half:]:
                raise ValueError(f"layer {li}: slot s and slot s+w/2 must belong to one head")
            pack_index(dC, sh, d_h)
        out.append((dC, sp, sh))
    return out

class LlamaMLAAttention(LlamaAttention):

    def _mla_qkv(self, hidden_states, position_embeddings, past_key_values=None):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, w, dh = self.n_heads, self.w, self.head_size
        cos, sin = position_embeddings
        cos_s = cos.index_select(-1, self.slot_plane)
        sin_s = sin.index_select(-1, self.slot_plane)
        c = self.dkv(hidden_states).unsqueeze(1)
        kr = self.kr(hidden_states).unsqueeze(1)
        if self.rotary_mode == "shared":
            qr = self.q_rot(hidden_states).view(B, T, H, w).transpose(1, 2)
        elif self.rotary_mode == "per_head":
            qr = self.q_rot(hidden_states).unsqueeze(1)
        else:
            raise RuntimeError(f"LlamaMLAAttention: unknown rotary_mode {self.rotary_mode!r}")
        qr, kr = apply_rotary_pos_emb(qr, kr, cos_s, sin_s)
        if past_key_values is not None:
            kr, c = past_key_values.update(kr, c, self.layer_idx)
        S = c.shape[2]
        kc = nn.functional.linear(c, self.uk, self.uk_bias)
        v = nn.functional.linear(c, self.uv, self.uv_bias).view(B, S, H, dh).transpose(1, 2)
        qc = self.q_content(hidden_states)
        if self.rotary_mode == "shared":
            d0 = self.dC0
            q = torch.cat((qc.view(B, T, H, d0).transpose(1, 2), qr), dim=-1)
            k = torch.cat((kc.view(B, S, H, d0).transpose(1, 2), kr.expand(-1, H, -1, -1)), dim=-1)
        else:
            q = torch.cat((qc, qr[:, 0]), dim=-1).index_select(-1, self.pack_idx)
            q = q.view(B, T, H, dh).transpose(1, 2)
            k = torch.cat((kc[:, 0], kr[:, 0]), dim=-1).index_select(-1, self.pack_idx)
            k = k.view(B, S, H, dh).transpose(1, 2)
        return q, k, v

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        q, k, v = self._mla_qkv(hidden_states, position_embeddings, past_key_values)
        attn_output, attn_weights = eager_attention_forward(
            self, q, k, v, attention_mask, scaling=self.mla_scale,
            dropout=0.0 if not self.training else self.attention_dropout)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.dense(attn_output), attn_weights

def install(model, spec):
    L, H, d_k, d_v, dm, r = geometry(model)
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention implementation is {impl!r}; LlamaMLAAttention calls "
                           f"eager_attention_forward directly and only the eager path is correct")
    if d_k != d_v:
        raise ValueError(f"d_k={d_k} != d_v={d_v}; the MLA value path assumes one head width")
    per_layer = _mla_check_spec(spec, L, H, d_k, r)
    w, d_c, mode = spec["w"], spec["d_c"], spec["rotary_mode"]
    bias_set, scale = tuple(spec["bias_set"]), float(spec["mla_scale"])
    for li, lay in enumerate(get_layers(model)):
        dC, sp, sh = per_layer[li]
        D = sum(dC)
        attn = get_attn(lay)
        if int(getattr(attn, "num_key_value_groups", 1)) != 1:
            raise RuntimeError(
                f"layer {li}: num_key_value_groups = {attn.num_key_value_groups} != 1. LLaMA's "
                f"eager_attention_forward would repeat_kv the MLA key, which is one cache head "
                f"shared by every query head already -- a GQA host needs its own path.")
        p = attn.q_proj
        dtype, device, rg = p.weight.dtype, p.weight.device, p.weight.requires_grad
        attn.dkv = _mla_linear(dm, d_c, "dkv" in bias_set, dtype, device, rg)
        attn.kr = _mla_linear(dm, w, True, dtype, device, rg)
        attn.q_content = _mla_linear(dm, D, True, dtype, device, rg)
        attn.q_rot = _mla_linear(dm, H * w if mode == "shared" else w, True, dtype, device, rg)
        attn.uk = nn.Parameter(torch.empty(D, d_c, dtype=dtype, device=device), requires_grad=rg)
        if "uk" in bias_set:
            attn.uk_bias = nn.Parameter(torch.empty(D, dtype=dtype, device=device), requires_grad=rg)
        else:
            attn.register_parameter("uk_bias", None)
        attn.uv = nn.Parameter(torch.empty(H * d_v, d_c, dtype=dtype, device=device), requires_grad=rg)
        if "uv" in bias_set:
            attn.uv_bias = nn.Parameter(torch.empty(H * d_v, dtype=dtype, device=device), requires_grad=rg)
        else:
            attn.register_parameter("uv_bias", None)
        attn.register_buffer("slot_plane", torch.tensor(sp, dtype=torch.long, device=device), persistent=True)
        attn.register_buffer("slot_head", torch.tensor(sh, dtype=torch.long, device=device), persistent=True)
        attn.register_buffer("dC", torch.tensor(dC, dtype=torch.long, device=device), persistent=True)
        if mode == "per_head":
            attn.register_buffer("pack_idx", torch.tensor(pack_index(dC, sh, d_k),
                                                          dtype=torch.long, device=device),
                                 persistent=True)
        attn.register_buffer("mla_scale", torch.tensor(scale, dtype=dtype, device=device), persistent=True)
        attn.dense = attn.o_proj
        del attn.o_proj
        del attn.q_proj, attn.k_proj, attn.v_proj
        attn.head_size = d_v
        attn.n_heads, attn.w, attn.d_c, attn.rotary_mode = H, w, d_c, mode
        attn.dC0 = dC[0] if mode == "shared" else None
        attn.bias_set, attn.mla_scale_value = bias_set, scale
        attn.__class__ = LlamaMLAAttention

def skeleton(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MLA_SCHEMA!r}")
    G, B = meta.get("G"), meta.get("B")
    if not isinstance(B, int) or isinstance(B, bool):
        raise TypeError(f"meta B must be an int, got {B!r} ({type(B).__name__})")
    if G is not None and (not isinstance(G, int) or isinstance(G, bool) or B != G * (d_k + d_v)):
        raise ValueError(f"meta G = {G!r} is not an int with B = G*(d_k+d_v) (B = {B})")
    w, d_c = meta["w"], meta["d_c"]
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (w, d_c)):
        raise TypeError(f"meta w/d_c must be ints, got {w!r}/{d_c!r}")
    if w + d_c != B:
        raise ValueError(f"cache identity BROKEN: w + d_c = {w} + {d_c} = {w + d_c} != the budget B = {B}. "
                         f"Refusing to load a branch measured against a different budget.")
    spec = {k: meta[k] for k in ("w", "d_c", "rotary_mode", "bias_set", "mla_scale")}
    spec["layers"] = meta["mla_layers"]
    install(model, spec)
    return dict(L=L, H=H, G=G, w=w, d_c=d_c, rotary_mode=meta["rotary_mode"], d_k=d_k, dm=dm,
                cache_width=w + d_c)

def mla_meta(model):
    L, H, d_k, d_v, dm, r = geometry(model)
    attns = [get_attn(lay) for lay in get_layers(model)]
    for li, a in enumerate(attns):
        if type(a).__name__ != "LlamaMLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not LlamaMLAAttention")
    a0 = attns[0]
    one = {(a.w, a.d_c, a.rotary_mode, a.bias_set, a.mla_scale_value) for a in attns}
    if len(one) != 1:
        raise AssertionError(f"layers disagree on (w, d_c, mode, bias_set, scale): "
                             f"{sorted(map(str, one))}")
    meta = {"mla_schema": MLA_SCHEMA, "rotary_mode": a0.rotary_mode, "w": int(a0.w),
            "d_c": int(a0.d_c), "mla_scale": float(a0.mla_scale_value),
            "bias_set": list(a0.bias_set),
            "mla_layers": [{k: [int(x) for x in a.get_buffer(k).detach().cpu().tolist()]
                            for k in ("dC", "slot_plane", "slot_head")} for a in attns]}
    meta["alloc_sha256"] = _mla_alloc_sha(meta)
    counts = [attn_param_count(H, d_k, dm, meta["w"], meta["d_c"], sum(lay["dC"]),
                                  meta["rotary_mode"], meta["bias_set"])
              for lay in meta["mla_layers"]]
    meta["attn_params_by_layer"] = counts
    meta["attn_params_per_layer"] = counts[0] if len(set(counts)) == 1 else None
    return meta

def mla_fingerprint(model):
    h = hashlib.sha256()
    for li, lay in enumerate(get_layers(model)):
        a = get_attn(lay)
        if type(a).__name__ != "LlamaMLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not LlamaMLAAttention")
        h.update(f"{li}|{a.rotary_mode}|{a.w}|{a.d_c}|{a.dC0}|{a.mla_scale_value!r}|"
                 f"{','.join(a.bias_set)}|".encode("utf-8"))
        h.update("|".join(sorted(n for n, _ in a.named_parameters())).encode("utf-8"))
        for kind, names in (("p", _mla_param_names(a)), ("b", _mla_buffer_names(a))):
            for nm in names:
                t = a.get_parameter(nm) if kind == "p" else a.get_buffer(nm)
                arr = t.detach().cpu().contiguous().numpy()
                h.update(f"{kind}:{nm}|{tuple(arr.shape)}|{arr.dtype}|".encode("utf-8"))
                h.update(arr.tobytes())
    return h.hexdigest()[:16]

def check_alloc(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    if meta.get("mla_schema") != MLA_SCHEMA:
        raise ValueError(f"meta mla_schema {meta.get('mla_schema')!r} != {MLA_SCHEMA!r}")
    sha = _mla_alloc_sha(meta)
    if sha != meta.get("alloc_sha256"):
        raise RuntimeError(f"alloc_sha256 mismatch: the meta's allocation hashes to {sha[:16]}, the "
                           f"meta records {str(meta.get('alloc_sha256'))[:16]} -- lists were edited")
    layers = get_layers(model)
    if len(meta["mla_layers"]) != len(layers):
        raise RuntimeError(f"meta has {len(meta['mla_layers'])} allocations, model has "
                           f"{len(layers)} layers")
    w, d_c, mode = int(meta["w"]), int(meta["d_c"]), meta["rotary_mode"]
    bias_set, scale = tuple(meta["bias_set"]), float(meta["mla_scale"])
    counts = meta.get("attn_params_by_layer")
    if counts is None or len(counts) != len(layers):
        raise RuntimeError("meta attn_params_by_layer is missing or has the wrong length")
    for li, lay in enumerate(layers):
        a = get_attn(lay)
        if type(a).__name__ != "LlamaMLAAttention":
            raise RuntimeError(f"layer {li} holds {type(a).__name__}, not LlamaMLAAttention")
        for stale in STALE_LLAMA_MODULES:
            if stale in a._modules:
                raise RuntimeError(f"layer {li} still carries {stale}; install() removes it, so this "
                                   f"layer was never converted or the skeleton ran on a live model")
        got = (a.w, a.d_c, a.rotary_mode, a.bias_set, a.n_heads, a.mla_scale_value)
        if got != (w, d_c, mode, bias_set, H, scale):
            raise RuntimeError(f"layer {li}: module attrs {got} != meta "
                               f"{(w, d_c, mode, bias_set, H, scale)}")
        want = meta["mla_layers"][li]
        for nm in ("slot_plane", "slot_head", "dC"):
            b = a.get_buffer(nm)
            if b.dtype != torch.long or b.dim() != 1 or nm in a._non_persistent_buffers_set:
                raise RuntimeError(f"layer {li}: buffer {nm} must be a persistent 1-D long tensor")
            have = [int(x) for x in b.detach().cpu().tolist()]
            ref = [int(x) for x in want[nm]]
            if have != ref:
                bad = next((i for i, (x, y) in enumerate(zip(have, ref)) if x != y),
                           min(len(have), len(ref)))
                raise RuntimeError(f"layer {li}: buffer {nm} disagrees with the sidecar meta at "
                                   f"index {bad} (buffer {have[bad:bad + 4]}, meta {ref[bad:bad + 4]}). "
                                   f"The state dict and the meta describe different allocations.")
        dC = [int(x) for x in want["dC"]]
        if mode == "shared":
            if "pack_idx" in a._buffers or a.dC0 != dC[0]:
                raise RuntimeError(f"layer {li}: shared mode carries no pack_idx and dC0 == dC[0]")
        else:
            pi = a.get_buffer("pack_idx")
            if a.dC0 is not None or pi.dtype != torch.long \
                    or "pack_idx" in a._non_persistent_buffers_set:
                raise RuntimeError(f"layer {li}: per_head needs a persistent long pack_idx, dC0 None")
            if pi.detach().cpu().tolist() != pack_index(dC, want["slot_head"], d_k):
                raise RuntimeError(f"layer {li}: pack_idx is not the permutation dC/slot_head imply")
        s = a.get_buffer("mla_scale")
        if s.dim() != 0 or not s.is_floating_point() \
                or "mla_scale" in a._non_persistent_buffers_set:
            raise RuntimeError(f"layer {li}: mla_scale must be a persistent 0-dim float buffer")
        if float(s.item()) != float(torch.tensor(scale, dtype=s.dtype).item()):
            raise RuntimeError(f"layer {li}: mla_scale buffer {float(s.item())!r} != meta {scale!r} "
                               f"cast to {s.dtype}")
        names = sorted(n for n, _ in a.named_parameters())
        if names != sorted(_mla_param_names(a)):
            raise RuntimeError(f"layer {li}: parameters {names} != the set bias_set {bias_set} "
                               f"implies {sorted(_mla_param_names(a))}")
        D = sum(dC)
        shapes = {"dkv.weight": (d_c, dm), "dkv.bias": (d_c,), "kr.weight": (w, dm), "kr.bias": (w,),
                  "q_content.weight": (D, dm), "q_content.bias": (D,),
                  "q_rot.weight": (H * w if mode == "shared" else w, dm),
                  "q_rot.bias": (H * w if mode == "shared" else w,),
                  "uk": (D, d_c), "uk_bias": (D,), "uv": (H * d_v, d_c), "uv_bias": (H * d_v,)}
        for n, t in a.named_parameters():
            if n in shapes and tuple(t.shape) != shapes[n]:
                raise RuntimeError(f"layer {li}: {n} has shape {tuple(t.shape)}, "
                                   f"expected {shapes[n]}")
        n_count = attn_param_count(H, d_k, dm, w, d_c, D, mode, bias_set)
        n_live = sum(t.numel() for n, t in a.named_parameters() if not n.startswith("dense."))
        if not (n_count == n_live == int(counts[li])):
            raise RuntimeError(f"layer {li}: attention parameter count formula {n_count}, live "
                               f"{n_live}, meta {counts[li]} disagree")
    return True

def _tensor_bytes(h, name, t):
    arr = t.detach().cpu().contiguous().numpy()
    h.update(f"{name}|{tuple(arr.shape)}|{arr.dtype}|".encode("utf-8"))
    h.update(arr.tobytes())

def teacher_attn_digest(model):
    if not is_llama(model):
        raise RuntimeError("teacher_attn_digest(llama) on a non-LLaMA model")
    h = hashlib.sha256()
    for li, lay in enumerate(get_layers(model)):
        a = get_attn(lay)
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(a, nm):
                raise RuntimeError(f"layer {li} has no {nm}: the model is already restructured")
            lin = getattr(a, nm)
            _tensor_bytes(h, f"{li}.{nm}.weight", lin.weight)
            if lin.bias is not None:
                _tensor_bytes(h, f"{li}.{nm}.bias", lin.bias)
    return h.hexdigest()

def _mla_probe_inputs():
    rng = np.random.default_rng(MLA_PROBE_SEED)
    half = rng.integers(0, MLA_PROBE_VOCAB, size=(2, 128))
    ids = np.concatenate([half, half], axis=1).astype(np.int64)
    pos = [int(x) for x in np.linspace(8, 254, 12).round()]
    voc = sorted(int(x) for x in rng.choice(MLA_PROBE_VOCAB, size=32, replace=False))
    return ids, pos, voc

@torch.no_grad()
def _mla_probe_eval(model, ids, pos, voc):
    was = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
        x = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=dev)
        logits = model(x).logits.double().cpu()
    finally:
        model.train(was)
    x = x.cpu()
    lp = torch.log_softmax(logits, dim=-1)
    nll = -lp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1)
    P = torch.tensor(pos, dtype=torch.long)
    V = torch.tensor(voc, dtype=torch.long)
    sel = logits[:, P][:, :, V]
    true = logits[:, P].gather(-1, x[:, P + 1][:, :, None]).squeeze(-1)
    return {"logits": sel.tolist(), "true_logit": true.tolist(), "nll": nll.tolist()}

def mla_probe(model):
    ids, pos, voc = _mla_probe_inputs()
    out = _mla_probe_eval(model, ids, pos, voc)
    return {"seed": MLA_PROBE_SEED, "ids": ids.tolist(), "positions": pos, "vocab": voc,
            "tol": MLA_PROBE_TOL, "dtype": str(next(model.parameters()).dtype), **out}

def check_probe(model, probe, verbose=True):
    for k in ("ids", "positions", "vocab", "logits", "true_logit", "nll", "tol"):
        if k not in probe:
            raise KeyError(f"mla probe is missing {k!r}")
    got = _mla_probe_eval(model, probe["ids"], probe["positions"], probe["vocab"])
    dev = {k: float(np.abs(np.asarray(got[k], dtype=np.float64)
                           - np.asarray(probe[k], dtype=np.float64)).max())
           for k in ("logits", "true_logit", "nll")}
    worst = max(dev.values())
    tol, stored = float(MLA_PROBE_TOL), float(probe["tol"])
    if verbose:
        note = "" if abs(stored - tol) <= 1e-12 else f", sidecar wrote {stored:.1e}"
        flag = "" if worst <= tol else "  <- above the reference; O(1) would be a fault, 1e-3..5e-2 is f32 noise"
        print(f"    mla probe (diagnostic, not a gate): max |dev| logits {dev['logits']:.2e}  "
              f"true {dev['true_logit']:.2e}  nll {dev['nll']:.2e}  (reference {tol:.1e}{note}){flag}")
    return dev

import sys as _sys

def _tw_rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def _tw_rope(x, cos, sin):
    r = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x_rot, x_pass = x[..., :r], x[..., r:]
    return torch.cat(((x_rot * cos) + (_tw_rotate_half(x_rot) * sin), x_pass), dim=-1)

def _tw_cos_sin_all(attn, hidden_states, position_embeddings, S, T):
    cos, sin = position_embeddings
    if S == T:
        return cos, sin
    pos = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
    cos_a, sin_a = attn._rotary_ref(hidden_states, pos)
    if not (torch.equal(cos_a[:, S - T:], cos) and torch.equal(sin_a[:, S - T:], sin)):
        raise RuntimeError(f"layer {attn.layer_idx}: the cached positions are not 0..{S - 1} ending at this call's "
                           f"{T}; this module re-rotates its pre-rotation cache and refuses to guess positions")
    return cos_a, sin_a

def _tw_attend(attn, q, k, v, attention_mask, B, T):
    out, w = attn._host_eager(attn, q, k, v, attention_mask, scaling=attn.scaling,
                              dropout=0.0 if not attn.training else attn.attention_dropout)
    return attn.o_proj(out.reshape(B, T, -1).contiguous()), w

class _GQAMixin:
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                **kwargs):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, G, d = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, G, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, G, d).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _tw_rope(q, cos, sin), _tw_rope(k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if G != H:
            k = k.index_select(1, self.head_to_group)
            v = v.index_select(1, self.head_to_group)
        return _tw_attend(self, q, k, v, attention_mask, B, T)

class _OptimalMixin:
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                **kwargs):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, Gk, Gv, d, c_k, c_v = self.n_heads, self.n_kv_heads, self.n_kv_heads_v, self.head_dim, self.c_k, self.c_v
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        bx = self.bk_proj(hidden_states).view(B, T, Gk, c_k).transpose(1, 2)
        bvx = self.bv_proj(hidden_states).view(B, T, Gv, c_v).transpose(1, 2)
        if past_key_values is not None:
            bx, bvx = past_key_values.update(bx, bvx, self.layer_idx)
        S = bx.shape[2]
        k = torch.einsum("bhtc,hdc->bhtd", bx.index_select(1, self.head_to_group_k), self.Ck)
        v = torch.einsum("bhtc,hdc->bhtd", bvx.index_select(1, self.head_to_group_v), self.Cv)
        cos, sin = position_embeddings
        cos_s, sin_s = _tw_cos_sin_all(self, hidden_states, position_embeddings, S, T)
        q, k = _tw_rope(q, cos, sin), _tw_rope(k, cos_s, sin_s)
        return _tw_attend(self, q, k, v, attention_mask, B, T)

class _DecoupledMixin:
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                **kwargs):
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H, d, c_KV, d_k_R = self.n_heads, self.head_dim, self.c_KV, self.d_k_R
        q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
        bkv = self.bkv_proj(hidden_states).view(B, T, 1, c_KV).transpose(1, 2)
        bkr = self.bkr_proj(hidden_states).view(B, T, 1, d_k_R).transpose(1, 2)
        if past_key_values is not None:
            bkr, bkv = past_key_values.update(bkr, bkv, self.layer_idx)
        S = bkv.shape[2]
        bkv_h, bkr_h = bkv.expand(-1, H, -1, -1), bkr.expand(-1, H, -1, -1)
        k_rot = torch.einsum("bhtc,hdc->bhtd", bkr_h, self.Ckr)
        k_free = torch.einsum("bhtc,hdc->bhtd", bkv_h, self.Ck)
        k = torch.cat((k_rot, k_free), dim=-1)
        v = torch.einsum("bhtc,hdc->bhtd", bkv_h, self.Cv)
        cos, sin = position_embeddings
        cos_s, sin_s = _tw_cos_sin_all(self, hidden_states, position_embeddings, S, T)
        q, k = _tw_rope(q, cos, sin), _tw_rope(k, cos_s, sin_s)
        return _tw_attend(self, q, k, v, attention_mask, B, T)

GQAAttention, OptimalGQAAttention, DecoupledLatentAttention = _GQAMixin, _OptimalMixin, _DecoupledMixin
_TWINS = {}

def _twin(mixin, host_cls):
    key = (mixin, host_cls)
    if key not in _TWINS:
        eager = getattr(_sys.modules[host_cls.__module__], "eager_attention_forward", None)
        if eager is None:
            raise RuntimeError(f"{host_cls.__module__} has no eager_attention_forward")
        name = mixin.__name__.strip("_").replace("Mixin", "") + host_cls.__name__
        _TWINS[key] = type(name, (mixin, host_cls), {"_host_eager": staticmethod(eager), "_grid_twin": True})
    return _TWINS[key]

def _host_attn(lay):
    a = get_attn(lay)
    if getattr(type(a), "_grid_twin", False) or type(a).__name__ == "LlamaMLAAttention":
        raise RuntimeError(f"layer {a.layer_idx} is already restructured ({type(a).__name__})")
    if getattr(a, "qk_layernorm", False):
        raise RuntimeError("qk_layernorm: the key is not linear in x; no construction here applies")
    return a

def _finish_twin(attn, model, mixin, H):
    attn.n_heads = H
    attn.num_key_value_groups = 1
    attn.head_size = attn.head_dim
    object.__setattr__(attn, "_rotary_ref", _rotary_emb(model))
    attn.__class__ = _twin(mixin, type(attn))

def _grid_expand_gqa(model):
    c = model.config
    H = int(c.num_attention_heads)
    Hkv = int(getattr(c, "num_key_value_heads", None) or H)
    if Hkv == H:
        return 1
    if H % Hkv:
        raise ValueError(f"H = {H} is not a multiple of the {Hkv} KV heads")
    n_rep = H // Hkv
    d = int(getattr(c, "head_dim", None) or c.hidden_size // H)
    with torch.no_grad():
        for lay in get_layers(model):
            a = get_attn(lay)
            for name in ("k_proj", "v_proj"):
                lin = getattr(a, name)
                if lin.weight.shape[0] != Hkv * d:
                    raise RuntimeError(f"{name} has {lin.weight.shape[0]} rows, expected Hkv*d = {Hkv * d}")
                new = nn.Linear(lin.in_features, H * d, bias=lin.bias is not None).to(
                    dtype=lin.weight.dtype, device=lin.weight.device)
                new.weight.copy_(lin.weight.view(Hkv, d, -1).repeat_interleave(n_rep, dim=0).reshape(H * d, -1))
                if lin.bias is not None:
                    new.bias.copy_(lin.bias.view(Hkv, d).repeat_interleave(n_rep, dim=0).reshape(H * d))
                setattr(a, name, new)
            a.num_key_value_groups = 1
            if hasattr(a, "num_key_value_heads"):
                a.num_key_value_heads = H
    c.num_key_value_heads = H
    return n_rep

def _empty_linear(dm, out, dtype, device):
    return nn.Linear(dm, out, bias=True).to(device=device, dtype=dtype)

def _groups_for(meta, H, G, L, wide):
    gbl, arm = meta.get("groups_by_layer"), meta.get("arm")
    if not isinstance(G, int) or isinstance(G, bool):
        raise TypeError(f"meta G must be an int, got {G!r} ({type(G).__name__}).")
    if gbl is None and arm in SEARCHED_ARMS and G > 1:
        raise RuntimeError(f"checkpoint for searched arm {arm!r} at G={G} has no 'groups_by_layer' in its meta: "
                           f"its partition cannot be re-derived from (H={H}, G={G}). Unusable without the search.")
    if wide:
        one = [list(range(H))]
        return [one] * L, [one] * L
    if gbl is not None:
        if len(gbl) != L:
            raise ValueError(f"groups_by_layer has {len(gbl)} entries, model has {L} layers")
        gbl = [[list(g) for g in per_layer] for per_layer in gbl]
        for li, per_layer in enumerate(gbl):
            flat = sorted(i for g in per_layer for i in g)
            if flat != list(range(H)) or len(per_layer) != G:
                raise ValueError(f"groups_by_layer[{li}] is not a partition of the {H} heads into {G} groups")
        return gbl, gbl
    G_v = meta.get("G_v") or G
    return [groups_of(H, G)] * L, [groups_of(H, G_v)] * L

def _skeleton_gqa(model, meta):
    L, H, d_k, d_v, dm, _r = geometry(model)
    G = meta["G"]
    if H % G:
        raise ValueError(f"H={H} not divisible by G={G}")
    if meta.get("G_v") not in (None, G):
        raise ValueError(f"meta has G_v={meta['G_v']} != G={G}; the gqa form has one head_to_group")
    gbl_k, _ = _groups_for(meta, H, G, L, wide=False)
    for li, lay in enumerate(get_layers(model)):
        attn = _host_attn(lay)
        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        groups = gbl_k[li]
        attn.q_proj = _empty_linear(dm, H * d_k, dtype, device)
        attn.k_proj = _empty_linear(dm, len(groups) * d_k, dtype, device)
        attn.v_proj = _empty_linear(dm, len(groups) * d_v, dtype, device)
        attn.n_kv_heads, attn.n_kv_groups = len(groups), H // len(groups)
        attn.register_buffer("head_to_group", torch.tensor(head_group(groups, H), dtype=torch.long,
                                                           device=device), persistent=False)
        _finish_twin(attn, model, _GQAMixin, H)
    return dict(L=L, H=H, G=G, d_k=d_k, dm=dm, cache_width=len(gbl_k[0]) * d_k)

def _skeleton_optimal(model, meta):
    L, H, d_k, d_v, dm, _r = geometry(model)
    G, B = meta["G"], meta.get("B")
    wide = meta.get("arm") in ("optwide", "optwide_fisher")
    G_v = meta.get("G_v") or G
    if wide and (G_v != G or meta.get("groups_by_layer") is not None):
        raise ValueError("wide fixes one group over all heads on both sides")
    if G is None:
        if not wide or not isinstance(B, int) or isinstance(B, bool):
            raise ValueError(f"meta G is None: a star budget is built only by the wide arm, with an int B (got {B!r})")
        c_k, c_v = B * d_k // (d_k + d_v), B * d_v // (d_k + d_v)
        gbl_k = gbl_v = [[list(range(H))]] * L
    else:
        c_k, c_v = (G * d_k, G * d_v) if wide else (d_k, d_v)
        gbl_k, gbl_v = _groups_for(meta, H, G, L, wide=wide)
    for li, lay in enumerate(get_layers(model)):
        attn = _host_attn(lay)
        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        groups_k, groups_v = gbl_k[li], gbl_v[li]
        attn.q_proj = _empty_linear(dm, H * d_k, dtype, device)
        attn.bk_proj = _empty_linear(dm, len(groups_k) * c_k, dtype, device)
        attn.bv_proj = _empty_linear(dm, len(groups_v) * c_v, dtype, device)
        attn.Ck = nn.Parameter(torch.empty(H, d_k, c_k, dtype=dtype, device=device))
        attn.Cv = nn.Parameter(torch.empty(H, d_v, c_v, dtype=dtype, device=device))
        del attn.k_proj, attn.v_proj
        attn.n_kv_heads, attn.n_kv_groups = len(groups_k), H // len(groups_k)
        attn.n_kv_heads_v, attn.n_kv_groups_v = len(groups_v), H // len(groups_v)
        attn.c_k, attn.c_v = c_k, c_v
        attn.register_buffer("head_to_group_k", torch.tensor(head_group(groups_k, H), dtype=torch.long,
                                                             device=device), persistent=False)
        attn.register_buffer("head_to_group_v", torch.tensor(head_group(groups_v, H), dtype=torch.long,
                                                             device=device), persistent=False)
        _finish_twin(attn, model, _OptimalMixin, H)
    return dict(L=L, H=H, G=G, G_v=G_v, c_k=c_k, c_v=c_v, wide=wide, d_k=d_k, dm=dm,
                cache_width=len(gbl_k[0]) * c_k)

def _skeleton_decoupled(model, meta):
    L, H, d_k, d_v, dm, r = geometry(model)
    G, B = meta["G"], meta.get("B")
    if G is not None and (not isinstance(G, int) or isinstance(G, bool)):
        raise TypeError(f"meta G must be an int or None, got {G!r} ({type(G).__name__}).")
    budget = B if B is not None else (None if G is None else G * (d_k + d_v))
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise TypeError(f"meta has neither an int B nor an int G (B {B!r}, G {G!r})")
    if G is not None and budget != G * (d_k + d_v):
        raise ValueError(f"meta B = {budget} != G*(d_k+d_v) = {G * (d_k + d_v)}")
    if r == 0 or r > d_k:
        raise ValueError(f"decoupled needs 0 < r <= d_k (r = {r}, d_k = {d_k}); at r = d_k the free block is "
                         f"empty and Ck is (H, 0, c_KV)")
    if meta.get("r") not in (None, r):
        raise ValueError(f"meta says r={meta['r']} but this base model has r={r}")
    if meta.get("groups_by_layer") is not None:
        raise ValueError("decoupled is one group over all heads and cannot carry a groups_by_layer")

    def _per_layer(v, default):
        if v is None:
            return [default] * L
        if isinstance(v, (list, tuple)):
            if len(v) != L:
                raise ValueError(f"sidecar width list has {len(v)} entries, model has {L} layers")
            return [int(x) for x in v]
        return [int(v)] * L

    d_k_R_l = _per_layer(meta.get("d_k_R"), r)
    c_KV_l = _per_layer(meta.get("c_KV"), None)
    if any(c is None for c in c_KV_l):
        c_KV_l = [budget - d for d in d_k_R_l]
    if max(d_k_R_l) > H * r:
        raise ValueError(f"d_k_R={max(d_k_R_l)} exceeds H*r = {H * r}")
    _bad = [(li, d, c) for li, (d, c) in enumerate(zip(d_k_R_l, c_KV_l)) if d + c != budget]
    if _bad:
        raise ValueError(f"cache width identity BROKEN at {len(_bad)} layer(s), first {_bad[0]}: d_k_R + c_KV "
                         f"must equal the budget {budget} at every layer")
    for li, lay in enumerate(get_layers(model)):
        d_kR_li, cKV_li = d_k_R_l[li], c_KV_l[li]
        attn = _host_attn(lay)
        dtype, device = attn.q_proj.weight.dtype, attn.q_proj.weight.device
        attn.q_proj = _empty_linear(dm, H * d_k, dtype, device)
        attn.bkv_proj = _empty_linear(dm, cKV_li, dtype, device)
        attn.bkr_proj = _empty_linear(dm, d_kR_li, dtype, device)
        attn.Ckr = nn.Parameter(torch.empty(H, r, d_kR_li, dtype=dtype, device=device))
        attn.Ck = nn.Parameter(torch.empty(H, d_k - r, cKV_li, dtype=dtype, device=device))
        attn.Cv = nn.Parameter(torch.empty(H, d_v, cKV_li, dtype=dtype, device=device))
        del attn.k_proj, attn.v_proj
        attn.c_KV, attn.d_k_R, attn.r = cKV_li, d_kR_li, r
        attn.w_kv = float(meta.get("w", 1.0))
        _finish_twin(attn, model, _DecoupledMixin, H)
    return dict(L=L, H=H, G=G, c_KV=c_KV_l[0], d_k_R=d_k_R_l[0], r=r, w=meta.get("w"), d_k=d_k, dm=dm,
                cache_width=c_KV_l[0] + d_k_R_l[0], d_k_R_by_layer=list(d_k_R_l),
                uniform_width=len(set(d_k_R_l)) == 1)

def _fp_names(model, names, extra=()):
    h = hashlib.sha256()
    for lay in get_layers(model):
        a = get_attn(lay)
        for name in names:
            p = getattr(a, name)
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
        for name in extra:
            h.update(getattr(a, name).detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def _fp_mha(model):
    return _fp_names(model, ("q_proj", "k_proj", "v_proj", "o_proj"))

def _fp_gqa(model):
    return _fp_names(model, ("q_proj", "k_proj", "v_proj", "o_proj"))

def _fp_optimal(model):
    return _fp_names(model, ("q_proj", "bk_proj", "bv_proj", "o_proj"), ("Ck", "Cv"))

def _fp_decoupled(model):
    return _fp_names(model, ("q_proj", "bkv_proj", "bkr_proj", "o_proj"), ("Ckr", "Ck", "Cv"))

def cache_tensors(model):
    out = []
    for li, lay in enumerate(get_layers(model)):
        a = get_attn(lay)
        if isinstance(a, DecoupledLatentAttention):
            names = ("bkv_proj", "bkr_proj")
        elif isinstance(a, OptimalGQAAttention):
            names = ("bk_proj", "bv_proj")
        elif isinstance(a, GQAAttention):
            names = ("k_proj", "v_proj")
        elif isinstance(a, LlamaMLAAttention):
            names = ("dkv", "kr")
        else:
            continue
        for nm in names:
            p = getattr(a, nm)
            out.append((f"{li}.{nm}.weight", p.weight))
            if p.bias is not None:
                out.append((f"{li}.{nm}.bias", p.bias))
    return out

def cache_fingerprint(model):
    ts = cache_tensors(model)
    if not ts:
        raise RuntimeError(
            "cache_fingerprint found NO shared-cache tensors -- no layer holds a GQAAttention or "
            "OptimalGQAAttention. The hash would be sha256 of nothing, a constant, and any gate "
            "comparing it would pass without measuring anything. Refusing to return it.")
    n_lay = len(get_layers(model))
    if len(ts) % n_lay:
        raise RuntimeError(f"cache_fingerprint found {len(ts)} tensors across {n_lay} layers, which "
                           f"does not divide evenly -- some layers were not class-swapped, so this "
                           f"hash would cover only part of the cache.")
    h = hashlib.sha256()
    for name, t in ts:
        a = t.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]

def cache_width_of(model):
    out = set()
    kinds = set()
    for lay in get_layers(model):
        a = get_attn(lay)
        kinds.add(type(a).__name__)
        if isinstance(a, DecoupledLatentAttention):
            out.add((a.d_k_R, a.c_KV))
        elif isinstance(a, OptimalGQAAttention):
            out.add((a.n_kv_heads * a.c_k, a.n_kv_heads_v * a.c_v))
        elif isinstance(a, GQAAttention):
            out.add((a.n_kv_heads * a.head_size, a.n_kv_heads * a.head_size))
        elif isinstance(a, LlamaMLAAttention):
            out.add((a.w, a.d_c))
    if len(out) != 1:
        if kinds != {"DecoupledLatentAttention"}:
            raise RuntimeError(
                f"layers disagree on the cached width: {sorted(out)}, across attention classes "
                f"{sorted(kinds)}. Only the decoupled arch may vary its rotary/content split by "
                f"layer; this model is not purely decoupled, so differently-shaped layers mean the "
                f"skeleton and the checkpoint disagree about what was built.")
        totals = {a + b for a, b in out}
        if len(totals) != 1:
            raise RuntimeError(
                f"layers disagree on the TOTAL cached width: {sorted(totals)}. A per-layer rotary "
                f"split is legitimate; a per-layer TOTAL is not -- it would mean different layers "
                f"were built against different budgets, and every cross-arm comparison at this G "
                f"would be void with nothing to show it. Widths seen: {sorted(out)}")
        ks = sorted(a for a, _ in out)
        med = ks[len(ks) // 2]
        return (med, totals.pop() - med)
    return out.pop()

def train_everything_(model):
    plan, kinds = [], set()
    layers = get_layers(model)
    for lay in layers:
        a = get_attn(lay)
        if isinstance(a, DecoupledLatentAttention):
            plan.append(((a.bkv_proj, a.bkr_proj), f"decoupled[{a.c_KV}+{a.d_k_R}]"))
        elif isinstance(a, OptimalGQAAttention):
            plan.append(((a.bk_proj, a.bv_proj), f"optimal[{a.n_kv_heads}x{a.c_k}]"))
        elif isinstance(a, GQAAttention):
            plan.append(((a.k_proj, a.v_proj), f"gqa[G={a.n_kv_heads}]"))
        elif isinstance(a, LlamaMLAAttention):
            plan.append(((a.dkv, a.kr), f"mla[{a.rotary_mode} w={a.w} d_c={a.d_c}]"))
    if not plan:
        raise RuntimeError("no restructured attention module found -- this model is not one of the "
                           "converted architectures. Refusing to train: nothing here is the arm it "
                           "claims to be. (If this was the teacher, it is still frozen.)")
    if len(plan) != len(layers):
        raise RuntimeError(f"only {len(plan)} of {len(layers)} layers hold a restructured attention "
                           f"module. A partial class swap means the other layers are not the "
                           f"architecture this arm claims, and its cache budget is not what it logs.")
    for p in model.parameters():
        p.requires_grad_(True)
    n = 0
    for mods, kind in plan:
        kinds.add(kind)
        for m in mods:
            for p in m.parameters():
                assert p.requires_grad, "a cache tensor is still frozen after train_everything_"
                n += 1
    return n, sorted(kinds)

def read_meta(path, name=META_NAME):
    with open(os.path.join(path, name), "r", encoding="utf-8") as f:
        return json.load(f)

def _raw_state_dict(path):
    from safetensors.torch import load_file
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"no model.safetensors and no .index.json under {path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(path, shard)))
    return state

def _reconcile_legacy_keys(state, model):
    aliases = {"embed_out.weight": "lm_head.weight"}
    keys = set(model.state_dict().keys())
    for on_disk, live in aliases.items():
        if on_disk in state and on_disk not in keys and live in keys:
            state[live] = state.pop(on_disk)
    return state

def load_ckpt(path, device="cpu", expect_arm=None, verbose=True, meta_name=META_NAME):
    meta = read_meta(path, meta_name)
    for k in ("arch", "arm", "base_model", "fingerprint"):
        if k not in meta:
            raise ValueError(f"{path}: meta missing required key {k!r}")
    arch, arm = meta["arch"], meta.get("arm")
    if arch not in SKELETON:
        raise ValueError(f"{path}: unknown arch {arch!r} (known: {sorted(SKELETON)})")
    if expect_arm is not None and arm != expect_arm:
        raise RuntimeError(f"{path}: meta says arm={arm!r}, expected {expect_arm!r}. The arm string is "
                           f"load-bearing -- 'optwide' is what selects the wide shapes.")
    if meta["base_model"] != MODEL:
        raise RuntimeError(f"{path}: built from {meta['base_model']!r}, but this notebook is {MODEL!r}")
    model = AutoModelForCausalLM.from_pretrained(meta["base_model"], revision=MODEL_REVISION,
                                                 dtype=torch.float32, attn_implementation="eager")
    _grid_expand_gqa(model)
    impl = getattr(model.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention fell back to {impl!r}; these modules expand K/V themselves and "
                           f"only the eager path is correct")
    model.config.use_cache = False

    step = SKELETON[arch]
    info = step(model, meta) if step is not None else {}

    state = _reconcile_legacy_keys(_raw_state_dict(path), model)
    _tied = dict(getattr(model, "_tied_weights_keys", None) or {})
    if not _tied:
        raise RuntimeError(f"{path}: this model ties its embeddings but transformers names no tie")
    for _al, _sr in _tied.items():
        if _al not in state and _sr in state:
            state[_al] = state[_sr]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        have, want = set(state), set(model.state_dict())
        raise RuntimeError(
            f"{path}: strict state_dict load failed.\n"
            f"  missing from the file ({len(want - have)}): {sorted(want - have)[:12]}\n"
            f"  unexpected in the file ({len(have - want)}): {sorted(have - want)[:12]}\n"
            f"  This usually means the installed transformers names a parameter differently from the "
            f"version that WROTE the checkpoint. ⛔ Do NOT relax strict=True: the mismatched tensors "
            f"would silently keep their random initialisation.\n  original: {e}") from e

    _pp = dict(model.named_parameters(remove_duplicate=False))
    for _al, _sr in _tied.items():
        if _pp[_al].data_ptr() != _pp[_sr].data_ptr():
            raise RuntimeError(f"{path}: the tie {_al} -> {_sr} did not survive the reload")
    fp = FINGERPRINT[arch](model)
    if fp != meta["fingerprint"]:
        raise RuntimeError(
            f"{path}: reload fingerprint mismatch -- saved {meta['fingerprint']}, reloaded {fp}. "
            f"The state_dict load silently dropped or misconnected a tensor.")
    if meta.get("state_fingerprint") and "_fp_state" in globals():
        st = globals()["_fp_state"](model)
        if st != meta["state_fingerprint"]:
            raise RuntimeError(
                f"{path}: FULL-STATE fingerprint mismatch -- saved {meta['state_fingerprint']}, "
                f"reloaded {st}, while the attention block matched. The damage is in the embeddings, "
                f"an MLP or a LayerNorm: the ~80% of the model no arch fingerprint covers.")
    if arch == "mla":
        check_alloc(model, meta)
        if not meta.get("mla_probe"):
            raise RuntimeError(
                f"{path}: an 'mla' sidecar carries no mla_probe. dt_checkpoint.save_checkpoint "
                f"requires it, so this directory was not written by the current pipeline -- and "
                f"without it nothing here checks the forward pass at all, only the tensors.")
        check_probe(model, meta["mla_probe"], verbose=True)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    if verbose:
        ck, cv = cache_width_of(model)
        if "c_KV" in info:
            extra = f" c_KV={info['c_KV']} d_k_R={info['d_k_R']} w={info['w']}"
            lbl = "rot/content"
        elif "rotary_mode" in info:
            extra = f" w={info['w']} d_c={info['d_c']} mode={info['rotary_mode']}"
            lbl = "rot/latent"
        elif "c_k" in info:
            extra, lbl = f" c_k={info['c_k']} wide={info['wide']}", "K/V"
        else:
            extra, lbl = "", "K/V"
        print(f"    loaded {arm}_G{meta.get('G')}  arch={arch}{extra}  cache {lbl} "
              f"width {ck}/{cv} (total {ck + cv})  fingerprint {fp} ✓")
    return model, meta, info

SKELETON = {"mha": None, "gqa": _skeleton_gqa, "optimal": _skeleton_optimal,
            "decoupled": _skeleton_decoupled, "mla": skeleton}
FINGERPRINT = {"mha": _fp_mha, "gqa": _fp_gqa, "optimal": _fp_optimal, "decoupled": _fp_decoupled,
               "mla": mla_fingerprint}

# %%
assert MODEL == GRID["hf_id"] and MODEL_REVISION == GRID["revision"]

def load_base(name=MODEL, revision=MODEL_REVISION, dtype=torch.float32):
    if revision is None:
        raise RuntimeError(f"{name}: no pinned revision. ⛔ Loading 'main' would make every number "
                           f"depend on whatever the repo holds today.")
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if GRID["model_type"] == "llama":
        tok.add_bos_token = False
    if GRID["model_type"] == "olmo":
        tok.add_bos_token = False
        tok.add_eos_token = False
    _p = "The cat sat on the mat. 12 + 7 = 19"
    if tok(_p).input_ids != tok(_p, add_special_tokens=False).input_ids:
        raise RuntimeError(f"{name}: the tokenizer still adds special tokens to a plain encode")
    m = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype,
                                             attn_implementation="eager")
    impl = getattr(m.config, "_attn_implementation", "unknown")
    if impl != "eager":
        raise RuntimeError(f"attention fell back to {impl!r}; the instrument needs eager")
    m.config.use_cache = False
    m.eval()
    return m, tok

teacher, tok = load_base()
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.to(DEV)

L, H, d_k, d_v, dm, r = geometry(teacher)
print(f"geometry   L={L} H={H} d_k={d_k} d_model={dm} r={r}  vocab={teacher.config.vocab_size}")
if r != GRID["r"]:
    raise RuntimeError(f"r={r}, expected {GRID['r']} from the registry. A wrong rotary width is SILENT -- it "
                       f"splits the rotary planes from the free block in the wrong place.")
print(f"params     {sum(p.numel() for p in teacher.parameters()):,}")
assert all(not p.requires_grad for p in teacher.parameters()), "the teacher must be frozen"
print(f"teacher    frozen, eval, use_cache=False, dtype={next(teacher.parameters()).dtype}")

for _g in G_SEEN:
    if H % _g:
        raise RuntimeError(f"a checkpoint claims G={_g}, which does not divide H={H}")
print(f"           every discovered G {G_SEEN} divides H={H}")

probe = tok("The capital of France is Paris, and the capital of Germany is",
            return_tensors="pt").input_ids.to(DEV)
with torch.no_grad():
    lg_teacher = teacher(probe).logits.double()
if not torch.isfinite(lg_teacher).all():
    raise RuntimeError("teacher logits are not finite -- check the dtype pin")
print(f"           teacher logits finite, range [{lg_teacher.min():.2f}, {lg_teacher.max():.2f}]")
TEACHER_FP = _fp_mha(teacher)
print(f"           teacher attention fingerprint {TEACHER_FP}  (re-checked after every branch)")

# %%
import io as _io, re as _re

FEASIBILITY_TASKS = ["lambada", "piqa", "arc_easy", "boolq", "siqa", "openbookqa", "hellaswag",
                     "induction", "kv_synth_N50", "arith_1d",
                     "arc_challenge", "winogrande", "mmlu"]
CHANCE = {"lambada": 0.0, "piqa": 0.50, "arc_easy": 0.25, "boolq": 0.50, "siqa": 1 / 3,
          "openbookqa": 0.25, "hellaswag": 0.25,
          "induction": 0.0, "kv_synth_N50": 0.0, "arith_1d": 0.098,
          "arc_challenge": 0.25, "winogrande": 0.50, "mmlu": 0.25}
PRIMARY_METRIC = {"openbookqa": "acc_norm", "hellaswag": "acc_norm"}
SIZES = {"lambada": (500, 500), "induction": (500, 500), "kv_synth_N50": (500, 500),
         "arith_1d": (500, 500), "piqa": (500, 500), "openbookqa": (500, 500),
         "siqa": (500, 500), "arc_easy": (500, 500), "hellaswag": (500, 500), "boolq": (500, 500),
         "arc_challenge": (500, 500), "winogrande": (500, 500), "mmlu": (500, 500)}

def primary_metric(task):
    return PRIMARY_METRIC.get(task, "acc")

def n_for(task, phase):
    return SIZES[task][0 if phase == "periodic" else 1]

def _dl(repo, fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")

def _parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()

def build_lambada():
    out = []
    with _io.open(_dl("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl"),
                  "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ctx, last = json.loads(line)["text"].strip().rsplit(" ", 1)
            out.append({"ctx": ctx, "conts": [last], "gold": 0})
    return out

def build_piqa():
    rows = _parquet(_dl("baber/piqa", "piqa_validation.parquet"))
    return [{"ctx": "Question: " + d["goal"] + "\nAnswer:", "conts": [d["sol1"], d["sol2"]],
             "gold": int(d["label"])} for d in rows]

def build_arc_easy():
    rows = _parquet(_dl("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"))
    num2let = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    letters = ["A", "B", "C", "D", "E"]
    out = []
    for d in rows:
        key = num2let.get(d["answerKey"], d["answerKey"])
        if key not in letters:
            continue
        gold, texts = letters.index(key), list(d["choices"]["text"])
        if gold >= len(texts):
            continue
        out.append({"ctx": "Question: " + d["question"] + "\nAnswer:", "conts": texts, "gold": gold})
    return out

def build_boolq():
    rows = _parquet(_dl("aps/super_glue", "boolq/validation-00000-of-00001.parquet"))
    return [{"ctx": f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:",
             "conts": ["no", "yes"], "gold": int(d["label"])} for d in rows]

def build_siqa():
    rows = _parquet(_dl("lighteval/siqa",
                        "data/validation-00000-of-00001-90bb85eab85fcfa8.parquet"))
    return [{"ctx": f"Question: {d['context']} {d['question']}\nAnswer:",
             "conts": [d["answerA"], d["answerB"], d["answerC"]],
             "gold": int(d["label"]) - 1} for d in rows]

def build_openbookqa():
    rows = _parquet(_dl("allenai/openbookqa", "main/test-00000-of-00001.parquet"))
    out = []
    for d in rows:
        labels, texts = list(d["choices"]["label"]), list(d["choices"]["text"])
        out.append({"ctx": d["question_stem"], "conts": texts, "gold": labels.index(d["answerKey"])})
    return out

def _hellaswag_preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    return _re.sub(r"\[.*?\]", "", text).replace("  ", " ")

def build_hellaswag():
    rows = _parquet(_dl("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"))
    out = []
    for d in rows:
        query = _hellaswag_preprocess(d["activity_label"] + ": " + d["ctx_a"] + " "
                                      + d["ctx_b"].capitalize())
        out.append({"ctx": query, "conts": [_hellaswag_preprocess(e) for e in d["endings"]],
                    "gold": int(d["label"])})
    return out

_SYNTH_TOKEN_MAX = 49151
_SYNTH_VALUE_LO, _SYNTH_VALUE_HI = 100, 500

def _synth_key_pool(min_chars=4, min_id=2000):
    out = []
    for i in range(min_id, _SYNTH_TOKEN_MAX + 1):
        s = tok.decode([i])
        if not s.startswith(" "):
            continue
        w = s[1:]
        if len(w) >= min_chars and w.isalpha() and w.islower() and w.isascii():
            out.append((i, w))
    return out

def build_induction(n_items=2000, seq_len=48, prefix=4, m=8, seed=0):
    ids = [i for i, _ in _synth_key_pool()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_items):
        s = [ids[j] for j in rng.choice(len(ids), size=seq_len, replace=False)]
        head = tok.decode(s[:prefix])
        target = tok.decode(s[prefix:prefix + m])
        out.append({"ctx": (tok.decode(s) + head).rstrip(), "conts": [target.lstrip()], "gold": 0})
    return out

def build_kv_synth_N50(n_items=2000, n_pairs=50, seed=0):
    words = [w for _, w in _synth_key_pool()]
    rng = np.random.default_rng(seed)
    values = list(range(_SYNTH_VALUE_LO, _SYNTH_VALUE_HI))
    out = []
    for _ in range(n_items):
        ks = rng.choice(len(words), size=n_pairs, replace=False)
        vs = rng.choice(len(values), size=n_pairs, replace=False)
        q = int(rng.integers(0, n_pairs))
        lines = "".join(f"The code for {words[k]} is {values[v]}.\n" for k, v in zip(ks, vs))
        out.append({"ctx": lines + f"The code for {words[ks[q]]} is",
                    "conts": [str(values[vs[q]])], "gold": 0})
    return out

def build_arith_1d(n_items=2000, k_shot=8, seed=0, op="plus"):
    rng = np.random.default_rng(seed)

    def one():
        a, b = int(rng.integers(0, 10)), int(rng.integers(0, 10))
        return a, b, a + b

    out = []
    for _ in range(n_items):
        shots = []
        for _ in range(k_shot):
            a, b, c = one()
            shots.append(f"\n\nQ: What is {a} {op} {b}?\n\nA: {c}")
        a, b, c = one()
        out.append({"ctx": "".join(shots) + f"\n\nQ: What is {a} {op} {b}?\n\nA:",
                    "conts": [str(c)], "gold": 0})
    return out

def build_arc_challenge():
    rows = _parquet(_dl("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"))
    num2let = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    letters = ["A", "B", "C", "D", "E"]
    out = []
    for d in rows:
        key = num2let.get(d["answerKey"], d["answerKey"])
        if key not in letters:
            continue
        gold, texts = letters.index(key), list(d["choices"]["text"])
        if gold >= len(texts):
            continue
        out.append({"ctx": "Question: " + d["question"] + "\nAnswer:", "conts": texts, "gold": gold})
    return out

def build_winogrande():
    rows = _parquet(_dl("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet"))
    out = []
    for d in rows:
        s = d["sentence"]
        i = s.index("_")
        cont = s[i + 1:].strip()
        out.append({"ctx": [s[:i] + d["option1"], s[:i] + d["option2"]], "conts": [cont, cont],
                    "gold": int(d["answer"]) - 1})
    return out

def build_mmlu():
    rows = _parquet(_dl("cais/mmlu", "all/test-00000-of-00001.parquet"))
    out = []
    for d in rows:
        ch = list(d["choices"])
        if len(ch) != 4:
            raise RuntimeError(f"mmlu item with {len(ch)} choices, not 4: {d['question'][:80]!r}")
        head = ("The following are multiple choice questions (with answers) about "
                + " ".join(d["subject"].split("_")) + ".\n\n")
        body = d["question"].strip() + "".join(f"\n{l}. {c}" for l, c in zip("ABCD", ch)) + "\nAnswer:"
        out.append({"ctx": head + body, "conts": ["A", "B", "C", "D"], "gold": int(d["answer"])})
    return out

BUILDERS = {"lambada": build_lambada, "piqa": build_piqa, "arc_easy": build_arc_easy,
            "boolq": build_boolq, "siqa": build_siqa, "openbookqa": build_openbookqa,
            "hellaswag": build_hellaswag, "induction": build_induction,
            "kv_synth_N50": build_kv_synth_N50, "arith_1d": build_arith_1d,
            "arc_challenge": build_arc_challenge, "winogrande": build_winogrande,
            "mmlu": build_mmlu}

REBUILD = False

print(f"task caches under {DATA_DIR}")
TASK_ITEMS = {}
for t in FEASIBILITY_TASKS:
    p = os.path.join(DATA_DIR, f"{t}.json")
    if os.path.exists(p) and not REBUILD:
        items = json.load(open(p, encoding="utf-8"))
        src = "cached"
    else:
        items = BUILDERS[t]()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f)
        src = "BUILT "
    TASK_ITEMS[t] = items
    nopt = sorted({len(i["conts"]) for i in items})
    clen = int(np.mean([len(i["ctx"][0] if isinstance(i["ctx"], list) else i["ctx"]) for i in items]))
    print(f"  {src} {t:<14}{len(items):>6} items  options {str(nopt):<10} mean ctx {clen:>5} chars  "
          f"chance {CHANCE[t]:.3f}  n_per/n_end {SIZES[t]}")

for t in FEASIBILITY_TASKS:
    have, want = len(TASK_ITEMS[t]), n_for(t, "endpoint")
    if have < want:
        raise RuntimeError(f"{t}: endpoint n is {want} but only {have} items exist -- the subsample "
                           f"would silently return {have} and be logged as {want}")
print("\n✔ every task has at least its endpoint n items")
for t in FEASIBILITY_TASKS:
    _bad = [k for k, it in enumerate(TASK_ITEMS[t])
            if isinstance(it["ctx"], list) and len(it["ctx"]) != len(it["conts"])]
    if _bad:
        raise RuntimeError(f"{t}: {len(_bad)} items have a ctx list whose length is not the number "
                           f"of options (first: item {_bad[0]})")
print("✔ every per-option ctx list has one context per option")

# %%
MAX_BATCH = 16
MAX_BATCH_TOKENS = 1536
MAX_LEN = 2048

def subsample(items, n, seed=0):
    if n is None or n >= len(items):
        return items, list(range(len(items)))
    idx = sorted(np.random.default_rng(seed).permutation(len(items))[:n].tolist())
    return [items[i] for i in idx], idx

def _encode(ctx, cont):
    whole = tok(ctx + cont, add_special_tokens=False)["input_ids"]
    c = tok(ctx, add_special_tokens=False)["input_ids"]
    n = len(c)
    if whole[:n] != c:
        n = 0
        while n < min(len(whole), len(c)) and whole[n] == c[n]:
            n += 1
    return whole[:n], whole[n:]

def _requests(items):
    reqs = []
    for i, it in enumerate(items):
        for j, cont in enumerate(it["conts"]):
            ctx = it["ctx"][j] if isinstance(it["ctx"], list) else it["ctx"]
            ce, ke = _encode(ctx, " " + cont)
            if len(ke) == 0:
                ke, ce = ce[-1:], ce[:-1]
            inp = (ce + ke)[-(MAX_LEN + 1):][:-1]
            reqs.append({"i": i, "j": j, "inp": inp, "cont": ke,
                         "inplen": len(inp), "contlen": len(ke)})
    return reqs

def _batches(reqs):
    order = sorted(range(len(reqs)), key=lambda k: -reqs[k]["inplen"])
    cur = []
    for k in order:
        T = reqs[k]["inplen"]
        if cur and (len(cur) + 1 > MAX_BATCH or (len(cur) + 1) * T > MAX_BATCH_TOKENS):
            yield cur
            cur = []
        cur.append(k)
    if cur:
        yield cur

@torch.no_grad()
def score(model, items):
    device = next(model.parameters()).device
    reqs = _requests(items)
    lls = [[0.0] * len(it["conts"]) for it in items]
    grd = [[False] * len(it["conts"]) for it in items]
    for batch in _batches(reqs):
        T = max(reqs[k]["inplen"] for k in batch)
        inp = torch.zeros((len(batch), T), dtype=torch.long)
        for b, k in enumerate(batch):
            inp[b, :reqs[k]["inplen"]] = torch.tensor(reqs[k]["inp"], dtype=torch.long)
        out = model(inp.to(device)).logits.float()
        for b, k in enumerate(batch):
            rq = reqs[k]
            sl = torch.log_softmax(out[b, rq["inplen"] - rq["contlen"]:rq["inplen"], :], dim=-1)
            ct = torch.tensor(rq["cont"], dtype=torch.long, device=device)
            grd[rq["i"]][rq["j"]] = bool((sl.argmax(dim=-1) == ct).all())
            lls[rq["i"]][rq["j"]] = float(sl.gather(1, ct.unsqueeze(-1)).sum())
        del out
    return lls, grd

def evaluate_mc(items, lls, grd):
    assert all(len(it["conts"]) > 1 for it in items), \
        "evaluate_mc got single-continuation items -- argmax over one option always equals gold, so " \
        "accuracy would read 1.0 regardless of the model. Those go to greedy exact match."
    n = len(items)
    acc = np.zeros(n, dtype=bool); accn = np.zeros(n, dtype=bool); marg = np.zeros(n)
    for i, it in enumerate(items):
        g = it["gold"]
        v = np.array(lls[i], dtype=float)
        nb = np.array([max(len(c), 1) for c in it["conts"]], dtype=float)
        acc[i] = int(np.argmax(v)) == g
        accn[i] = int(np.argmax(v / nb)) == g
        other = np.delete(v, g)
        marg[i] = v[g] - (other.max() if len(other) else 0.0)
    return {"acc": acc, "acc_norm": accn, "margin": marg}

def evaluate_greedy(items, lls, grd):
    acc = np.array([bool(g[0]) for g in grd], dtype=bool)
    ll = np.array([v[0] for v in lls], dtype=float)
    return {"acc": acc, "acc_norm": acc, "margin": ll, "ll": ll, "ppl": float(np.exp(-ll.mean()))}

def evaluate(items, lls, grd):
    single = bool(items) and len(items[0]["conts"]) == 1
    return (evaluate_greedy if single else evaluate_mc)(items, lls, grd)

def summarise(m):
    out = {"n": int(len(m["acc"])), "acc": float(m["acc"].mean()),
           "acc_norm": float(m["acc_norm"].mean()), "margin": float(m["margin"].mean())}
    if "ppl" in m:
        out["ppl"] = m["ppl"]
    return out

def run_battery(model, phase, tasks=FEASIBILITY_TASKS, verbose=True):
    was_training = model.training
    model.eval()
    out = {}
    for t in tasks:
        n = n_for(t, phase)
        items, _idx = subsample(TASK_ITEMS[t], n, seed=0)
        t0 = time.time()
        lls, grd = score(model, items)
        s = summarise(evaluate(items, lls, grd))
        s.update(task=t, phase=phase, chance=CHANCE[t], primary=primary_metric(t),
                 primary_value=s[primary_metric(t)], secs=round(time.time() - t0, 1))
        out[t] = s
        if verbose:
            pv, ch = s["primary_value"], CHANCE[t]
            print(f"      {t:<14} n={s['n']:<5} {s['primary']}={pv:.4f} (chance {ch:.3f}, "
                  f"+{pv - ch:+.4f})  margin {s['margin']:+8.3f}  {s['secs']:>5.1f}s")
    if was_training:
        model.train()
    return out

T0_EXPECTED = {
    "optwide_G8":         {"lambada": 0.527, "piqa": 0.662, "arc_easy": 0.554, "siqa": 0.396,
                           "openbookqa": 0.290, "induction": 0.682, "kv_synth_N50": 0.696,
                           "arith_1d": 0.230},
    "optwide_G4":         {"lambada": 0.438, "piqa": 0.625, "arc_easy": 0.503, "siqa": 0.367,
                           "openbookqa": 0.288, "induction": 0.073, "kv_synth_N50": 0.126,
                           "arith_1d": 0.121},
    "optimal_G8":         {"lambada": 0.346, "piqa": 0.609, "arc_easy": 0.462, "siqa": 0.342,
                           "openbookqa": 0.260, "induction": 0.000, "kv_synth_N50": 0.010,
                           "arith_1d": 0.085},
    "pool_G8":            {"lambada": 0.000, "piqa": 0.539, "arc_easy": 0.303, "siqa": 0.341,
                           "openbookqa": 0.294, "induction": 0.000, "kv_synth_N50": 0.000,
                           "arith_1d": 0.000},
    "gauge3B_G8":         {"lambada": 0.000, "piqa": 0.529, "arc_easy": 0.321, "siqa": 0.347,
                           "openbookqa": 0.268, "induction": 0.000, "kv_synth_N50": 0.000,
                           "arith_1d": 0.000},
    "procrustes_simk_G8": {"lambada": 0.000, "piqa": 0.559, "arc_easy": 0.312, "siqa": 0.349,
                           "openbookqa": 0.268, "induction": 0.000, "kv_synth_N50": 0.000,
                           "arith_1d": 0.000},
}
T0_EXPECTED = {}
T0_TOL = 0.05

def check_t0(branch_key, got):
    exp = T0_EXPECTED.get(branch_key)
    if exp is None:
        return f"no t=0 reference for {branch_key} -- not gated", None
    rows, worst = [], 0.0
    for t, want in exp.items():
        if t not in got:
            continue
        have = got[t]["primary_value"]
        dev = abs(have - want)
        worst = max(worst, dev)
        rows.append((t, want, have, dev))
    lines = [f"    t=0 ACCEPTANCE vs the 2026-09-10 run ({branch_key}), tol {T0_TOL}, on each task's "
             f"PRIMARY metric:"]
    for t, want, have, dev in rows:
        flag = "OK " if dev <= T0_TOL else "⚠ OFF"
        lines.append(f"      {flag} {t:<14} ({primary_metric(t):<8}) was {want:.3f}  now {have:.3f}  "
                     f"|dev| {dev:.3f}")
    if worst > T0_TOL:
        lines.append(f"    ⚠ ⚠ worst deviation {worst:.3f} EXCEEDS {T0_TOL}. Either the port is wrong "
                     f"or this is a different checkpoint.")
        lines.append(f"       If induction and kv_synth_N50 are the tasks that moved and the MC tasks "
                     f"did not, suspect the OptimalGQAAttention operation order: correction must come "
                     f"BEFORE rotation. Nothing downstream is interpretable until this is resolved.")
    else:
        lines.append(f"    ✔ reproduces the previous run within {T0_TOL} (worst {worst:.3f}) -- the "
                     f"loader, the skeleton and the forward order are all confirmed.")
    return "\n".join(lines), worst

# %%
def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

# %%
def _fp_state(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        t = sd[k]
        h.update(k.encode("utf-8"))
        h.update(f"{tuple(t.shape)}|{t.dtype}".encode("utf-8"))
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    for li, lay in enumerate(get_layers(model)):
        a = get_attn(lay)
        for nm in ("head_to_group", "head_to_group_k", "head_to_group_v"):
            b = getattr(a, nm, None)
            if b is not None:
                h.update(f"{li}.{nm}".encode("utf-8"))
                h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

# %%
import gc
import gzip
import traceback

EVAL_TASKS = ('lambada', 'arc_easy', 'piqa', 'hellaswag', 'openbookqa', 'winogrande', 'arc_challenge', 'boolq', 'siqa', 'mmlu')
EVAL_SCHEMA = "full_eval-2026-10-25.1"
EVAL_DIR = os.path.join(os.path.dirname(RUNS_ROOT), "full_eval")
EVAL_SCRIPT_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
REPRO_TOL = 0.02
PPL_TOL = 0.02
ORIGINAL_REF_SRC = "SmolLM-360M's logged 500-item battery: the 5M run's unconverted (mha) branch at t = 0 (grid-sm360m-SmolLM2-1.7B-float32-5M-lr2e-05-s42)"
ORIGINAL_REF = {
 "lambada": {
  "primary": "acc",
  "primary_value": 0.464,
  "n": 500
 },
 "arc_easy": {
  "primary": "acc",
  "primary_value": 0.714,
  "n": 500
 },
 "piqa": {
  "primary": "acc",
  "primary_value": 0.714,
  "n": 500
 },
 "hellaswag": {
  "primary": "acc_norm",
  "primary_value": 0.526,
  "n": 500
 },
 "openbookqa": {
  "primary": "acc_norm",
  "primary_value": 0.376,
  "n": 500
 },
 "winogrande": {
  "primary": "acc",
  "primary_value": 0.556,
  "n": 500
 },
 "arc_challenge": {
  "primary": "acc",
  "primary_value": 0.316,
  "n": 500
 },
 "boolq": {
  "primary": "acc",
  "primary_value": 0.584,
  "n": 500
 },
 "siqa": {
  "primary": "acc",
  "primary_value": 0.422,
  "n": 500
 },
 "mmlu": {
  "primary": "acc",
  "primary_value": 0.244,
  "n": 500
 }
}
assert set(EVAL_TASKS) <= set(FEASIBILITY_TASKS) and set(EVAL_TASKS) <= set(ORIGINAL_REF)

os.makedirs(EVAL_DIR, exist_ok=True)
_probe_e = os.path.join(EVAL_DIR, f".write_probe.{os.environ.get('PBS_JOBID', os.getpid())}")
with open(_probe_e, "w") as _f:
    _f.write("probe")
with open(_probe_e) as _f:
    assert _f.read() == "probe", "Drive read back something other than what was written"
os.remove(_probe_e)
print(f"\nfull-set results -> {EVAL_DIR}  (writable)")

_man = json.load(open(os.path.join(OUT_DIR, "_manifest.json"), encoding="utf-8"))
if _man.get("stamp") != STAMP or _man.get("registry_sha") != REGISTRY_SHA:
    raise RuntimeError(f"{OUT_DIR}/_manifest.json is stamp {_man.get('stamp')!r}, registry {_man.get('registry_sha')!r}; "
                       f"this script is {STAMP!r}, {REGISTRY_SHA!r}")
ITEMS_SHA = {}
for _t in EVAL_TASKS:
    ITEMS_SHA[_t] = _sha(TASK_ITEMS[_t])
    if ITEMS_SHA[_t] != _man["battery"][_t]["items_sha"]:
        raise RuntimeError(f"{_t}: the cached items hash to {ITEMS_SHA[_t][:16]}, the stamp's manifest says "
                           f"{_man['battery'][_t]['items_sha'][:16]}. These are not the items the models were measured on.")
print(f"items      the {len(EVAL_TASKS)} tasks' full lists are the manifest's (items_sha): "
      + ", ".join(f"{t} {len(TASK_ITEMS[t])}" for t in EVAL_TASKS))

def _branch_rows(branch, run_id):
    f = os.path.join(OUT_DIR, f"kd_{STAMP}_{branch}.jsonl")
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("run_id") == run_id]
    out = []
    for th in (0, MAX_TOKENS):
        hit = [r for r in rows if r.get("battery") and r.get("token_threshold") == th]
        if len(hit) != 1 or hit[0].get("battery_phase") != "endpoint":
            raise RuntimeError(f"{os.path.basename(f)}: run {run_id} has {len(hit)} battery row(s) at {th:,} tokens "
                               f"(phase {[r.get('battery_phase') for r in hit]}); expected one endpoint row")
        out.append(hit[0])
    return f, out[0], out[1]

PLAN = [{"id": "original", "kind": "original", "arm": None, "G": None, "B": None, "branch": None,
         "path": f"{MODEL}@{MODEL_REVISION}", "fingerprint": TEACHER_FP, "ref": ORIGINAL_REF, "ref_src": ORIGINAL_REF_SRC}]
_t0_plan = []
for _arm, _G, _B, _toks in GRID_CELLS:
    _branch = f"{_arm}_G{_G}_s{SEED_TRAIN}"
    _mdir = os.path.join(OUT_DIR, "models", f"kd_{STAMP}_{_branch}")
    _sc = read_meta(_mdir)
    _bad = [k for k, v in (("trained", True), ("branch", _branch), ("stamp", STAMP), ("registry_sha", REGISTRY_SHA),
                           ("arm", _arm), ("original", MODEL), ("original_fingerprint", TEACHER_FP))
            if _sc.get(k) != v]
    if _bad:
        raise RuntimeError(f"{_mdir}: its sidecar disagrees with this stamp on {_bad}")
    _f, _r0, _rend = _branch_rows(_branch, _sc["run_id"])
    _idir = CKPTS[(_arm, _G)]
    _im = read_meta(_idir)
    if (os.path.basename(_idir) != _r0["ckpt"] or _im["fingerprint"] != _r0["ckpt_fingerprint"]
            or _sc["init_fingerprint"] != _im["fingerprint"] or _im.get("trained")):
        raise RuntimeError(f"{_branch}: the init {os.path.basename(_idir)} (fingerprint {_im['fingerprint']}) is not the "
                           f"one run {_sc['run_id']} started from ({_r0['ckpt']}, {_r0['ckpt_fingerprint']}; the saved "
                           f"model says {_sc['init_fingerprint']})")
    if _sc["tokens"] != _rend["tokens"]:
        raise RuntimeError(f"{_branch}: the saved model is at {_sc['tokens']:,} tokens, its last battery row at "
                           f"{_rend['tokens']:,}")
    _cell = {"arm": _arm, "G": _G, "B": _B, "branch": _branch, "run_id": _sc["run_id"]}
    PLAN.append(dict(_cell, id=f"{_arm}_G{_G}_50M", kind="50M", path=_mdir, fingerprint=_sc["fingerprint"],
                     state_fingerprint=_sc["state_fingerprint"], init_fingerprint=_im["fingerprint"], ref=_rend["battery"],
                     ref_src=f"{os.path.basename(_f)}, run {_sc['run_id']}, row at {_rend['tokens']:,} tokens"))
    _t0_plan.append(dict(_cell, id=f"{_arm}_G{_G}_t0", kind="t0", path=_idir, fingerprint=_im["fingerprint"],
                         init_fingerprint=_im["fingerprint"], ref=_r0["battery"],
                         ref_src=f"{os.path.basename(_f)}, run {_sc['run_id']}, row at 0 tokens"))
PLAN += _t0_plan
assert len({e["id"] for e in PLAN}) == len(PLAN) == 1 + 2 * len(GRID_CELLS)
print(f"plan       {len(PLAN)} models: the original, {len(GRID_CELLS)} saved 50M models, {len(GRID_CELLS)} t = 0 inits; "
      f"every 50M sidecar names this stamp, its run's rows and its init, and the original's fingerprint {TEACHER_FP}")

def _paths(eid):
    return os.path.join(EVAL_DIR, f"{eid}.json"), os.path.join(EVAL_DIR, f"{eid}.items.json.gz")

def _atomic(path, data):
    tmp = f"{path}.tmp.{os.environ.get('PBS_JOBID', os.getpid())}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    with open(path, "rb") as f:
        if f.read() != data:
            raise RuntimeError(f"{path}: read back something other than what was written")

def _finished(e):
    js, it = _paths(e["id"])
    if not os.path.exists(js):
        return None
    j = json.load(open(js, encoding="utf-8"))
    if j.get("schema") != EVAL_SCHEMA or j.get("eval_script_sha256") != EVAL_SCRIPT_SHA:
        raise RuntimeError(f"{js} was written by another version of this script (schema {j.get('schema')}, script "
                           f"{str(j.get('eval_script_sha256'))[:12]}; this is {EVAL_SCRIPT_SHA[:12]}). Move {EVAL_DIR}'s "
                           f"files aside, do not delete them, and tell the authors.")
    if j.get("model_id") != e["id"] or j.get("fingerprint") != e["fingerprint"]:
        raise RuntimeError(f"{js} is {j.get('model_id')} / {j.get('fingerprint')}, expected {e['id']} / {e['fingerprint']}")
    if not os.path.exists(it) or sha256_file(it) != j.get("items_sha256"):
        raise RuntimeError(f"{it} is missing or does not match {os.path.basename(js)}'s items_sha256: tell the authors")
    return j

def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _rerun_battery(model, e):
    now = run_battery(model, "endpoint", tasks=EVAL_TASKS, verbose=False)
    per = {}
    for t in EVAL_TASKS:
        ref = e["ref"].get(t)
        if ref is None or ref.get("primary") != primary_metric(t):
            raise RuntimeError(f"{e['id']}: the logged battery has no {primary_metric(t)} for {t} ({e['ref_src']})")
        r = {"now": now[t]["primary_value"], "logged": ref["primary_value"], "n": now[t]["n"],
             "d": now[t]["primary_value"] - ref["primary_value"]}
        if "ppl" in now[t] and ref.get("ppl"):
            r.update(ppl_now=now[t]["ppl"], ppl_logged=ref["ppl"], ppl_rel=now[t]["ppl"] / ref["ppl"] - 1.0)
        per[t] = r
    max_d = max(abs(r["d"]) for r in per.values())
    max_ppl = max((abs(r["ppl_rel"]) for r in per.values() if "ppl_rel" in r), default=0.0)
    ok = max_d <= REPRO_TOL and max_ppl <= PPL_TOL
    return {"source": e["ref_src"], "per_task": per, "max_abs_d": max_d, "max_ppl_rel": max_ppl,
            "n_exact": sum(r["d"] == 0 for r in per.values()), "tol": REPRO_TOL, "ppl_tol": PPL_TOL,
            "verdict": "REPRODUCES" if ok else "DIFFERS"}, now

def _full_sets(model, e, now):
    tasks, per_item = {}, {}
    for t in EVAL_TASKS:
        items = TASK_ITEMS[t]
        t0 = time.time()
        lls, grd = score(model, items)
        m = evaluate(items, lls, grd)
        s = summarise(m)
        p = primary_metric(t)
        x = m[p].astype(np.float64)
        _, idx = subsample(items, n_for(t, "endpoint"), seed=0)
        sub = {"acc": float(m["acc"][idx].mean()), "acc_norm": float(m["acc_norm"][idx].mean())}
        if "ppl" in m:
            sub["ppl"] = float(np.exp(-m["ll"][idx].mean()))
        s.update(task=t, primary=p, primary_value=s[p], stderr=float(x.std(ddof=1) / math.sqrt(len(x))),
                 chance=CHANCE[t], items_sha=ITEMS_SHA[t], secs=round(time.time() - t0, 1), sub500=sub,
                 sub500_vs_battery=sub[p] - now[t]["primary_value"])
        tasks[t] = s
        per_item[t] = {"ll": lls, "greedy": grd, "acc": m["acc"].astype(int).tolist(),
                       "acc_norm": m["acc_norm"].astype(int).tolist()}
        print(f"      {t:<14} n={s['n']:<6} {p:<8} {s[p]:.4f} ± {s['stderr']:.4f}   (its 500: {sub[p]:.4f}; "
              f"logged {e['ref'][t]['primary_value']:.4f})" + (f"   ppl {s['ppl']:.3f}" if "ppl" in s else "")
              + f"   {s['secs']:>6.1f}s")
    return tasks, per_item

def _write(e, rep, tasks, per_item, secs):
    js, it = _paths(e["id"])
    blob = gzip.compress(json.dumps({"model_id": e["id"], "tasks": per_item}, separators=(",", ":")).encode("utf-8"),
                         mtime=0)
    _atomic(it, blob)
    j = {"schema": EVAL_SCHEMA, "tag": MODEL_TAG, "model_id": e["id"], "kind": e["kind"], "arm": e["arm"], "G": e["G"],
         "B": e["B"], "branch": e["branch"], "run_id": e.get("run_id"),
         "path": os.path.relpath(e["path"], DRIVE_ROOT) if e["kind"] != "original" else e["path"],
         "fingerprint": e["fingerprint"], "state_fingerprint": e.get("state_fingerprint"),
         "init_fingerprint": e.get("init_fingerprint"), "model": MODEL, "model_revision": MODEL_REVISION,
         "stamp": STAMP, "registry_sha": REGISTRY_SHA, "eval_script_sha256": EVAL_SCRIPT_SHA,
         "pbs_jobid": os.environ.get("PBS_JOBID"), "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "device": torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else str(DEV),
         "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
         "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
         "scorer": {"max_batch": MAX_BATCH, "max_batch_tokens": MAX_BATCH_TOKENS, "max_len": MAX_LEN},
         "tasks": tasks, "battery500": rep, "secs": round(secs, 1),
         "items_file": os.path.basename(it), "items_sha256": hashlib.sha256(blob).hexdigest()}
    _atomic(js, (json.dumps(j, indent=1) + "\n").encode("utf-8"))
    return j

DONE, FAILED = {}, []
print(f"\n{'=' * 118}\nfull datasets: {len(PLAN)} models x {len(EVAL_TASKS)} tasks "
      f"({sum(len(TASK_ITEMS[t]) for t in EVAL_TASKS):,} items per model)\n{'=' * 118}")
for _k, e in enumerate(PLAN):
    have = _finished(e)
    if have is not None:
        DONE[e["id"]] = have
        print(f"[{_k + 1}/{len(PLAN)}] {e['id']}: already done ({have['evaluated_at']}, job {have['pbs_jobid']}), skipped")
        continue
    print(f"\n[{_k + 1}/{len(PLAN)}] {e['id']}  ({e['path']})")
    model, _t_model = None, time.time()
    try:
        if e["kind"] == "original":
            model = teacher
        else:
            model, _meta, _ = load_ckpt(e["path"], device=DEV, expect_arm=e["arm"], verbose=True)
            if _meta["fingerprint"] != e["fingerprint"]:
                raise RuntimeError(f"{e['path']}: loaded fingerprint {_meta['fingerprint']}, planned {e['fingerprint']}")
        rep, now = _rerun_battery(model, e)
        print(f"    the run's 500-item battery again: {rep['verdict']}  (max |d| {rep['max_abs_d']:.4f} over "
              f"{len(EVAL_TASKS)} tasks, {rep['n_exact']} exact; lambada ppl {rep['max_ppl_rel']:+.2e} relative)  "
              f"against {e['ref_src']}")
        if rep["verdict"] != "REPRODUCES":
            for t, r in rep["per_task"].items():
                print(f"      {t:<14} now {r['now']:.4f}  logged {r['logged']:.4f}  d {r['d']:+.4f}")
        tasks, per_item = _full_sets(model, e, now)
        DONE[e["id"]] = _write(e, rep, tasks, per_item, time.time() - _t_model)
        print(f"    written {os.path.basename(_paths(e['id'])[0])} + items  ({time.time() - _t_model:.0f}s for this model)")
    except Exception as ex:
        traceback.print_exc()
        FAILED.append((e["id"], f"{type(ex).__name__}: {str(ex)[:400]}"))
        print(f"    ⛔ {e['id']} FAILED; nothing written for it. The job goes on to the next model and exits 1 at the end.")
    finally:
        if model is not None and model is not teacher:
            del model
        gc.collect()
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

_short = {"lambada": "lamb", "arc_easy": "arc_e", "piqa": "piqa", "hellaswag": "hella", "openbookqa": "obqa",
          "winogrande": "wino", "arc_challenge": "arc_c", "boolq": "boolq", "siqa": "siqa", "mmlu": "mmlu"}
print(f"\n{'=' * 118}\n{MODEL_TAG}: primary metric on the FULL sets (acc_norm for openbookqa and hellaswag, acc "
      f"elsewhere); lambada ppl last\n")
print(f"{'model':<36}" + "".join(f"{_short[t]:>7}" for t in EVAL_TASKS) + f"{'ppl':>11}   500-item re-run")
for e in PLAN:
    j = DONE.get(e["id"])
    if j is None:
        print(f"{e['id']:<36}  -- not done --")
        continue
    print(f"{e['id']:<36}" + "".join(f"{j['tasks'][t]['primary_value']:>7.3f}" for t in EVAL_TASKS)
          + f" {j['tasks']['lambada']['ppl']:>10.2f}   {j['battery500']['verdict']} (max |d| "
          f"{j['battery500']['max_abs_d']:.3f})")
DIFFERS = [i for i, j in DONE.items() if j["battery500"]["verdict"] != "REPRODUCES"]
MISSING = [e["id"] for e in PLAN if e["id"] not in DONE]
print(f"\n{len(DONE)} of {len(PLAN)} models done; {len(DIFFERS)} DIFFERS; {len(FAILED)} failed in this job")
for i, msg in FAILED:
    print(f"  FAILED  {i}: {msg}")
if FAILED or DIFFERS or MISSING:
    raise RuntimeError(f"{MODEL_TAG}: {len(FAILED)} failed, {len(DIFFERS)} DIFFERS ({DIFFERS}), {len(MISSING)} not done. "
                       f"Every result that was written is kept; send the authors this log.")
print(f"✅ ALL {len(PLAN)} MODELS OF {MODEL_TAG} DONE, every one reproducing its logged 500-item battery")
