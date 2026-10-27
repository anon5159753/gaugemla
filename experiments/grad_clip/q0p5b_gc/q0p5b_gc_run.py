# q0p5b_gc run: distils each branch toward the teacher, logging KL and the task battery; saves each final model.
# Run by colab/launch.py from a Colab notebook, which mounts Drive, clones the repository to /content/repo
# and passes BRANCH, TEST_TOKENS as environment variables.

import os

DRIVE_ROOT = "/content/drive"
CKPT_ROOT  = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/q0p5b_gc/init/ckpt"
RUNS_ROOT  = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/q0p5b_gc/runs"
DATA_DIR   = DRIVE_ROOT + "/MyDrive/gaugemla/grid/_tok/33ea6c72ebb9/battery13"

if not os.path.isdir(DRIVE_ROOT + "/MyDrive"):
    raise RuntimeError(
        f"{DRIVE_ROOT}/MyDrive does not exist after drive.mount(). The mount did not complete -- "
        "usually the authorisation popup was dismissed or a different account was chosen. "
        "Re-run this cell and complete the prompt. ⛔ There is no alternative save path.")

for _name, _d in (("CKPT_ROOT", CKPT_ROOT), ("RUNS_ROOT", RUNS_ROOT), ("DATA_DIR", DATA_DIR)):
    if not _d.startswith(DRIVE_ROOT + "/"):
        raise RuntimeError(f"{_name} = {_d!r} is not under {DRIVE_ROOT}. ⛔ Drive is the only "
                           f"permitted location; fix the path, do not work around it.")

for _d in (CKPT_ROOT, RUNS_ROOT, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

for _d in (CKPT_ROOT, RUNS_ROOT, DATA_DIR):
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
    print("zstandard   MISSING -- installing")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"], check=True)
    import zstandard
    print("zstandard   ", zstandard.__version__)

if not torch.cuda.is_available():
    raise RuntimeError(
        "NO GPU. This notebook refuses to run on CPU.\n"
        "Runtime -> Change runtime type -> T4 GPU (or better), then rerun.\n"
        "There is deliberately no CPU fallback: it would produce numbers that look fine, take 100x "
        "longer, and say nothing about it in the output.")

DEV = torch.device("cuda")
_props = torch.cuda.get_device_properties(0)
VRAM_GB = _props.total_memory / 1e9
print("device      ", _props.name)
print("vram        ", f"{VRAM_GB:.1f} GB")
if VRAM_GB < 15:
    raise RuntimeError(f"{VRAM_GB:.1f} GB is not enough: fp32 peak is ~8.7 GB with gradient "
                       f"checkpointing and ~13.9 GB without. Use a T4 or better.")
if VRAM_GB < 20:
    print("             NOTE: 16 GB class device -- keep gradient checkpointing ON (peak ~8.7 GB).")

from transformers.models.qwen2 import modeling_qwen2 as _mg
_problems = []
_sig = list(inspect.signature(_mg.eager_attention_forward).parameters)
if _sig[:5] != ["module", "query", "key", "value", "attention_mask"]:
    _problems.append(f"eager_attention_forward's leading params are {_sig[:5]}, expected "
                     f"['module','query','key','value','attention_mask']")
if "scaling" not in _sig:
    _problems.append("eager_attention_forward has no `scaling` keyword")
try:
    _lay_src = inspect.getsource(_mg.Qwen2DecoderLayer.forward)
except (OSError, TypeError):
    _lay_src = ""
    _problems.append("cannot read Qwen2DecoderLayer.forward source -- the 2-tuple unpack is unverifiable")
if _lay_src and ", _ = self.self_attn(" not in _lay_src:
    _problems.append("Qwen2DecoderLayer.forward no longer unpacks a 2-TUPLE from self.self_attn(...); "
                     "our forward returns (output, weights) and would be misread")
for _kw in ("attention_mask=", "past_key_values=", "position_embeddings="):
    if _lay_src and _kw not in _lay_src:
        _problems.append(f"Qwen2DecoderLayer.forward no longer passes {_kw} -- our forward's signature "
                         f"would receive it positionally or not at all")
if not hasattr(_mg, "apply_rotary_pos_emb"):
    _problems.append("apply_rotary_pos_emb is gone; rotary must be IMPORTED, never reimplemented "
                     "(only 16 of 64 head dims rotate on pythia-410m)")
if not hasattr(_mg, "rotate_half"):
    _problems.append("rotate_half is gone; the MLA slot layout is split-half BECAUSE that is what "
                     "rotate_half does, and it is checked numerically below")
if _problems:
    raise RuntimeError(
        f"transformers {transformers.__version__} has changed the Qwen2DecoderLayer attention contract this "
        f"notebook depends on:\n  - " + "\n  - ".join(_problems) +
        "\n\nThe old tree pinned transformers==5.15.0 for this reason. Either install that version or "
        "adapt the modules in the architecture cell to the new shape. ⛔ Do not skip the check: a "
        "changed call shape is SILENT -- wrong numbers, no traceback.")
print(f"contract    Qwen2DecoderLayer call shape OK under transformers {transformers.__version__} "
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
MODEL_TAG      = 'q0p5b_gc'
REGISTRY_SHA   = 'cc3f5cbd21bd2c5bf920e30d9c2491dbf27e8df37e8b85d1dbd7634d05af9b3f'
GRID = {'H': 16,
 'Hkv': 16,
 'L': 24,
 'arms_only': ('mla_w64_ours_unrope_cost',
               'mla_w64_transmla_sigma',
               'mla_w64_mha2mla_s2norm',
               'mla_w64_care_sigma'),
 'base': 'q0p5b',
 'd_k': 64,
 'dm': 1024,
 'family': 'llama_layout',
 'fisher_exclude_top': 0.01,
 'fisher_labels': 'model',
 'grad_clip': 1.0,
 'hf_id': 'Qwen/Qwen1.5-0.5B',
 'init_budgets': 'trained',
 'init_layout': 'B',
 'init_plan': 'mla_only',
 'kl_width_rule': 'unrope_argmin',
 'max_ctx': 32768,
 'model_type': 'qwen2',
 'norm_kind': 'rms',
 'o_bias': False,
 'qkv_bias': True,
 'r': 64,
 'registry_sha': 'cc3f5cbd21bd2c5bf920e30d9c2491dbf27e8df37e8b85d1dbd7634d05af9b3f',
 'registry_version': 'grid-2026-10-04.3',
 'revision': '8f445e3628f3500ee69f24e1303c9f10f5342a39',
 'role': 'trained',
 'rope_base': 1000000,
 'save_final': True,
 'tag': 'q0p5b_gc',
 'teacher': {'dtype': 'float32',
             'hf_id': 'Qwen/Qwen1.5-4B',
             'revision': 'a66363a0c24e2155c561e4b53c658b1d3965474e',
             'tokenizer_oid': '33ea6c72ebb92a237fa2bdf26c5ff16592efcdae',
             'vocab': 151936},
 'tie_word_embeddings': True,
 'tokenizer_oid': '33ea6c72ebb92a237fa2bdf26c5ff16592efcdae',
 'train_plan': 'mla50',
 'vocab': 151936}
GRID_CELLS = [('mla_w64_transmla_sigma', 4, 512, [50000000]),
 ('mla_w64_mha2mla_s2norm', 4, 512, [50000000]),
 ('mla_w64_care_sigma', 4, 512, [50000000]),
 ('mla_w64_ours_unrope_cost', 4, 512, [50000000]),
 ('mla_w64_transmla_sigma', 2, 256, [50000000]),
 ('mla_w64_mha2mla_s2norm', 2, 256, [50000000]),
 ('mla_w64_care_sigma', 2, 256, [50000000]),
 ('mla_w64_ours_unrope_cost', 2, 256, [50000000]),
 ('mla_w64_transmla_sigma', 1, 128, [50000000]),
 ('mla_w64_mha2mla_s2norm', 1, 128, [50000000]),
 ('mla_w64_care_sigma', 1, 128, [50000000]),
 ('mla_w64_ours_unrope_cost', 1, 128, [50000000])]
CELL_W = {('mla_w64_care_sigma', 1): ('fixed:64', 64),
 ('mla_w64_care_sigma', 2): ('fixed:64', 64),
 ('mla_w64_care_sigma', 4): ('fixed:64', 64),
 ('mla_w64_mha2mla_s2norm', 1): ('fixed:64', 64),
 ('mla_w64_mha2mla_s2norm', 2): ('fixed:64', 64),
 ('mla_w64_mha2mla_s2norm', 4): ('fixed:64', 64),
 ('mla_w64_ours_unrope_cost', 1): ('fixed:64', 64),
 ('mla_w64_ours_unrope_cost', 2): ('fixed:64', 64),
 ('mla_w64_ours_unrope_cost', 4): ('fixed:64', 64),
 ('mla_w64_transmla_sigma', 1): ('fixed:64', 64),
 ('mla_w64_transmla_sigma', 2): ('fixed:64', 64),
 ('mla_w64_transmla_sigma', 4): ('fixed:64', 64)}
CARE_ARMS = ('mla_care_sigma', 'mla_w64_care_sigma')
MAX_TOKENS     = 50_000_000
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
KD_CHUNK       = 64
CARE_REQUIRED_VERSION = 'care-2026-09-30.1'
RUN_CELLS      = {(a, g) for a, g, _b, _toks in GRID_CELLS if MAX_TOKENS in _toks}
RUN_ARMS       = tuple(dict.fromkeys(a for a, g, _b, _toks in GRID_CELLS if MAX_TOKENS in _toks))
RUN_GS         = tuple(sorted({g for a, g in RUN_CELLS if a != "mha"}))
TEST_TOKENS = int(os.environ.get("TEST_TOKENS") or 0)
if TEST_TOKENS:
    MAX_TOKENS = TEST_TOKENS
    STAMP = f"{STAMP}-test{TEST_TOKENS}"
    print(f"TEST RUN: {MAX_TOKENS:,} training tokens per branch, under its own stamp {STAMP}")
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
if TEST_TOKENS:
    if TEST_TOKENS < _TOKENS_PER_STEP:
        raise RuntimeError(f"TEST_TOKENS = {TEST_TOKENS:,} is under one optimiser step ({_TOKENS_PER_STEP:,} tokens)")
    LADDERS[MAX_TOKENS] = {"kl": [0, MAX_TOKENS // 2, MAX_TOKENS], "battery": [0, MAX_TOKENS]}
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
_probe = os.path.join(OUT_DIR, ".write_probe")
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
           "mla_w64_ours_unrope_cost": "mla",
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
               "mla_w64_ours_unrope_cost")
WANTED_ARMS = ('mla_w64_transmla_sigma', 'mla_w64_mha2mla_s2norm', 'mla_w64_care_sigma', 'mla_w64_ours_unrope_cost')
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
    if isinstance(G, int) and (arm, G) in CELL_W:
        _rr, _rw = CELL_W[(arm, G)]
        if arch == "mla":
            _hr, _hw = (j.get("mla_width") or {}).get("width_rule", "argmin"), j.get("w")
        else:
            _hr, _hw = _rr, j.get("d_k_R")
        if _hr != _rr or (_rw is not None and _hw != _rw):
            problems.append(f"{rel}: built at rotary width {_hw} (rule {_hr}), but the registry's cell "
                            f"{arm} G={G} expects {_rw} (rule {_rr}) -- a valid checkpoint at the "
                            f"wrong width under this name")
    if isinstance(G, int):
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
MLA_PROBE_VOCAB = 50000
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
    G = meta.get("G")
    if not isinstance(G, int) or isinstance(G, bool):
        raise TypeError(f"meta G must be an int, got {G!r} ({type(G).__name__})")
    w, d_c = meta["w"], meta["d_c"]
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (w, d_c)):
        raise TypeError(f"meta w/d_c must be ints, got {w!r}/{d_c!r}")
    if w + d_c != G * (d_k + d_v):
        raise ValueError(f"cache identity BROKEN: w + d_c = {w} + {d_c} = {w + d_c} != "
                         f"B = G*(d_k+d_v) = {G * (d_k + d_v)}. Refusing to load a branch measured "
                         f"against a different budget.")
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

_SYNTH_TOKEN_MAX = 50253
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
from datasets import load_dataset

CORPUS     = "pile"
N_EVAL_DOC = 64
SEQ_LEN    = 512
SEED_EVAL  = 20260912
DOC_OFFSET = 2000

SHUFFLE_BUF = 10_000

if CORPUS == "pile":
    TRAIN_SRC = "monology/pile-uncopyrighted"
    ev = load_dataset("NeelNanda/pile-10k", split="train")
    _eval_text = lambda j: ev[int(j)]["text"]
    _n_eval_src = len(ev)
elif CORPUS == "smoltalk":
    ROLE_TAGS = {"system": "\n<|system|>\n", "user": "\n<|user|>\n", "assistant": "\n<|assistant|>\n"}
    _render = lambda turns: "".join(ROLE_TAGS[t["role"]] + t["content"] for t in turns)
    ev = load_dataset("HuggingFaceTB/smoltalk", "all", split="test")
    TRAIN_SRC = "HuggingFaceTB/smoltalk"
    _eval_text = lambda j: _render(ev[int(j)]["messages"])
    _n_eval_src = len(ev)
else:
    raise ValueError(f"CORPUS must be 'pile' or 'smoltalk', got {CORPUS!r}")

rng = np.random.default_rng(SEED_EVAL)
eval_ids, eval_pick = [], []
for j in rng.permutation(_n_eval_src):
    if CORPUS == "pile" and j < DOC_OFFSET:
        continue
    e = tok(_eval_text(j))["input_ids"]
    if len(e) < SEQ_LEN + 1:
        continue
    eval_ids.append(e[:SEQ_LEN]); eval_pick.append(int(j))
    if len(eval_ids) == N_EVAL_DOC:
        break
if len(eval_ids) != N_EVAL_DOC:
    raise RuntimeError(f"only {len(eval_ids)} eval docs were long enough; need {N_EVAL_DOC}")
EVAL = torch.tensor(eval_ids, dtype=torch.long)
print(f"eval   {N_EVAL_DOC} docs x {SEQ_LEN} tok = {N_EVAL_DOC*SEQ_LEN} positions  "
      f"(corpus={CORPUS}, seed {SEED_EVAL})")

eval_hashes = {hashlib.sha256(_eval_text(j).encode()).hexdigest() for j in eval_pick}

if CORPUS == "pile":
    _fh = globals().get("FISHER_EVAL_HASHES")
    _fsrc = "the Fisher cell, this session"
    if _fh is None:
        import glob
        _fdir = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/q0p5b_gc/fisher"
        _cands = sorted(glob.glob(os.path.join(_fdir, "fisher_*_heldout.npz"))) \
            if os.path.isdir(_fdir) else []
        if len(_cands) == 1:
            _fh = set(np.load(_cands[0])["eval_sha256"].tolist())
            _fsrc = os.path.basename(_cands[0])
        elif len(_cands) > 1:
            raise RuntimeError(
                f"{len(_cands)} Fisher caches in {_fdir}: {[os.path.basename(c) for c in _cands]}. "
                f"Each was fitted on a different corpus, and nothing here can tell which one the "
                f"checkpoints were built against. Remove the stale one(s) before continuing.")

    _fisher_arms = [a for a in (globals().get("WANTED_ARMS") or ())
                    if "fisher" in a or a.startswith("mla_")]
    if _fh is None:
        if _fisher_arms:
            raise RuntimeError(
                f"no Fisher eval-split record is available, but {_fisher_arms} are in WANTED_ARMS. "
                f"Neither FISHER_EVAL_HASHES (this session) nor a cache in "
                f"{DRIVE_ROOT}/MyDrive/gaugemla/fisher was found, so there is NO WAY to show that "
                f"the metric those arms were built from was fitted on documents disjoint from the "
                f"ones scored here. That is the whole leakage control. Run the Fisher cell (it "
                f"writes the record into its own cache), or drop the Fisher arms.")
        print("       no Fisher arms in WANTED_ARMS and no Fisher cache -- eval-split cross-check "
              "not applicable")
    elif _fh != eval_hashes:
        raise RuntimeError(
            f"the Fisher's eval split and this cell's disagree ({_fsrc}): "
            f"{len(_fh - eval_hashes)} documents it thought were eval are not, and "
            f"{len(eval_hashes - _fh)} the other way. The four corpus constants have drifted "
            f"between the two cells, so the Fisher's 'disjoint' fit corpus may overlap the "
            f"documents scored here and the leakage control is void. Reconcile them before reading "
            f"any Fisher-arm number.")
    else:
        print(f"       Fisher's eval split MATCHES this one ({len(eval_hashes)} documents, from "
              f"{_fsrc}) -- its fit corpus is genuinely disjoint")

    _sh = globals().get("SIGMA_EVAL_HASHES")
    _ssrc = "the Sigma cell, this session"
    if _sh is None:
        import glob
        _sdir = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/q0p5b_gc/sigma"
        _scands = sorted(glob.glob(os.path.join(_sdir, "sigma_*_heldout.npz"))) \
            if os.path.isdir(_sdir) else []
        if len(_scands) == 1:
            _sh = set(np.load(_scands[0])["eval_sha256"].tolist())
            _ssrc = os.path.basename(_scands[0])
        elif len(_scands) > 1:
            raise RuntimeError(
                f"{len(_scands)} Sigma caches in {_sdir}: {[os.path.basename(c) for c in _scands]}. "
                f"Each was fitted on a different corpus and nothing here can tell which one the "
                f"checkpoints were built against. Remove the stale one(s).")

    _sigma_arms = [a for a in (globals().get("WANTED_ARMS") or ())
                   if "sigma" in a or a.startswith("mla_")]
    if _sh is None:
        if _sigma_arms:
            raise RuntimeError(
                f"no Sigma eval-split record is available, but {_sigma_arms} are in WANTED_ARMS. "
                f"Neither SIGMA_EVAL_HASHES (this session) nor a cache in {_sdir} was found, so "
                f"there is NO WAY to show the second moment those arms were built from was fitted "
                f"on documents disjoint from the ones scored here -- and Sigma is the least "
                f"determined object in the pipeline. Run the @@SIGMA@@ cell, or drop the arms.")
        print("       no Sigma arms in WANTED_ARMS and no Sigma cache -- eval-split cross-check "
              "not applicable")
    elif _sh != eval_hashes:
        raise RuntimeError(
            f"Sigma's eval split and this cell's disagree ({_ssrc}): "
            f"{len(_sh - eval_hashes)} documents it thought were eval are not, and "
            f"{len(eval_hashes - _sh)} the other way. Sigma's 'disjoint' fit corpus may overlap "
            f"the documents scored here, and the held-out number that licensed this arm would not "
            f"apply to the checkpoint actually built.")
    else:
        print(f"       Sigma's eval split MATCHES this one ({len(eval_hashes)} documents, from "
              f"{_ssrc}) -- its fit corpus is genuinely disjoint")

    _s2h = globals().get("S2NORM_EVAL_HASHES")
    _s2src = "the @@S2@@ block, this session"
    _s2fit = None
    if _s2h is None:
        import glob
        _s2dir = DRIVE_ROOT + "/MyDrive/gaugemla/grad_clip/q0p5b_gc/s2norm"
        _s2cands = sorted(glob.glob(os.path.join(_s2dir, "s2norm_*.npz"))) \
            if os.path.isdir(_s2dir) else []
        if len(_s2cands) == 1:
            _s2z = np.load(_s2cands[0], allow_pickle=False)
            if "eval_doc_sha256" not in _s2z.files:
                raise RuntimeError(
                    f"the S2 cache {os.path.basename(_s2cands[0])} records no eval_doc_sha256, so "
                    f"there is no way to show the statistic MHA2MLA's selection came from was fitted "
                    f"on documents disjoint from the ones scored here. It was written without the fit "
                    f"and eval hash sets; delete it and re-run the @@S2@@ cell, which passes both.")
            _s2h = set(_s2z["eval_doc_sha256"].tolist())
            _s2fit = set(_s2z["fit_doc_sha256"].tolist()) if "fit_doc_sha256" in _s2z.files else None
            _s2src = os.path.basename(_s2cands[0])
        elif len(_s2cands) > 1:
            raise RuntimeError(
                f"{len(_s2cands)} S2 caches in {_s2dir}: {[os.path.basename(c) for c in _s2cands]}. "
                f"Each was fitted on different ids or a different teacher and nothing here can tell "
                f"which one the checkpoints were built against. Remove the stale one(s).")

    _s2_arms = [a for a in (globals().get("WANTED_ARMS") or ()) if a.endswith("_s2norm")]
    if _s2h is None:
        if _s2_arms:
            raise RuntimeError(
                f"no S2 eval-split record is available, but {_s2_arms} are in WANTED_ARMS. Neither "
                f"S2NORM_EVAL_HASHES (this session) nor a cache in {DRIVE_ROOT}/MyDrive/gaugemla/s2norm "
                f"was found, so there is NO WAY to show that the activation statistic those arms' "
                f"rotary selection came from was fitted on documents disjoint from the ones scored "
                f"here. Run the @@S2@@ block in the init cell (it writes the record into its own "
                f"cache), or drop the MHA2MLA arms.")
        print("       no MHA2MLA arm in WANTED_ARMS and no S2 cache -- eval-split cross-check not "
              "applicable")
    elif _s2h != eval_hashes:
        raise RuntimeError(
            f"the S2 statistic's eval split and this cell's disagree ({_s2src}): "
            f"{len(_s2h - eval_hashes)} documents it thought were eval are not, and "
            f"{len(eval_hashes - _s2h)} the other way. Its 'disjoint' fit corpus may overlap the "
            f"documents scored below, so MHA2MLA's kept rotary pairs may have been chosen on the "
            f"eval set and that arm's numbers would not be its method's.")
    elif _s2fit is not None and _s2fit & eval_hashes:
        raise RuntimeError(
            f"the S2 cache ({_s2src}) records {len(_s2fit & eval_hashes)} FIT documents that are also "
            f"eval documents here. The fit split itself overlaps the scored set -- this is the leak "
            f"directly, not a drift between two beliefs about what eval is.")
    else:
        print(f"       S2's eval split MATCHES this one ({len(eval_hashes)} documents, from {_s2src})"
              + (f" and its {len(_s2fit)}-document fit split is disjoint from it"
                 if _s2fit is not None else " (fit set not recorded)")
              + " -- MHA2MLA's rotary selection is genuinely held out")

def train_batches(batch_size, seq_len, max_tokens, seed=None):
    if seed is None:
        seed = SEED_TRAIN
    stream = load_dataset(TRAIN_SRC, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=SHUFFLE_BUF)
    buf, seen, dropped = [], 0, 0
    for rec in stream:
        text = rec["text"] if CORPUS == "pile" else _render(rec["messages"])
        if hashlib.sha256(text.encode()).hexdigest() in eval_hashes:
            dropped += 1
            continue
        e = tok(text)["input_ids"]
        if len(e) < seq_len + 1:
            continue
        buf.append(e[:seq_len])
        if len(buf) == batch_size:
            yield torch.tensor(buf, dtype=torch.long), dropped
            seen += batch_size * seq_len
            buf = []
            if seen >= max_tokens:
                return

_n, _drop = 0, 0
for _b, _drop in train_batches(2, SEQ_LEN, 2 * SEQ_LEN * 2):
    _n += 1
print(f"train  {TRAIN_SRC} (streaming, shuffled seed={SEED_TRAIN} buf={SHUFFLE_BUF:,}); "
      f"probe yielded {_n} batches, {_drop} eval-overlapping docs dropped")

# %%
@torch.no_grad()
def kl_docs(t_model, s_model, ids, bs=2):
    out = []
    for i in range(0, ids.shape[0], bs):
        chunk = ids[i:i+bs].to(DEV)
        lt = torch.log_softmax(t_model(chunk).logits.float(), dim=-1)
        ls = torch.log_softmax(s_model(chunk).logits.float(), dim=-1)
        out.extend((lt.exp() * (lt - ls)).sum(-1).mean(dim=1).double().cpu().tolist())
        del lt, ls
    return np.array(out)

_self = kl_docs(teacher, teacher, EVAL)
EPS_NUM = float(np.abs(_self).max())
print(f"EPS_NUM   KL(teacher||teacher) over {EVAL.shape[0]} docs: max |KL| = {EPS_NUM:.3e}")
if EPS_NUM > 1e-3:
    raise RuntimeError(f"self-KL {EPS_NUM:.3e} is too large; the instrument is not trustworthy")
print("          This is the NUMERICAL floor -- what 'identical' reads as. The SAMPLING floor is the")
print("          SE column in the summary. They are DIFFERENT quantities and both get quoted.")
print("          Neither is a threshold for whether an effect MATTERS; that needs a criterion from")
print("          the theory, not from this run's variance.")

# %%
import torch.nn.functional as _F

if SEQ_LEN * BATCH != _TOKENS_PER_STEP:
    raise RuntimeError(f"the ladders assume {_TOKENS_PER_STEP} tokens per step; the corpus cell gives "
                       f"SEQ_LEN x BATCH = {SEQ_LEN * BATCH}")
kd_teacher, _kd_tok = load_base(TEACHER, TEACHER_REVISION, dtype=getattr(torch, TEACHER_DTYPE))
for p in kd_teacher.parameters():
    p.requires_grad_(False)
kd_teacher.to(DEV)
if _kd_tok.get_vocab() != tok.get_vocab():
    raise RuntimeError(f"{TEACHER} does not share {MODEL}'s vocabulary: per-token KD is undefined")
for _attr in ("bos_token_id", "eos_token_id", "unk_token_id"):
    if getattr(_kd_tok, _attr) != getattr(tok, _attr):
        raise RuntimeError(f"{_attr} differs: {TEACHER} {getattr(_kd_tok, _attr)} vs {MODEL} "
                           f"{getattr(tok, _attr)}")
if kd_teacher.config.vocab_size != teacher.config.vocab_size:
    raise RuntimeError(f"logit widths differ: {TEACHER} {kd_teacher.config.vocab_size} vs {MODEL} "
                       f"{teacher.config.vocab_size}. KL over the vocabulary would compare different "
                       f"index sets.")
_texts = [tok.decode(d) for d in EVAL[:8]]
for _t in FEASIBILITY_TASKS:
    for _it in TASK_ITEMS[_t][:50]:
        _texts += (_it["ctx"] if isinstance(_it["ctx"], list) else [_it["ctx"]]) + list(_it["conts"])
_bad = [s for s in _texts
        if _kd_tok(s, add_special_tokens=False)["input_ids"] != tok(s, add_special_tokens=False)["input_ids"]]
if _bad:
    raise RuntimeError(f"{len(_bad)} of {len(_texts)} scored texts tokenise differently under {TEACHER}: "
                       f"first {_bad[0][:80]!r}")
with torch.no_grad():
    if not torch.isfinite(kd_teacher(probe).logits).all():
        raise RuntimeError(f"{TEACHER} logits are not finite -- check the dtype")

def _fp_kd(model):
    h = hashlib.sha256()
    for lay in get_layers(model):
        a = get_attn(lay)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            p = getattr(a, name)
            h.update(p.weight.detach().float().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

if _fp_kd(teacher) != TEACHER_FP:
    raise RuntimeError("_fp_kd disagrees with _fp_mha on the float32 original -- the two must coincide")
KD_TEACHER_FP = _fp_kd(kd_teacher)
print(f"kd teacher {TEACHER} @ {TEACHER_REVISION[:12]}: "
      f"{sum(p.numel() for p in kd_teacher.parameters()) / 1e9:.2f}B params, frozen, {TEACHER_DTYPE}; "
      f"same vocabulary ({kd_teacher.config.vocab_size}), special ids and tokenisation on "
      f"{len(_texts)} scored texts; attention fingerprint {KD_TEACHER_FP}")

def _mha_qkv_fp(model):
    h = hashlib.sha256()
    for lay in get_layers(model):
        for p in (get_attn(lay).q_proj, get_attn(lay).k_proj, get_attn(lay).v_proj):
            h.update(p.weight.detach().cpu().numpy().tobytes())
            if p.bias is not None:
                h.update(p.bias.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

@torch.no_grad()
def eval_docs(s_model, ids, bs=2):
    kl_kd, kl_or, ce = [], [], []
    for i in range(0, ids.shape[0], bs):
        chunk = ids[i:i + bs].to(DEV)
        ls = torch.log_softmax(s_model(chunk).logits.float(), dim=-1)
        lk = torch.log_softmax(kd_teacher(chunk).logits.float(), dim=-1)
        kl_kd.extend((lk.exp() * (lk - ls)).sum(-1).mean(dim=1).double().cpu().tolist())
        del lk
        lo = torch.log_softmax(teacher(chunk).logits.float(), dim=-1)
        kl_or.extend((lo.exp() * (lo - ls)).sum(-1).mean(dim=1).double().cpu().tolist())
        del lo
        nll = -ls[:, :-1].gather(-1, chunk[:, 1:, None]).squeeze(-1)
        ce.extend(nll.mean(dim=1).double().cpu().tolist())
        del ls
    return np.array(kl_kd), np.array(kl_or), np.array(ce)

_d_kd, _d_or, _ce_orig = eval_docs(teacher, EVAL)
_ce_kd = eval_docs(kd_teacher, EVAL)[2]
REF_LINES = {"teacher": TEACHER, "teacher_revision": TEACHER_REVISION, "teacher_dtype": TEACHER_DTYPE,
             "original": MODEL, "original_revision": MODEL_REVISION,
             "kl_teacher_vs_original": float(_d_kd.mean()), "kl_original_self": float(_d_or.mean()),
             "ce_original": float(_ce_orig.mean()), "ce_teacher": float(_ce_kd.mean())}
print(f"reference  KL({TEACHER.split('/')[-1]} || original) = {REF_LINES['kl_teacher_vs_original']:.4f} "
      f"nats/token on the eval docs -- the floor an UNCONVERTED model starts from")
print(f"           held-out next-token loss: original {REF_LINES['ce_original']:.4f}, "
      f"teacher {REF_LINES['ce_teacher']:.4f} nats  (self-KL sanity {REF_LINES['kl_original_self']:.1e})")

import datasets as _datasets

def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

def _init_fp(a, g):
    d = CKPTS.get((a, g))
    return json.load(open(os.path.join(d, META_NAME), encoding="utf-8"))["fingerprint"] if d else "MISSING"

def _first_batch_sha():
    for _ids, _ in train_batches(BATCH, SEQ_LEN, BATCH * SEQ_LEN):
        return hashlib.sha256(_ids.cpu().numpy().astype("int64").tobytes()).hexdigest()

_guard_path = os.path.join(RUNS_ROOT, GUARD_STAMP, "_guard.json")
_needs_guard = False
MANIFEST = {"stamp": STAMP, "registry_sha": REGISTRY_SHA, "model": MODEL, "model_revision": MODEL_REVISION,
            "teacher": TEACHER, "teacher_revision": TEACHER_REVISION, "teacher_dtype": TEACHER_DTYPE,
            "recipe": {"lr_peak": LR, "warmup_frac": WARMUP_FRAC, "lr_min_frac": LR_MIN_FRAC,
                       "batch": BATCH, "seq_len": SEQ_LEN, "max_tokens": MAX_TOKENS, "seed": SEED_TRAIN},
            "eval_at_tokens": EVAL_AT_TOKENS, "battery_at_tokens": BATTERY_AT_TOKENS,
            "run_cells": sorted([list(c) for c in RUN_CELLS]),
            "init_fingerprints": {f"{a}_G{g}": _init_fp(a, g) for a, g in sorted(RUN_CELLS) if a != "mha"},
            "train_stream": {"corpus": CORPUS, "src": TRAIN_SRC, "shuffle_buf": SHUFFLE_BUF,
                             "datasets": _datasets.__version__, "first_batch_sha": _first_batch_sha()},
            "eval_docs_sha": hashlib.sha256(EVAL.cpu().numpy().tobytes()).hexdigest(),
            "battery": {t: {"n": n_for(t, "endpoint"), "items_sha": _sha(TASK_ITEMS[t])}
                        for t in FEASIBILITY_TASKS},
            "guard_rule": GUARD_RULE,
            "guard_sha": (hashlib.sha256(open(_guard_path, "rb").read()).hexdigest()
                          if _needs_guard and os.path.exists(_guard_path) else None)}
INFO = {"kl_teacher_vs_original": REF_LINES["kl_teacher_vs_original"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "transformers": transformers.__version__}
_absent = sorted(k for k, v in MANIFEST["init_fingerprints"].items() if v == "MISSING")
if _absent:
    raise RuntimeError(f"registry cells with NO init under {CKPT_ROOT}: {', '.join(_absent)}. Build them with "
                       f"the init notebook first, then re-run from the config cell. Nothing was written for "
                       f"{STAMP}. ⛔ The grid is never trained smaller than it was designed.")
_mp = os.path.join(OUT_DIR, "_manifest.json")
if os.path.exists(_mp):
    _old = json.load(open(_mp, encoding="utf-8"))
    _old_info = _old.pop("info", {})
    _diff = sorted(k for k in set(_old) | set(MANIFEST) if _old.get(k) != MANIFEST.get(k))
    if _diff:
        raise RuntimeError(
            f"this session does not match {STAMP}'s manifest in {_diff}. ⛔ A stamp is one experiment. If "
            f"what differs changed by MISTAKE, change it back. If it changed ON PURPOSE (this model's cell "
            f"table in the registry, a rebuilt init, a new guard decision), it is a different experiment: "
            f"set STAMP_SUFFIX in the config cell to give it its own stamp.")
    _dk = abs(_old_info.get("kl_teacher_vs_original", INFO["kl_teacher_vs_original"])
              - INFO["kl_teacher_vs_original"])
    if _dk > 1e-4:
        raise RuntimeError(f"the KL floor moved by {_dk:.2e} since {STAMP}'s first session "
                           f"({_old_info.get('gpu')}, torch {_old_info.get('torch')}) -> now ({INFO['gpu']}, "
                           f"torch {INFO['torch']}). The same teacher on the same documents should not move.")
    print(f"manifest   matches {STAMP}'s first session ({_mp})")
else:
    with open(_mp, "w", encoding="utf-8") as _f:
        json.dump(dict(MANIFEST, info=INFO), _f, indent=1)
    print(f"manifest   written: {_mp}")
with open(os.path.join(OUT_DIR, f"kd_{STAMP}_reference.json"), "w") as _f:
    json.dump(REF_LINES, _f, indent=1)

# %%
BRANCHES_LEFT = None

def kd_loss(student_logits, teacher_logits):
    ls = torch.log_softmax(student_logits.float(), dim=-1)
    lt = torch.log_softmax(teacher_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(ls, lt, log_target=True, reduction="sum") / (ls.shape[0] * ls.shape[1])

def kd_loss_chunked_backward(student_m, teacher_m, batch, chunk):
    with torch.no_grad():
        th = teacher_m.model(batch).last_hidden_state
    sh = student_m.model(batch).last_hidden_state
    sh_d = sh.detach().requires_grad_(True)
    B, T = batch.shape
    total = torch.zeros((), dtype=torch.float64, device=batch.device)
    for s in range(0, T, chunk):
        with torch.no_grad():
            lt = torch.log_softmax(teacher_m.lm_head(th[:, s:s + chunk]).float(), dim=-1)
        ls = torch.log_softmax(student_m.lm_head(sh_d[:, s:s + chunk]).float(), dim=-1)
        part = torch.nn.functional.kl_div(ls, lt, log_target=True, reduction="sum") / (B * T)
        part.backward()
        total += part.detach().double()
        del lt, ls, part
    sh.backward(sh_d.grad)
    return total.float()

def train_branch(student, branch, log_path, ident, max_tokens=None, mha_baseline=False):
    torch.manual_seed(SEED_TRAIN)
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    student.to(DEV)
    if GRAD_CKPT:
        student.gradient_checkpointing_enable()
    student.train()

    if mha_baseline:
        for _p in student.parameters():
            _p.requires_grad_(True)
        n_cache, kinds = len(get_layers(student)), ["mha[unconverted]"]
        ck = cv = teacher.config.hidden_size
    else:
        n_cache, kinds = train_everything_(student)
        ck, cv = cache_width_of(student)
    _cfp = _mha_qkv_fp if mha_baseline else cache_fingerprint
    cache0 = _cfp(student)
    print(f"    trainable  {n_cache} cache tensors across {kinds} are FREE  "
          f"cache K/V width {ck}/{cv}  fingerprint {cache0}")

    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    total_steps = math.ceil(max_tokens / (BATCH * SEQ_LEN))
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    lr_now = None
    print(f"    lr schedule  warmup {warmup_steps} of {total_steps} steps up to {LR:.1e}, then cosine "
          f"down to {LR_MIN_FRAC * LR:.1e} at step {total_steps - 1}")
    _opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    for _nm, _t in cache_tensors(student):
        if not _t.requires_grad:
            raise RuntimeError(f"{branch}: {_nm} does NOT require grad -- the cache is still frozen, "
                               f"so this is the old experiment. train_everything_ did not take.")
        if id(_t) not in _opt_params:
            raise RuntimeError(f"{branch}: {_nm} is NOT in the optimiser's parameter list -- it "
                               f"carries requires_grad but no step can reach it, which is a freeze "
                               f"by another route.")
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"    trainable {n_train:,} of {sum(p.numel() for p in student.parameters()):,} params; "
          f"the shared cache is TRAINABLE and inside the optimiser")

    rows, step, tokens, t0, done = [], 0, 0, time.time(), set()
    pending = list(EVAL_AT_TOKENS)
    t0_battery, battery_done = {}, set()
    acc = {"loss": 0.0, "gnorm": 0.0, "n": 0,
           "loss_max": 0.0, "gnorm_max": 0.0, "n_clipped": 0}

    def record(step, tokens, thresh, force_battery=False):
        if step in done:
            return None
        done.add(step)
        was_training = student.training
        student.eval()
        k, k_orig, ce = eval_docs(student, EVAL)
        bat, phase = None, None
        if thresh in BATTERY_AT_TOKENS or force_battery:
            phase = BATTERY_PHASE.get(thresh, "endpoint" if force_battery else "periodic")
            print(f"    {branch:<20} battery ({phase}) at {tokens/1e6:.2f}M tokens:")
            bat = run_battery(student, phase)
            battery_done.add(thresh)
        if was_training:
            student.train()
        row = {**ident, "branch": branch, "step": step, "tokens": tokens,
               "kl_mean": float(k.mean()), "kl_se": float(k.std(ddof=1)/np.sqrt(len(k))),
               "n_doc": int(len(k)), "seq_len": SEQ_LEN, "kl_per_doc": [float(x) for x in k],
               "kd_teacher": TEACHER,
               "teacher_revision": TEACHER_REVISION, "teacher_dtype": TEACHER_DTYPE,
               "registry_sha": REGISTRY_SHA, "model_revision": MODEL_REVISION,
               "kl_orig_mean": float(k_orig.mean()),
               "kl_orig_se": float(k_orig.std(ddof=1) / np.sqrt(len(k_orig))),
               "kl_orig_per_doc": [float(x) for x in k_orig],
               "ce_mean": float(ce.mean()), "ce_se": float(ce.std(ddof=1) / np.sqrt(len(ce))),
               "ce_per_doc": [float(x) for x in ce],
               "eps_num": EPS_NUM, "lr": LR, "batch": BATCH, "grad_ckpt": GRAD_CKPT,
               "lr_schedule": "warmup_cosine", "warmup_frac": WARMUP_FRAC,
               "lr_min_frac": LR_MIN_FRAC, "total_steps": total_steps,
               "warmup_steps": warmup_steps, "lr_now": lr_now,
               "corpus": CORPUS, "cache_k_width": ck, "cache_v_width": cv,
               "cache_fingerprint": _cfp(student),
               "battery": bat, "battery_phase": phase,
               "train_loss_mean": (acc["loss"] / acc["n"]) if acc["n"] else None,
               "grad_norm_mean": (acc["gnorm"] / acc["n"]) if (acc["n"] and LOG_GRAD_NORM) else None,
               "train_steps": acc["n"],
               "train_loss_max": acc["loss_max"] if acc["n"] else None,
               "grad_norm_max": acc["gnorm_max"] if (acc["n"] and LOG_GRAD_NORM) else None,
               "clipped_frac": (acc["n_clipped"] / acc["n"]) if acc["n"] else None,
               "grad_clip": GRAD_CLIP, "loss_reduction": "token_mean",
               "eval_at_tokens": list(EVAL_AT_TOKENS),
               "battery_at_tokens": list(BATTERY_AT_TOKENS),
               "token_threshold": thresh, "wall_s": round(time.time()-t0, 1)}
        rows.append(row)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"    {branch:<20} step {step:>6}  {tokens/1e6:6.2f}M tok  "
              f"KL {row['kl_mean']:8.4f} +/- {row['kl_se']:.4f}   KL_orig {row['kl_orig_mean']:.4f}  "
              f"CE {row['ce_mean']:.4f}   lr {'--' if lr_now is None else format(lr_now, '.2e')}")
        if thresh == 0 and bat is not None:
            t0_battery.update(bat)
        acc["loss"] = acc["gnorm"] = acc["loss_max"] = acc["gnorm_max"] = 0.0
        acc["n_clipped"] = 0
        acc["n"] = 0
        return row

    record(0, 0, 0)
    pending.pop(0)

    if t0_battery:
        verdict, worst = check_t0(f"{ident['arm']}_G{ident['G']}", t0_battery)
        print(verdict)

    for batch, _ in train_batches(BATCH, SEQ_LEN, max_tokens):
        batch = batch.to(DEV)
        if KD_CHUNK is None:
            with torch.no_grad():
                tl = kd_teacher(batch).logits
            loss = kd_loss(student(batch).logits, tl)
            del tl
            loss.backward()
        else:
            loss = kd_loss_chunked_backward(student, kd_teacher, batch, KD_CHUNK)
        _g = float(torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.grad is not None], GRAD_CLIP, error_if_nonfinite=True))
        if LOG_TRAIN_LOSS:
            _l = float(loss.item())
            acc["loss"] += _l
            acc["loss_max"] = max(acc["loss_max"], _l)
            if LOG_GRAD_NORM:
                acc["gnorm"] += _g
                acc["gnorm_max"] = max(acc["gnorm_max"], _g)
            acc["n_clipped"] += int(_g > GRAD_CLIP)
            acc["n"] += 1
        lr_now = lr_at(step, total_steps)
        for _pg in opt.param_groups:
            _pg["lr"] = lr_now
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1; tokens += batch.numel()

        if step == TIME_PROBE:
            rate = tokens / (time.time() - t0)
            per_branch_h = max_tokens / rate / 3600
            print(f"    {branch:<20} THROUGHPUT {rate:,.0f} tok/s over {TIME_PROBE} steps  ->  "
                  f"{per_branch_h:.2f} h for {max_tokens/1e6:.1f}M tokens")
            print(f"    {branch:<20} peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
                  f"(BATCH={BATCH}, GRAD_CKPT={GRAD_CKPT})")
            n_left = BRANCHES_LEFT if BRANCHES_LEFT else 1
            overhead_h = (rows[0]["wall_s"] if rows else 0.0) / 3600.0
            total_h = n_left * (per_branch_h + overhead_h * len(BATTERY_AT_TOKENS))
            print(f"    {branch:<20} GRID: {n_left} branch(es) left this session x "
                  f"({per_branch_h:.2f} h train + ~{overhead_h*len(BATTERY_AT_TOKENS):.2f} h eval) "
                  f"= {total_h:.1f} h")
            print(f"    {branch:<20} stop the cell NOW if that is too long. Completed branches are "
                  f"skipped on a re-run, so stopping costs only this branch, and a partial grid is "
                  f"a valid result -- every completed branch is already closed on Drive.")
        crossed = []
        while pending and tokens >= pending[0]:
            crossed.append(pending.pop(0))
        if crossed:
            if len(crossed) > 1:
                print(f"    {branch:<20} one batch crossed {len(crossed)} thresholds "
                      f"{[c/1e6 for c in crossed]}M; logging ONE row at {crossed[-1]/1e6:.2f}M -- they "
                      f"were not measured separately, so there is nothing to log for the others.")
            record(step, tokens, crossed[-1])
    record(step, tokens, None, force_battery=(max_tokens not in battery_done))
    if pending:
        print(f"    ⚠ {branch}: the stream ended at {tokens/1e6:.2f}M tokens with thresholds "
              f"{[t/1e6 for t in pending]}M never reached. This is a PARTIAL trajectory: the row above "
              f"carries token_threshold=None and the read-out will not plot it as an endpoint.")

    moved = _cfp(student) != cache0
    print(f"    {branch:<20} cache fingerprint {cache0} -> {_cfp(student)}")
    if not moved:
        raise RuntimeError(f"{branch}: THE SHARED CACHE DID NOT MOVE during training. Every arm's "
                           f"basis is meant to be trainable now, so a bit-identical cache means the "
                           f"freeze is still in force somewhere and this run is the old experiment.")
    if not mha_baseline and cache_width_of(student) != (ck, cv):
        raise RuntimeError(f"{branch}: the cached width changed during training")
    with torch.no_grad():
        if not torch.equal(teacher(probe).logits.double(), lg_teacher):
            raise RuntimeError("THE TEACHER MOVED during training. Every comparison is void.")
    if _fp_mha(teacher) != TEACHER_FP:
        raise RuntimeError("the teacher's attention weights changed. Every comparison is void.")
    if _fp_kd(kd_teacher) != KD_TEACHER_FP:
        raise RuntimeError("the KD TEACHER's attention weights changed. Every comparison is void.")
    print(f"    {branch:<20} teacher unchanged, cache MOVED, width intact")
    return rows

# %%
_branch = os.environ.get("BRANCH", "")
if _branch:
    _arm, _sep, _g = _branch.rpartition("_G")
    _gv = {str(g): g for a, g in RUN_CELLS}.get(_g) if _sep else None
    if _gv is None or (_arm, _gv) not in RUN_CELLS:
        raise RuntimeError(f"BRANCH={_branch!r} is not a branch this run trains: "
                           f"{sorted(f'{a}_G{g}' for a, g in RUN_CELLS)}")
    RUN_ARMS, RUN_GS = (_arm,), (_gv,)
    print(f"BRANCH     {_branch}: this job trains that one branch ({MAX_TOKENS // 1_000_000}M tokens)")

# %%
for _n in ("MAX_TOKENS", "EVAL_AT_TOKENS", "LR", "BATCH", "SEQ_LEN", "GRAD_CKPT", "probe",
           "TEACHER_FP", "MODEL", "TEACHER", "KD_TEACHER_FP", "WARMUP_FRAC", "LR_MIN_FRAC", "REGISTRY_SHA", "CORPUS", "TRAIN_SRC", "SHUFFLE_BUF", "FINGERPRINT", "load_ckpt",
           "cache_fingerprint", "META_NAME", "SEARCHED_ARMS", "OUT_DIR", "CKPT_ROOT", "DRIVE_ROOT"):
    if _n not in globals():
        raise NameError(f"{_n} is not defined -- run the cells above this one first, in order.")

import datasets

MODEL_OUT = os.path.join(OUT_DIR, "models")

if not MODEL_OUT.startswith(DRIVE_ROOT + "/"):
    raise RuntimeError(
        f"MODEL_OUT = {MODEL_OUT!r} is not under {DRIVE_ROOT}. ⛔ Drive is the only permitted "
        f"location -- no runtime-local disk, no /tmp, no local staging. Staging locally would be worse than an "
        f"alternative destination, not better: it would SUCCEED while Drive was broken and fail at "
        f"the copy, turning a Drive failure into a late one after the whole branch had been written.")
if os.path.abspath(MODEL_OUT).startswith(os.path.abspath(CKPT_ROOT) + os.sep):
    raise RuntimeError(
        f"MODEL_OUT = {MODEL_OUT!r} is inside CKPT_ROOT = {CKPT_ROOT!r}. ⛔ Stage 0 would discover "
        f"these trained directories as initialisations, flag every one as a duplicate (arm, G), and "
        f"raise before anything trains.")
os.makedirs(MODEL_OUT, exist_ok=True)

import shutil as _shutil
_probe_m = os.path.join(MODEL_OUT, f".write_probe.{os.environ.get('PBS_JOBID', os.getpid())}")
try:
    with open(_probe_m, "w") as _f:
        _f.write("probe")
    os.remove(_probe_m)
except Exception as _e:
    raise RuntimeError(f"{MODEL_OUT} is not writable ({type(_e).__name__}: {_e}). ⛔ Fix Drive; do "
                       f"not redirect the models elsewhere.") from _e

def _init_bytes(arm, G):
    _fs = glob.glob(os.path.join(CKPTS[(arm, G)], "*.safetensors"))
    if not _fs:
        raise RuntimeError(f"{CKPTS[(arm, G)]}: no .safetensors, though STAGE 0 accepted this init. Its "
                           f"weights went missing since: find out why before training anything.")
    return sum(os.path.getsize(f) for f in _fs)

_sizes = [_init_bytes(a, g) for a, g in BRANCHES]
_total_gb = sum(_sizes) / 1e9
_free_gb = _shutil.disk_usage(MODEL_OUT).free / 1e9
print(f"endpoint models -> {MODEL_OUT}")
print(f"  {len(BRANCHES)} branch(es) in the discovered grid, {min(_sizes)/1e9:.2f}-{max(_sizes)/1e9:.2f}"
      f" GB each (fp32) = {_total_gb:.1f} GB if every one is trained")
print(f"  Drive reports {_free_gb:.1f} GB free")
if _free_gb < _total_gb:
    print(f"  ⚠ ⚠ THAT IS NOT ENOUGH for the full grid ({_total_gb:.1f} GB needed, {_free_gb:.1f} GB "
          f"free). Not raising -- a partial grid is a legitimate session, and a branch that cannot "
          f"save still produces valid measurements. But the save WILL fail part-way through, at the "
          f"end of a branch that trained perfectly, so decide now: free space, or narrow RUN.")
print(f"  No optimiser state, no mid-training checkpoints, no resume. Read the comment above before "
      f"changing that.")

PROBE_CPU = probe.to("cpu")

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

def _safetensors_declared_size(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if n <= 0 or n > 100_000_000:
            raise RuntimeError(f"{path}: the safetensors header length reads {n}, which is not "
                               f"plausible -- either this is not a safetensors file or its first 8 "
                               f"bytes never landed.")
        raw = f.read(n)
        if len(raw) != n:
            raise RuntimeError(f"{path}: only {len(raw):,} of {n:,} header bytes are present. The "
                               f"write was TRUNCATED inside the safetensors header itself. Re-run the "
                               f"branch; do not patch the file.")
        hdr = json.loads(raw.decode("utf-8"))
    ends = [v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"]
    if not ends:
        raise RuntimeError(f"{path}: the safetensors header names no tensors.")
    return 8 + n + max(ends)

def save_branch(student, model_dir, init_meta, ident, log_path, rows):
    if not rows:
        raise RuntimeError(f"{model_dir}: train_branch returned no rows, so there is no endpoint row "
                           f"to describe this checkpoint -- it would be 1.5 GB of weights that cannot "
                           f"say what they are. Refusing to write.")
    last = rows[-1]
    branch = last["branch"]

    if last["tokens"] < max(EVAL_AT_TOKENS):
        print(f"    ⚠ ⚠ {branch}: PARTIAL trajectory ({last['tokens']/1e6:.2f}M of "
              f"{max(EVAL_AT_TOKENS)/1e6:.0f}M tokens) -- NO checkpoint written.")
        print(f"         The measurements are on Drive and intact. Nothing is saved because a partial "
              f"endpoint is not an endpoint.")
        return None

    meta_path = os.path.join(model_dir, META_NAME)
    if os.path.exists(meta_path):
        raise RuntimeError(
            f"{model_dir} already holds a sidecar -- a checkpoint an earlier run wrote AND verified -- "
            f"while the log for {branch} is incomplete. Continuing would truncate those weights in "
            f"place. ⛔ Move or delete {os.path.relpath(model_dir, OUT_DIR)} deliberately, or work out "
            f"where its log went.")
    os.makedirs(model_dir, exist_ok=True)

    student.eval()
    student.to("cpu")

    arch = init_meta["arch"]
    fp = FINGERPRINT[arch](student)
    state_fp = _fp_state(student)
    if fp == init_meta["fingerprint"]:
        raise RuntimeError(
            f"{branch}: the trained attention fingerprint EQUALS the initialisation's ({fp}) after "
            f"{last['tokens']:,} tokens. q_proj and dense train in every arm, so this is impossible "
            f"unless nothing trained or the wrong object is being saved. Not writing it as an endpoint.")
    with torch.no_grad():
        lg_pre = student(PROBE_CPU).logits.double()
    if not torch.isfinite(lg_pre).all():
        raise RuntimeError(f"{branch}: the trained model's own logits are not finite. There is nothing "
                           f"here worth saving and nothing downstream that could use it.")

    t_w = time.time()
    student.save_pretrained(model_dir)

    wfile = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(wfile):
        raise RuntimeError(
            f"{model_dir}: save_pretrained returned but there is no model.safetensors. At this size "
            f"the model is one shard by construction, so a sharded layout means something changed "
            f"upstream -- check for model.safetensors.index.json before assuming the write failed.")
    nbytes = os.path.getsize(wfile)
    declared = _safetensors_declared_size(wfile)
    if nbytes != declared:
        raise RuntimeError(
            f"{wfile} is {nbytes:,} bytes on disk but its own header declares {declared:,}. The write "
            f"was TRUNCATED -- Drive dropped the upload part-way. ⛔ No sidecar has been written, so "
            f"no reader will treat this directory as a checkpoint. Re-run the branch; do not patch "
            f"the file.")
    dt = max(time.time() - t_w, 1e-9)
    print(f"    {branch:<20} wrote {nbytes/1e9:.3f} GB in {dt/60:.1f} min ({nbytes/1e6/dt:.1f} MB/s, "
          f"measured on this mount rather than assumed)")

    meta = dict(init_meta)
    meta["fingerprint"] = fp
    meta["state_fingerprint"] = state_fp
    if arch == "mla":
        if not init_meta.get("mla_probe"):
            raise RuntimeError(
                f"{branch}: the initialisation meta of an 'mla' branch carries no mla_probe. "
                f"dt_checkpoint.save_checkpoint requires it of every 'mla' sidecar, so this branch "
                f"was built by something else -- and this endpoint would be written with no "
                f"forward-pass check at all.")
        meta["init_mla_probe"] = init_meta["mla_probe"]
        meta["mla_probe"] = mla_probe(student)
        print(f"    {branch:<20} mla probe RECOMPUTED on the trained CPU model; the t=0 vector is "
              f"kept as init_mla_probe (a copied-forward probe fails every verify)")
    for k in ("arch", "arm", "base_model", "fingerprint", "G"):
        if k not in meta:
            raise RuntimeError(
                f"{branch}: the initialisation meta carried no {k!r}, so this checkpoint could not be "
                f"reloaded. load_ckpt names the first four by hand; G is read unguarded by both "
                f"skeleton builders and would surface as a bare KeyError after the base download.")
    if meta.get("groups_by_layer") is None and meta["arm"] in SEARCHED_ARMS and meta["G"] > 1:
        raise RuntimeError(
            f"{branch}: searched arm {meta['arm']!r} at G={meta['G']} is about to be written with NO "
            f"groups_by_layer. The reload would fall back to the contiguous default, load_state_dict "
            f"would pass, both fingerprints would match, and every head would read the wrong group's "
            f"K/V. Refusing to write a checkpoint that can only be loaded wrongly.")

    meta.update({
        "trained": True,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "init_ckpt": ident["ckpt"],
        "init_fingerprint": init_meta["fingerprint"],
        "file_bytes": nbytes,
        "dtype": str(next(student.parameters()).dtype).replace("torch.", ""),
        "run_id": ident["run_id"], "stamp": ident["stamp"], "branch": branch,
        "seed": ident["seed"],
        "log": os.path.relpath(log_path, OUT_DIR),
        "step": last["step"], "tokens": last["tokens"], "max_tokens": MAX_TOKENS,
        "eval_at_tokens": list(EVAL_AT_TOKENS),
        "kl_mean": last["kl_mean"], "kl_se": last["kl_se"], "n_doc": last["n_doc"],
        "cache_fingerprint": last["cache_fingerprint"],
        "cache_k_width": last["cache_k_width"], "cache_v_width": last["cache_v_width"],
        "teacher": TEACHER, "teacher_fingerprint": KD_TEACHER_FP,
        "original": MODEL, "original_fingerprint": TEACHER_FP,
        "lr": LR, "batch": BATCH, "seq_len": SEQ_LEN, "grad_ckpt": GRAD_CKPT,
        "registry_sha": REGISTRY_SHA, "teacher_revision": TEACHER_REVISION,
        "teacher_dtype": TEACHER_DTYPE, "original_revision": MODEL_REVISION,
        "lr_schedule": "warmup_cosine", "warmup_frac": WARMUP_FRAC,
        "grad_clip": GRAD_CLIP, "loss_reduction": "token_mean",
        "lr_min_frac": LR_MIN_FRAC,
        "corpus": CORPUS, "train_src": TRAIN_SRC, "shuffle_buf": SHUFFLE_BUF,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "datasets": datasets.__version__},
        "resume_state": "none -- final weights only. No optimiser moments and no stream position, so "
                        "this cannot be continued; it can only be measured.",
        "load_with": "load_ckpt() from q0p5b_gc_run.ipynb. AutoModelForCausalLM.from_pretrained on "
                     "this directory builds the stock FUSED skeleton and silently random-initialises "
                     "the attention -- config.json here describes the unrestructured model.",
    })
    pending_path = meta_path + ".unverified"
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return {"model_dir": model_dir, "branch": branch, "meta_path": meta_path,
            "pending_path": pending_path,
            "logits": lg_pre, "state_fingerprint": state_fp,
            "cache_fingerprint": last["cache_fingerprint"]}

def verify_branch(saved, expect_arm):
    model_dir, branch = saved["model_dir"], saved["branch"]
    try:
        m, meta_back, _info = load_ckpt(model_dir, expect_arm=expect_arm, verbose=False,
                                        meta_name=os.path.basename(saved["pending_path"]))
        st = _fp_state(m)
        if st != saved["state_fingerprint"]:
            raise RuntimeError(
                f"full-state fingerprint {st} != {saved['state_fingerprint']} taken before the write. "
                f"The attention block round-tripped (load_ckpt just checked it) so the damage is in "
                f"the embeddings, an MLP or a LayerNorm -- the ~80% no arch fingerprint covers.")
        with torch.no_grad():
            lg = m(PROBE_CPU).logits.double()
        if not torch.equal(lg, saved["logits"]):
            d = float((lg - saved["logits"]).abs().max())
            raise RuntimeError(
                f"the reloaded model's logits differ from the trained model's (max |delta| {d:.3e}) "
                f"while EVERY parameter is bit-identical. The weights are fine and the RECONSTRUCTION "
                f"is not: suspect meta['groups_by_layer'], meta['arm'] or meta['G'] -- the partition "
                f"and the wide flag are rebuilt from the sidecar and live in no tensor.")
        cf = cache_fingerprint(m)
        if cf != saved["cache_fingerprint"]:
            raise RuntimeError(
                f"the saved shared cache hashes {cf}, but the log row written from this model says "
                f"{saved['cache_fingerprint']}. The checkpoint is not the model the endpoint row was "
                f"measured on.")
    except KeyboardInterrupt:
        print(f"    ⚠ {branch}: verify INTERRUPTED. The checkpoint is written but NOT verified -- it "
              f"carries {os.path.basename(saved['pending_path'])} rather than {META_NAME}, so no "
              f"reader will treat it as complete. Re-run this branch, or verify it by hand.")
        raise
    except Exception:
        failed = saved["meta_path"] + ".FAILED"
        try:
            os.replace(saved["pending_path"], failed)
            where = os.path.basename(failed)
        except OSError as _e:
            where = f"(could not be renamed: {type(_e).__name__}) {saved['pending_path']}"
        print(f"    ⛔ {branch}: VERIFY FAILED. The sidecar is at {where}, NOT at {META_NAME}, so this "
              f"directory reads as no checkpoint at all. The weights are left in place as evidence "
              f"and the log is untouched and complete.")
        raise
    os.replace(saved["pending_path"], saved["meta_path"])
    del m

    files = sorted(os.listdir(model_dir))
    gb = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files) / 1e9
    print(f"    {branch:<20} VERIFIED  reloaded from Drive, every parameter bit-identical, logits "
          f"bit-identical, cache {cf} ✓")
    print(f"    {branch:<20} on Drive at {os.path.relpath(model_dir, OUT_DIR)}: {len(files)} files, "
          f"{gb:.2f} GB, fingerprint {meta_back['fingerprint']} (init was "
          f"{meta_back['init_fingerprint']})")

RUN = [(a, g) for a, g in BRANCHES
       if (RUN_ARMS is None or a in RUN_ARMS) and (RUN_GS is None or g in RUN_GS)]
RUN = [(a, g) for a, g in RUN if (a, g) in RUN_CELLS]
_missing = sorted(c for c in RUN_CELLS if c[0] != "mha" and c not in set(BRANCHES)
                  and (RUN_ARMS is None or c[0] in RUN_ARMS) and (RUN_GS is None or c[1] in RUN_GS))
if _missing:
    raise RuntimeError(f"registry cells with NO init under {CKPT_ROOT}: "
                       f"{', '.join(f'{a}_G{g}' for a, g in _missing)}. Build them with the init "
                       f"notebook first. ⛔ The grid is never trained smaller than it was designed.")
if RUN_ARMS is not None or RUN_GS is not None:
    _want = {(a, g) for a in (RUN_ARMS if RUN_ARMS is not None else {a for a, _ in BRANCHES})
             if a != "mha"
             for g in (RUN_GS if RUN_GS is not None else {g for _, g in BRANCHES})}
    _absent = sorted(_want - set(BRANCHES))
    print(f"config: RUN_ARMS={RUN_ARMS}  RUN_GS={RUN_GS}  ->  {len(RUN)} of {len(BRANCHES)} "
          f"discovered branch(es)")
    if _absent:
        print(f"  ⚠ requested but NOT on disk, so not trained: "
              f"{', '.join(f'{a}_G{g}' for a, g in _absent)}")
        print(f"     (at G=1 {sorted(G1_DEDUP_BASE)} are deduplicated onto their base arm, so "
              f"their absence THERE is by construction. Any other arm missing at G=1 is a real "
              f"gap: procrustes_simk left that table on 2026-09-27 and is trained at G=1 in its "
              f"own right, so its absence would NOT be by construction.)")
if ("mha", 0) in RUN_CELLS:
    RUN = [("mha", 0)] + RUN
    print("config: + the MHA baseline mha_G0 (the unconverted pythia-410m; no init checkpoint)")
_care_bad = []
for _a, _g in RUN:
    if _a in CARE_ARMS:
        _cm = json.load(open(os.path.join(CKPTS[(_a, _g)], META_NAME), encoding="utf-8"))
        _cv = (_cm.get("mla_provenance") or {}).get("version")
        if _cv != CARE_REQUIRED_VERSION:
            _care_bad.append(f"{_a}_G{_g} (version {_cv!r})")
if _care_bad:
    raise RuntimeError(
        f"CARE checkpoint(s) {_care_bad} were not built by the stable solver "
        f"({CARE_REQUIRED_VERSION}). The release solver is numerically broken at G = 8. Delete "
        f"those init directories (init/<CARE arm>_G*_t0/) on Drive and rebuild them with q0p5b_gc_init.ipynb. "
        f"⛔ Do not drop a CARE arm: the registry's cells are the grid.")
if not RUN:
    raise RuntimeError(
        f"RUN is empty: RUN_ARMS={RUN_ARMS} and RUN_GS={RUN_GS} select none of the "
        f"{len(BRANCHES)} discovered branches {sorted(BRANCHES)}. Widen them in the config cell.")

_mf = os.path.join(OUT_DIR, "_manifest.json")
if not os.path.exists(_mf) or json.load(open(_mf, encoding="utf-8")).get("stamp") != STAMP:
    raise RuntimeError(f"no manifest for {STAMP} in {OUT_DIR}: the teacher cell has not run for this "
                       f"stamp. Re-run every cell from the config cell down.")
if False:
    _gp = os.path.join(RUNS_ROOT, GUARD_STAMP, "_guard.json")
    if not os.path.exists(_gp):
        raise RuntimeError(f"no guard decision at {_gp}. Run the 5M grid at LR_PEAK = 2e-5, seed "
                           f"{GUARD_SEED}, and its guard cell first: every other run depends on it.")
    _guard = json.load(open(_gp, encoding="utf-8"))
    _want_g = {"registry_sha": REGISTRY_SHA, "stamp": GUARD_STAMP, "rule": GUARD_RULE,
               "warmup_end_tokens": GUARD_WARMUP_END}
    _bad_g = {k: (_guard.get(k), v) for k, v in _want_g.items() if _guard.get(k) != v}
    if _bad_g:
        raise RuntimeError(f"{_gp} was not decided under this notebook: {_bad_g} (file, expected). "
                           f"Re-run the {GUARD_STAMP} grid and its guard cell.")
    _wrong = []
    for _g in sorted({g for a, g in RUN if a != "mha"}):
        _d = _guard["per_G"].get(str(_g))
        if _d is None:
            raise RuntimeError(f"the guard has no decision for G = {_g} ({_gp})")
        if (2e-6 if _d["tripped"] else 2e-5) != LR_PEAK:
            _wrong.append(f"G={_g} (guard: {'tripped' if _d['tripped'] else 'clear'})")
    if _wrong:
        raise RuntimeError(f"LR_PEAK = {LR_PEAK} contradicts the guard for {_wrong}. Tripped budgets "
                           f"run at 2e-6, clear ones at 2e-5: set LR_PEAK and RUN_GS to match.")
    print(f"guard      every requested G agrees with {_gp} at LR_PEAK = {LR_PEAK}")
RUN_ID = f"{STAMP}-{int(time.time())}"
print(f"run_id {RUN_ID}   (rows carry it; the read-out uses ONE run per branch, never a mixture)")

_per_batt = {p: sum(SIZES[t][0 if p == "periodic" else 1] for t in FEASIBILITY_TASKS)
             for p in ("periodic", "endpoint")}
_items_per_branch = sum(_per_batt[BATTERY_PHASE.get(t, "periodic")] for t in BATTERY_AT_TOKENS)
print(f"\nper branch: {MAX_TOKENS/1e6:.0f}M training tokens, {len(EVAL_AT_TOKENS)} KL points, "
      f"{len(BATTERY_AT_TOKENS)} batteries ({_items_per_branch:,} scored items)")
print(f"this session: {len(RUN)} branch(es) requested of {len(BRANCHES)} discovered")
print("⚠ The throughput probe at step 50 prints a real per-branch hour projection. Read it and stop")
print("  the cell if the total is longer than the session will last -- the skip logic makes that safe.")

_need = sum(_init_bytes(a, g) for a, g in RUN if a != "mha")
_free = _shutil.disk_usage(MODEL_OUT).free
if _free < _need:
    raise RuntimeError(f"{MODEL_OUT} has {_free / 1e9:.1f} GB free; this run's {len(RUN)} branch(es) will write "
                       f"{_need / 1e9:.1f} GB of endpoint weights. Free space first: nothing has trained yet.")
print(f"endpoint room: this run writes {_need / 1e9:.1f} GB, {_free / 1e9:.1f} GB free")

ALL = {}
no_model = []
save_failed = []

for _i, (arm, G) in enumerate(RUN):
    BRANCHES_LEFT = len(RUN) - _i
    branch = f"{arm}_G{G}_s{SEED_TRAIN}"
    log = os.path.join(OUT_DIR, f"kd_{STAMP}_{branch}.jsonl")
    model_dir = os.path.join(MODEL_OUT, f"kd_{STAMP}_{branch}")
    ckpt_dir = None if arm == "mha" else CKPTS[(arm, G)]

    if os.path.exists(log):
        prev = [json.loads(l) for l in open(log) if l.strip()]
        reach = {}
        for rr in prev:
            rid = rr.get("run_id", "legacy")
            reach[rid] = max(reach.get(rid, 0), rr["tokens"])
        if reach and max(reach.values()) >= max(EVAL_AT_TOKENS):
            best = max(reach, key=reach.get)
            print(f"\n=== {branch}: already complete -- SKIPPED (run {best} reached "
                  f"{reach[best]/1e6:.2f}M tokens, {len(prev)} rows in the file)")
            if arm != "mha" and not os.path.exists(os.path.join(model_dir, META_NAME)):
                print(f"    ⚠ ⚠ LOG COMPLETE, NO ENDPOINT MODEL at "
                      f"{os.path.relpath(model_dir, OUT_DIR)}/")
                print(f"         The measurements are intact and valid. The weights for THAT run "
                      f"cannot be recovered by re-running -- a new run is a new trajectory.")
                print(f"         If the endpoint weights are wanted, move this log aside and re-run "
                      f"the branch deliberately.")
                no_model.append(branch)
            ALL[branch] = prev
            continue
        if prev:
            print(f"\n=== {branch}: {len(prev)} rows from {len(reach)} earlier run(s), none complete "
                  f"(furthest {max(reach.values())/1e6:.2f}M tokens).")
            print(f"    Re-running from t=0 -- the endpoint save happens only when a branch REACHES "
                  f"{max(EVAL_AT_TOKENS)/1e6:.1f}M tokens and is never read back into training, so "
                  f"this is a NEW run, not a continuation. It appends under run_id {RUN_ID}, and the "
                  f"read-out will use ONE run's rows only: it will NOT splice this run's early points "
                  f"onto an older run's late ones.")

    if os.path.exists(os.path.join(model_dir, META_NAME)):
        raise RuntimeError(
            f"{branch}: a VERIFIED endpoint model already exists at "
            f"{os.path.relpath(model_dir, OUT_DIR)}/ but this branch's log is not complete against the "
            f"current budget ({max(EVAL_AT_TOKENS)/1e6:.1f}M tokens). Training it would spend hours "
            f"and then refuse to write, because the write truncates in place.\n"
            f"  If MAX_TOKENS was raised, that model is an endpoint for a DIFFERENT budget -- move it "
            f"aside deliberately.\n"
            f"  If the log went missing, the model is the only record of that run.")

    if arm == "mha":
        print(f"\n=== {branch}  <-  {MODEL} (UNCONVERTED: the MHA baseline) ===")
        student, _ = load_base(MODEL)
        ident = {"stamp": STAMP, "run_id": RUN_ID, "arm": arm, "G": G, "seed": SEED_TRAIN,
                 "arch": "mha", "wide": False, "ckpt": f"{MODEL} (unconverted)",
                 "ckpt_fingerprint": _fp_mha(student), "groups_searched": False}
        ALL[branch] = train_branch(student, branch, log, ident, mha_baseline=True)
        print(f"    {branch:<20} MHA baseline: measurements on Drive; no endpoint weights saved")
        del student
        torch.cuda.empty_cache()
        continue
    print(f"\n=== {branch}  <-  {os.path.relpath(ckpt_dir, CKPT_ROOT)} ===")
    student, meta, info = load_ckpt(ckpt_dir, expect_arm=arm)
    if meta.get("G") != G:
        raise RuntimeError(f"{ckpt_dir}: meta says G={meta.get('G')}, the grid says G={G}")
    ident = {"stamp": STAMP, "run_id": RUN_ID, "arm": arm, "G": G, "seed": SEED_TRAIN,
             "arch": meta["arch"], "wide": bool(info.get("wide", False)),
             "ckpt": os.path.relpath(ckpt_dir, CKPT_ROOT),
             "ckpt_fingerprint": meta["fingerprint"],
             "groups_searched": meta.get("groups_by_layer") is not None}
    ALL[branch] = train_branch(student, branch, log, ident)
    try:
        saved = save_branch(student, model_dir, meta, ident, log, ALL[branch])
        del student
        torch.cuda.empty_cache()
        if saved is not None:
            verify_branch(saved, arm)
    except Exception as _e:
        import traceback
        traceback.print_exc()
        print(f"    ⛔ {branch}: THE SAVE FAILED ({type(_e).__name__}). Its MEASUREMENTS are complete "
              f"and on Drive -- only the weights are lost. Continuing to the next branch rather than "
              f"ending the grid here.")
        save_failed.append(branch)
        globals().pop("student", None)
        torch.cuda.empty_cache()

print(f"\nlogs   -> {OUT_DIR}")
print(f"models -> {MODEL_OUT}")
if no_model:
    print(f"\n⚠ {len(no_model)} complete branch(es) have NO endpoint model on Drive: {no_model}")
    print(f"  Their measurements are complete and valid; only the weights are missing. Deliberately "
          f"not re-run -- see the note printed against each one above.")
if save_failed:
    print(f"\n⛔ {len(save_failed)} branch(es) TRAINED BUT FAILED TO SAVE: {save_failed}")
    print(f"  Their measurements are complete and on Drive. Each printed its own traceback above. A "
          f"re-run would produce a different trajectory, so the weights for THOSE runs are gone -- "
          f"decide per branch whether the artefact is worth a fresh run.")
_orphans = []
for _d in sorted(glob.glob(os.path.join(MODEL_OUT, "*"))):
    if os.path.isdir(_d) and not os.path.exists(os.path.join(_d, META_NAME)):
        _gb = sum(os.path.getsize(os.path.join(_d, f)) for f in os.listdir(_d)
                  if os.path.isfile(os.path.join(_d, f))) / 1e9
        if _gb > 0.01:
            _orphans.append((os.path.relpath(_d, MODEL_OUT), _gb))
if _orphans:
    print(f"\n⚠ {len(_orphans)} directory(ies) under models/ hold weights with NO completion marker, "
          f"{sum(g for _n, g in _orphans):.2f} GB in total:")
    for _n, _gb in _orphans:
        print(f"    {_n:<40} {_gb:>6.2f} GB")
    print(f"  These are unverified or half-written. Nothing reads them; they only cost space. Look "
          f"for a .unverified or .FAILED sidecar inside to see how far each got, then delete.")
if save_failed or no_model:
    raise RuntimeError(f"endpoint models missing: {len(save_failed)} branch(es) failed to save {save_failed}, "
                       f"{len(no_model)} complete branch(es) have none on Drive {no_model}. Their measurements "
                       f"are complete; see each branch's message above.")

# %%
T0_CELLS = []

_mf = os.path.join(OUT_DIR, "_manifest.json")
if not os.path.exists(_mf) or json.load(open(_mf, encoding="utf-8")).get("stamp") != STAMP:
    raise RuntimeError(f"no manifest for {STAMP} in {OUT_DIR}: the teacher cell has not run for this "
                       f"stamp. Re-run every cell from the config cell down.")
_t0_absent = [f"{a}_G{g}" for a, g, _ in T0_CELLS if (a, g) not in CKPTS]
if _t0_absent:
    raise RuntimeError(f"t = 0-only cells with NO init under {CKPT_ROOT}: {', '.join(_t0_absent)}. Build "
                       f"them with the init notebook first. ⛔ The grid is never measured smaller than it "
                       f"was designed.")
_t0_care = []
for _a, _g, _B in T0_CELLS:
    if _a in CARE_ARMS:
        _cv = (json.load(open(os.path.join(CKPTS[(_a, _g)], META_NAME), encoding="utf-8"))
               .get("mla_provenance") or {}).get("version")
        if _cv != CARE_REQUIRED_VERSION:
            _t0_care.append(f"{_a}_G{_g} (version {_cv!r})")
if _t0_care:
    raise RuntimeError(f"CARE t = 0 init(s) {_t0_care} were not built by the stable solver "
                       f"({CARE_REQUIRED_VERSION}). Delete those init directories on Drive and rebuild them "
                       f"with the init notebook. ⛔ Do not drop a CARE cell: the registry's cells are the grid.")
T0_DIR = os.path.join(OUT_DIR, "t0_only")
os.makedirs(T0_DIR, exist_ok=True)
T0_RUN_ID = f"{STAMP}-t0-{int(time.time())}"
T0_ROWS = {}
for arm, G, B in T0_CELLS:
    branch = f"{arm}_G{G}_t0only"
    log = os.path.join(T0_DIR, f"kd_{STAMP}_{branch}.jsonl")
    if os.path.exists(log):
        _prev = [json.loads(l) for l in open(log) if l.strip()]
        if _prev:
            print(f"=== {branch}: measured -- SKIPPED (run {_prev[-1]['run_id']})")
            T0_ROWS[branch] = _prev[-1]
            continue
    ckpt_dir = CKPTS[(arm, G)]
    print(f"\n=== {branch}  <-  {os.path.relpath(ckpt_dir, CKPT_ROOT)}  (t = 0 only) ===")
    student, meta, info = load_ckpt(ckpt_dir, expect_arm=arm)
    if meta.get("G") != G:
        raise RuntimeError(f"{ckpt_dir}: meta says G={meta.get('G')}, the grid says G={G}")
    student.to(DEV)
    student.eval()
    _t = time.time()
    k, k_orig, ce = eval_docs(student, EVAL)
    phase = BATTERY_PHASE.get(0, "periodic")
    print(f"    {branch:<20} battery ({phase}) at t = 0:")
    bat = run_battery(student, phase)
    ck, cv = cache_width_of(student)
    row = {"stamp": STAMP, "run_id": T0_RUN_ID, "arm": arm, "G": G, "B": B, "seed": None, "t0_only": True,
           "arch": meta["arch"], "wide": bool(info.get("wide", False)),
           "ckpt": os.path.relpath(ckpt_dir, CKPT_ROOT), "ckpt_fingerprint": meta["fingerprint"],
           "groups_searched": meta.get("groups_by_layer") is not None,
           "branch": branch, "step": 0, "tokens": 0,
           "kl_mean": float(k.mean()), "kl_se": float(k.std(ddof=1) / np.sqrt(len(k))),
           "n_doc": int(len(k)), "seq_len": SEQ_LEN, "kl_per_doc": [float(x) for x in k],
           "kd_teacher": TEACHER, "teacher_revision": TEACHER_REVISION, "teacher_dtype": TEACHER_DTYPE,
           "registry_sha": REGISTRY_SHA, "model_revision": MODEL_REVISION,
           "kl_orig_mean": float(k_orig.mean()),
           "kl_orig_se": float(k_orig.std(ddof=1) / np.sqrt(len(k_orig))),
           "kl_orig_per_doc": [float(x) for x in k_orig],
           "ce_mean": float(ce.mean()), "ce_se": float(ce.std(ddof=1) / np.sqrt(len(ce))),
           "ce_per_doc": [float(x) for x in ce],
           "eps_num": EPS_NUM, "corpus": CORPUS, "cache_k_width": ck, "cache_v_width": cv,
           "cache_fingerprint": cache_fingerprint(student),
           "battery": bat, "battery_phase": phase, "token_threshold": 0,
           "wall_s": round(time.time() - _t, 1)}
    with open(log, "a") as _f:
        _f.write(json.dumps(row) + "\n")
    T0_ROWS[branch] = row
    print(f"    {branch:<20} t = 0  KL {row['kl_mean']:8.4f} +/- {row['kl_se']:.4f}   KL_orig "
          f"{row['kl_orig_mean']:.4f}  CE {row['ce_mean']:.4f}   cache {ck}/{cv}")
    if bat:
        print(check_t0(f"{arm}_G{G}", bat)[0])
    del student
    torch.cuda.empty_cache()

print(f"\nt = 0 only, {len(T0_ROWS)} of {len(T0_CELLS)} cells -> {T0_DIR}")
print(f"  {'cell':<44} {'KL teacher':>10} {'KL orig':>9} {'CE':>7}")
for _b, _r in sorted(T0_ROWS.items(), key=lambda kv: kv[1]["kl_orig_mean"]):
    print(f"  {_b:<44} {_r['kl_mean']:>10.4f} {_r['kl_orig_mean']:>9.4f} {_r['ce_mean']:>7.4f}")

# %%
N_BOOT = 20000
brng = np.random.default_rng(20260914)
REFERENCE = ("mla_w64_ours_unrope_cost", None)

def rows_for(branch):
    if branch in ALL and ALL[branch]:
        rs = ALL[branch]
    else:
        p = os.path.join(OUT_DIR, f"kd_{STAMP}_{branch}.jsonl")
        if not os.path.exists(p):
            return None
        rs = [json.loads(l) for l in open(p) if l.strip()]
    if not rs:
        return None
    runs = {}
    for rr in rs:
        runs.setdefault(rr.get("run_id", "legacy"), []).append(rr)
    best = max(runs, key=lambda k: max(x["tokens"] for x in runs[k]))
    fps = {rr.get("ckpt_fingerprint") for rr in runs[best]}
    outside = sorted({str(rr.get("ckpt", "")) for rr in runs[best]
                      if str(rr.get("ckpt", "")).startswith("..")})
    if len(fps) > 1 or outside:
        raise RuntimeError(
            f"{branch}: run {best} is not a single line from one t=0 initialisation.\n"
            f"  starting fingerprints within this run: {sorted(f for f in fps if f)}\n"
            f"  rows whose checkpoint is OUTSIDE the init tree: {outside}\n"
            f"⛔ Every selector in this cell assumes one run is one line. If a branch was trained from "
            f"a SAVED endpoint, these rows are a CONTINUATION: the trajectory they imply was never "
            f"followed by any single run, the early thresholds belong to a different optimiser state, "
            f"and the token counts restart from zero. Do not relax this -- splice deliberately, with "
            f"the join verified by fingerprint, or report the segments separately.")
    if len(runs) > 1:
        print(f"    {branch}: {len(runs)} runs in the log; using {best} ({len(runs[best])} rows) and "
              f"DISCARDING {len(rs)-len(runs[best])} rows from the rest. Splicing runs would invent a "
              f"trajectory no run followed.")
    out = {}
    for rr in runs[best]:
        key = rr["token_threshold"] if rr.get("token_threshold") is not None else rr["tokens"]
        out[key] = rr
    return out

LOADED = {}
for arm, G in list(BRANCHES) + [("mha", 0)]:
    rs = rows_for(f"{arm}_G{G}_s{SEED_TRAIN}")
    if rs:
        LOADED[(arm, G)] = rs
if not LOADED:
    raise RuntimeError("no logs found -- run the training cell first")

_grids = {}
for (arm, G), rs in LOADED.items():
    for rr in rs.values():
        _grids.setdefault(tuple(rr.get("eval_at_tokens") or ()), []).append(f"{arm}_G{G}")
if len(_grids) > 1:
    print("⚠ ⚠ THE BRANCHES WERE NOT MEASURED ON THE SAME GRID:")
    for g, who in _grids.items():
        print(f"     {[t/1e6 for t in g]}M  <-  {sorted(set(who))}")
    print("     The differences below are computed only at thresholds the two branches SHARE, so they")
    print("     are real but the comparison is narrower than it looks. Say so in any figure.")
elif tuple(EVAL_AT_TOKENS) not in _grids and () not in _grids:
    print(f"⚠ the logs were written on grid {[t/1e6 for t in list(_grids)[0]]}M but this notebook now "
          f"declares {[t/1e6 for t in EVAL_AT_TOKENS]}M. The logs are authoritative; the tables below "
          f"use the thresholds present in them.")

print(f"KL(teacher||student), full vocabulary, n_doc={int(EVAL.shape[0])}, corpus={CORPUS}")
print(f"numerical floor EPS_NUM = {EPS_NUM:.3e}. No magnitude threshold is applied anywhere below.\n")
print(f"{'branch':<24}" + "".join(f"{t/1e6:>10.2f}M" for t in EVAL_AT_TOKENS))
print("-" * (24 + 11 * len(EVAL_AT_TOKENS)))
for (arm, G), rs in sorted(LOADED.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    cells = []
    for t in EVAL_AT_TOKENS:
        cells.append(f"{rs[t]['kl_mean']:>11.4f}" if t in rs else f"{'--':>11}")
    print(f"{arm+'_G'+str(G):<24}" + "".join(cells))

print(f"\nPaired difference  KL[arm] - KL[{REFERENCE[0]}]  at each G, {N_BOOT} resamples.")
print("PAIRED over documents: every arm is scored on the SAME docs, so an unpaired formula would be")
print("the wrong one. A POSITIVE value means the arm is FURTHER from the teacher than the reference.\n")
print(f"{'G':>3} {'arm':<18}{'tokens':>9} {'diff':>11} {'SE':>9} {'95% CI':>23} {'sign':>6} "
      f"{'MDE80':>8}  CI excl 0")
summary = []
for G in sorted({g for _a, g in LOADED}):
    ref = LOADED.get((REFERENCE[0], G))
    if ref is None:
        print(f"{G:>3}  no {REFERENCE[0]}_G{G} -- differences at this G are not computed")
        continue
    for arm in sorted({a for a, g in LOADED if g == G and a != REFERENCE[0]}):
        rs = LOADED[(arm, G)]
        for t in sorted(set(rs) & set(ref)):
            a = np.array(rs[t]["kl_per_doc"]); b = np.array(ref[t]["kl_per_doc"])
            if len(a) != len(b) or len(a) < 2:
                continue
            d = a - b
            e1, se = float(d.mean()), float(d.std(ddof=1)/np.sqrt(len(d)))
            boot = d[brng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            held = float(np.mean(np.sign(boot) == np.sign(e1)))
            mde80 = 2.8 * se
            excl = bool(lo > 0.0 or hi < 0.0)
            print(f"{G:>3} {arm:<18}{rs[t]['tokens']/1e6:>8.2f}M {e1:>+11.6f} {se:>9.6f}  "
                  f"[{lo:>+9.6f},{hi:>+9.6f}] {held*100:>5.1f}% {mde80:>8.4f}  "
                  f"{'yes' if excl else 'no'}")
            summary.append({"G": G, "arm": arm, "reference": REFERENCE[0], "seed": SEED_TRAIN,
                            "token_threshold": t, "tokens": rs[t]["tokens"],
                            "run_id": rs[t].get("run_id"), "run_id_ref": ref[t].get("run_id"),
                            "cache_k_width": rs[t].get("cache_k_width"),
                            "cache_k_width_ref": ref[t].get("cache_k_width"),
                            "diff": e1, "paired_se": se, "ci95": [float(lo), float(hi)],
                            "sign_stability": held, "mde80_from_se": float(mde80),
                            "ci_excludes_0": excl})

print(f"\nBattery, primary metric per task (acc_norm for openbookqa/hellaswag, acc elsewhere).")
print("⚠ Accuracy is CENSORED BELOW CHANCE: once an arm is at chance it stops moving and cannot rank")
print("  two arms that differ in damage. The `margin` in the JSON still orders them there, and KL has")
print("  no floor at all. Read this table WITH the KL table, never instead of it.\n")
_bt = [t for t in BATTERY_AT_TOKENS]
for (arm, G), rs in sorted(LOADED.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    have = [t for t in _bt if t in rs and rs[t].get("battery")]
    if not have:
        continue
    print(f"  {arm}_G{G}")
    print(f"    {'task':<15}{'chance':>8}" + "".join(f"{t/1e6:>11.2f}M" for t in have))
    for task in FEASIBILITY_TASKS:
        cells = []
        for t in have:
            b = rs[t]["battery"].get(task)
            cells.append(f"{b['primary_value']:>12.4f}" if b else f"{'--':>12}")
        print(f"    {task:<15}{CHANCE[task]:>8.3f}" + "".join(cells))

if summary:
    outp = os.path.join(OUT_DIR, f"kd_{STAMP}_summary.json")
    with open(outp, "w") as f:
        json.dump({"eps_num": EPS_NUM, "n_doc": int(EVAL.shape[0]), "seq_len": SEQ_LEN,
                   "n_boot": N_BOOT, "lr": LR, "batch": BATCH, "grad_ckpt": GRAD_CKPT,
                   "lr_schedule": "warmup_cosine", "warmup_frac": WARMUP_FRAC,
                   "kl_target": "KL(kd_teacher || student); kl_orig_* in the rows is "
                                "KL(original || student)",
                   "kd_teacher": TEACHER, "teacher_revision": TEACHER_REVISION,
                   "teacher_dtype": TEACHER_DTYPE, "registry_sha": REGISTRY_SHA,
                   "eps_num_is_against": "the ORIGINAL (the KL instrument cell)",
                   "lr_min_frac": LR_MIN_FRAC,
                   "corpus": CORPUS, "max_tokens": MAX_TOKENS, "eval_at_tokens": EVAL_AT_TOKENS,
                   "battery_at_tokens": BATTERY_AT_TOKENS, "stamp": STAMP, "seed": SEED_TRAIN,
                   "reference_arm": REFERENCE[0],
                   "branches": [f"{a}_G{g}" for a, g in sorted(LOADED, key=lambda x: (x[1], x[0]))],
                   "rows": summary}, f, indent=2)
    print(f"\nwrote {os.path.basename(outp)}")

if MAX_TOKENS < N_TOKENS_MIN:
    _e2_line = (
        f"⛔ It is also BELOW this experiment's own pre-registered floor: B-2 fixed N_TOKENS_MIN at "
        f"{N_TOKENS_MIN/1e6:.0f}M and section 5.2 says that below it, E2 is declared underpowered "
        f"BEFORE the run rather than after a null. That applies here automatically and is not a "
        f"judgement made after seeing these numbers. B-3 had already retired R(t) normalisation, so "
        f"E2 was largely off the critical path -- a mitigation, not an excuse.")
else:
    _e2_line = (
        f"✅ The budget MEETS B-2's N_TOKENS_MIN of {N_TOKENS_MIN/1e6:.0f}M, so section 5.2's ex-ante "
        f"power gate on E2 does NOT fire for this run.\n"
        f"⚠ That closes the BUDGET gate and nothing else. Section 7 admissibility still needs C2 and "
        f"C4, which have never been run, and C3 is open pending B-4. A met budget is not a "
        f"confirmatory result.")
print(f"""
HOW TO READ THIS, AND WHAT NOT TO CLAIM
---------------------------------------
Seed {SEED_TRAIN}, one run per branch. Differences are paired WITHIN a seed; re-run with a different
SEED_TRAIN for an independent replicate, and the spread across seeds is the TRAINING variance. With
one run per arm there is NO estimate of it, so a surviving gap cannot yet be separated from "this run
happened to go better". The bootstrap above covers EVALUATION sampling only.

MDE80 is the 80%-power floor computed from each row's OWN SE, printed beside the effect so the power
of that row is visible without gating it. A |diff| under its MDE80 is not powered at this n_doc --
a fact about the sampling budget, NOT about the gauge. SE goes as 1/sqrt(n_doc), so the same effect
crosses that line by scoring more documents and nothing else. No `verdict` is stored anywhere.

{MAX_TOKENS/1e6:.1f}M tokens is {PRETRAIN_FRAC:.5f}% of pretraining ({PRETRAIN_TOKENS/1e9:.0f}B). This
is a RATE measurement. It cannot reach recovery and no figure from it may be reported as if it could.
{_e2_line}
E1(0) -- the PRIMARY endpoint -- is an initialisation quantity, unaffected by the budget either way,
and it exists only if BOTH pool and gauge3B were in RUN_ARMS.
The literature's smallest measured recovery point is 0.025% of pretraining, so this budget
sits ~{0.025/PRETRAIN_FRAC:.0f}x below any published measurement: the "rate, not recovery" framing is a
one-sided bound and survives, but nothing here may be compared to a published recovery curve.

⚠ ON optwide SPECIFICALLY. At t=0 and G=8 it sits at the unconverted base on the easy tasks and gives
up ~19% of induction and ~10% of kv_synth_N50 -- the two long-range retrieval tasks, which is exactly
where a KV-cache method should be judged. It is NOT lossless, and lambada's +0.012 is inside one SE.
It also COLLAPSES by G=4 (induction 0.073 against the base's 0.845), so the wide latent buys roughly
one factor of two in cache, not immunity. The defensible claim is that optwide dominates every
alternative AT MATCHED CACHE -- the cache_k_width fields above are what make that comparison legitimate
-- and that it still costs the hard tasks. Not that it is free.
""")

_mha = LOADED.get(("mha", 0))
print(f"\nAgainst the MHA baseline (paired over docs; + = the arm is worse than full MHA):")
if _mha is None:
    print("  no mha_G0 log -- the mha control trains on the 5M grid only (the registry's cells)")
else:
    print(f"{'G':>3} {'arm':<24}{'tokens':>9} {'KL vs teacher':>16} {'KL vs original':>16} {'held-out CE':>14}")
    for (arm, G), rs in sorted(LOADED.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if arm == "mha":
            continue
        for t in sorted(set(rs) & set(_mha)):
            cells = []
            for key in ("kl_per_doc", "kl_orig_per_doc", "ce_per_doc"):
                a, b = rs[t].get(key), _mha[t].get(key)
                if a is None or b is None or len(a) != len(b):
                    cells.append(f"{'--':>16}")
                    continue
                d = np.array(a) - np.array(b)
                cells.append(f"{d.mean():>+9.4f}±{d.std(ddof=1) / np.sqrt(len(d)):.4f}")
            print(f"{G:>3} {arm:<24}{rs[t]['tokens']/1e6:>8.2f}M " + " ".join(cells))

# %%
if MAX_TOKENS == 5_000_000 and LR_PEAK == 2e-5 and SEED_TRAIN == GUARD_SEED:
    _absent = sorted(c for c in RUN_CELLS if c[0] != "mha" and c not in LOADED)
    if _absent:
        raise RuntimeError(f"the guard needs EVERY converted cell of the 5M grid; {len(_absent)} have no "
                           f"log: {', '.join(f'{a}_G{g}' for a, g in _absent)}. Finish them first.")
    per_G = {}
    for (arm, G), rs in sorted(LOADED.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if arm == "mha":
            continue
        need = [0, WARMUP_END_TOKENS, MAX_TOKENS]
        if any(t not in rs for t in need):
            raise RuntimeError(f"{arm}_G{G}: the guard needs rows at {need}; the log has "
                               f"{sorted(rs)[:5]}... -- is this branch complete?")
        k0 = np.array(rs[0]["kl_per_doc"])
        for t in need[1:]:
            d = np.array(rs[t]["kl_per_doc"]) - k0
            se = d.std(ddof=1) / np.sqrt(len(d))
            rose = bool(d.mean() > 2 * se)
            e = per_G.setdefault(str(G), {"tripped": False, "evidence": []})
            e["evidence"].append({"arm": arm, "token_threshold": t, "mean_rise": float(d.mean()),
                                  "se": float(se), "rose": rose})
            e["tripped"] = e["tripped"] or rose
    want = {str(g) for a, g in RUN_CELLS if a != "mha"}
    if set(per_G) != want:
        raise RuntimeError(f"the guard read budgets {sorted(per_G)}, the grid has {sorted(want)}: "
                           f"complete every 5M branch before deciding it")
    GUARD = {"rule": GUARD_RULE, "stamp": STAMP, "registry_sha": REGISTRY_SHA,
             "warmup_end_tokens": WARMUP_END_TOKENS, "per_G": per_G}
    with open(os.path.join(OUT_DIR, "_guard.json"), "w", encoding="utf-8") as _f:
        json.dump(GUARD, _f, indent=1)
    for g, e in sorted(per_G.items(), key=lambda kv: int(kv[0])):
        worst = max(e["evidence"], key=lambda x: x["mean_rise"] / max(x["se"], 1e-12))
        print(f"G={g:>2}  {'TRIPPED -> peak 2e-6' if e['tripped'] else 'clear   -> peak 2e-5'}   worst: "
              f"{worst['arm']} at {worst['token_threshold']:,} tokens, rise {worst['mean_rise']:+.4f} "
              f"(SE {worst['se']:.4f})")
    print(f"\nguard written: {os.path.join(OUT_DIR, '_guard.json')}")
else:
    print(f"no guard for {MODEL_TAG}: it trains its MLA group at 50M only, peak 2e-5 (MLA50, the authors 2026-10-06)")
