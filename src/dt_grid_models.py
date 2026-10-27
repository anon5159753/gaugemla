import hashlib
import json

REGISTRY_VERSION = "grid-2026-10-04.3"

COMPARE_W = 64
COMPARE = ("mla_w64_ours_fisher_sigma", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm",
           "mla_w64_care_sigma")
KLW = "mla_ours_fisher_sigma"
DEC_FIXED = "decoupled_optwide_fisher_sigma_dkr64"
LONG = COMPARE + (KLW, DEC_FIXED)
SHORT = ("pool", "gauge3B", "procrustes_simk", "optimal", "optwide", "decoupled_optwide_median")
CONTROL = "mha"
PENDING_ARMS = {}
RETIRED_ARMS = {
    "mla_ours_fisher_sigma_eig": "D-R4 (2026-10-04): dropped for the KL-objective width",
    "mla_transmla_sigma": "superseded 2026-10-04 by mla_w64_transmla_sigma (the comparison is at w = 64)",
    "mla_mha2mla_s2norm": "superseded 2026-10-04 by mla_w64_mha2mla_s2norm (the comparison is at w = 64)",
    "mla_care_sigma": "superseded 2026-10-04 by mla_w64_care_sigma (the comparison is at w = 64)",
    "decoupled_optwide_fisher_sigma_dstar": (
        "superseded 2026-10-04 by decoupled_optwide_fisher_sigma_dkr64 (d_k^R = 64). DSTAR['fisher_sigma'] "
        "is still computed: c10's MLA cross-check needs it and reads these old inits, so KEEP them on Drive"),
}
OURS_FULL_W64 = ("mla_w64_ours_unrope_energy", "mla_w64_ours_unrope_cost")
OURS_FULL_KL = ("mla_ours_unrope_energy", "mla_ours_unrope_cost")
OURS_FULL = OURS_FULL_W64 + OURS_FULL_KL
SUBST_FULL = {"mla_w64_ours_fisher_sigma": OURS_FULL_W64, KLW: OURS_FULL_KL}
CARE = ("mla_care_sigma", "mla_w64_care_sigma")
FIXED_W = COMPARE + (DEC_FIXED,) + OURS_FULL_W64

D_A1_TRAINS = "cost"

def d_a1_trains(m):
    return m.get("d_a1", D_A1_TRAINS)

EXTRA_ARMS = ("mla_w64_ours_unrope_tmcopy", "mla_w64_ours_unrope_tmdamped",
              "mla_w64_ours_unrope_tmcopy_fi",
              "mla_w64_ours_unrope_cost_fi",
              "mla_w16_ours_unrope_cost_fi", "mla_w64_ours_unrope_cost_si", "mla_w16_ours_unrope_cost_si",
              "mla_w64_ours_unrope_cost_ii", "mla_w16_ours_unrope_cost_ii")

FISHER_LABELS = ("text", "model")

def fisher_labels(m):
    v = m.get("fisher_labels", "text")
    if v not in FISHER_LABELS:
        raise ValueError(f"fisher_labels {v!r} not in {FISHER_LABELS}")
    return v

FISHER_EXCLUDE_MAX = 0.25

def fisher_exclude_top(m):
    v = m.get("fisher_exclude_top", 0.0)
    if not (isinstance(v, float) and 0.0 <= v <= FISHER_EXCLUDE_MAX):
        raise ValueError(f"fisher_exclude_top {v!r} is not a float in [0, {FISHER_EXCLUDE_MAX}]")
    if v and fisher_labels(m) != "model":
        raise ValueError("fisher_exclude_top is defined on the true Fisher only (fisher_labels 'model')")
    return v

FISHER_LABEL_SEED = 20261014
FISHER_SEED_SPACING = 1000

def fisher_label_seeds(m):
    v = m.get("fisher_label_seeds", (FISHER_LABEL_SEED,))
    if (not isinstance(v, tuple) or not v or len(set(v)) != len(v)
            or any(not isinstance(s, int) or isinstance(s, bool) or (s - FISHER_LABEL_SEED) % FISHER_SEED_SPACING
                   for s in v)):
        raise ValueError(f"fisher_label_seeds {v!r} is not a tuple of distinct FISHER_LABEL_SEED + k {FISHER_SEED_SPACING}")
    if "fisher_label_seeds" in m and fisher_labels(m) != "model":
        raise ValueError("fisher_label_seeds is defined on the true Fisher only (fisher_labels 'model')")
    return v

def t0_only(m):
    v = m.get("t0_only", False)
    if not isinstance(v, bool):
        raise ValueError(f"t0_only {v!r} is not a bool")
    return v

GRAD_CLIP_STANDARD = 1.0

def grad_clip(m):
    v = m.get("grad_clip")
    if v is not None and not (isinstance(v, float) and v > 0.0):
        raise ValueError(f"grad_clip {v!r} is not None or a positive float")
    return v

def save_final(m):
    v = m.get("save_final", False)
    if not isinstance(v, bool):
        raise ValueError(f"save_final {v!r} is not a bool")
    return v

def comparison_width(m):
    return m.get("w_compare", COMPARE_W)

def at_width(arm, w):
    if w == COMPARE_W or arm not in FIXED_W:
        return arm
    if arm == DEC_FIXED:
        return arm.replace(f"_dkr{COMPARE_W}", f"_dkr{w}")
    return arm.replace(f"mla_w{COMPARE_W}_", f"mla_w{w}_")

GROUPING = ("pool", "gauge3B", "procrustes_simk", "optimal")
MLA = COMPARE + (KLW,) + OURS_FULL
DECOUPLED = (DEC_FIXED, "decoupled_optwide_median")

TOKENS_LONG = (5_000_000, 50_000_000)
TOKENS_SHORT = (5_000_000,)
TRAIN_PLAN_MLA50 = "mla50"
TOKENS_MLA50 = (50_000_000,)
TOKENS_MLA20_50 = (20_000_000, 50_000_000)
INIT_PLAN_MLA_ONLY = "mla_only"
MLA_ONLY_KEEP = MLA + (DEC_FIXED,)
INIT_BUDGETS_TRAINED = "trained"
TOKENS_UR5M = (5_000_000,)
FRACS =(0.5, 0.25, 0.125, 0.0625)
TRAINED_FRACS = (0.25, 0.125, 0.0625)

MODELS = {
    "p410m": dict(
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin",
        regression=dict(dstar_built={1: 8, 2: 16, 4: 36, 8: 134}, mla_w={1: 20, 2: 30, 4: 86, 8: 208}),
        init_layout="legacy-G",
        assumes={"D-P1": "reuse the existing inits under kvtying/init (recommended; not yet decided)"},
    ),
    "slm1p6b": dict(
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        train_tokens=TOKENS_MLA20_50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
    ),
    "tl1b": dict(
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
    ),
    "sm360m": dict(
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
    ),
    "mn4b": dict(
        hf_id="nvidia/Minitron-4B-Base", revision="3478fb8d5e0a4347e4ab08727f9f95f33f97df3b",
        family="llama_layout", model_type="nemotron", role="open:D-R3",
        L=32, dm=3072, H=24, Hkv=8, d_k=128, r=64, rope_base=10000, max_ctx=4096, vocab=256000,
        qkv_bias=False, o_bias=False, norm_kind="ln1p", tie_word_embeddings=False,
        tokenizer_oid="4278610cd0b2caac512e5d7c2d0a3c3916f75ba7",
        shapes_checked=False,
        teacher=dict(hf_id="nvidia/Minitron-8B-Base", revision="e2c30f2d4c8751f21b33be2b403d10d5f373ffc1",
                     dtype="float32", vocab=256000, tokenizer_oid="4278610cd0b2caac512e5d7c2d0a3c3916f75ba7"),
        kl_width_rule="argmin", init_layout="B",
    ),
    "olmo1b": dict(
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
    ),
    "p12b": dict(
        hf_id="EleutherAI/pythia-12b", revision="bb1e3e710cdf6b524461d543cfb5ba773f0a81b6",
        family="neox", model_type="gpt_neox", role="t0",
        L=36, dm=5120, H=40, Hkv=40, d_k=128, r=32, rope_base=10000, max_ctx=2048, vocab=50688,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=None, kl_width_rule="argmin", init_layout="B",
    ),
    "q0p5b": dict(
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
    ),
    "l3_1b": dict(
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
    ),
    "p410m_ur": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
    ),
    "slm1p6b_ur": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
    ),
    "p410m_ur5m": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        train_tokens=TOKENS_UR5M,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost", "mla_ours_unrope_cost"),
    ),
    "slm1p6b_ue": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_energy",),
        d_a1="energy",
    ),
    "slm1p6b_tm": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_tmcopy", "mla_w64_ours_unrope_tmdamped"),
        arms_only=("mla_w64_ours_unrope_tmcopy", "mla_w64_ours_unrope_tmdamped"),
    ),
    "slm1p6b_tmi": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_tmcopy_fi",),
        arms_only=("mla_w64_ours_unrope_tmcopy_fi",),
    ),
    "slm1p6b_uri": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_fi",),
        arms_only=("mla_w64_ours_unrope_cost_fi",),
    ),
    "slm1p6b_urt": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
    ),
    "slm1p6b_urx5": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.05,
    ),
    "slm1p6b_urx2": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.02,
    ),
    "slm1p6b_urx1": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "p410m_uri": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_fi",),
        arms_only=("mla_w64_ours_unrope_cost_fi",),
    ),
    "p410m_urx1": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "tl1b_tx1": dict(
        base="tl1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "sm360m_tx1": dict(
        base="sm360m",
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "olmo1b_tx1": dict(
        base="olmo1b",
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "q0p5b_tx1": dict(
        base="q0p5b",
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "l3_1b_tx1": dict(
        base="l3_1b",
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "slm1p6b_urx1_seeds": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "p410m_urx1_seeds": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "tl1b_tx1_seeds": dict(
        base="tl1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "sm360m_tx1_seeds": dict(
        base="sm360m",
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "olmo1b_tx1_seeds": dict(
        base="olmo1b",
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "q0p5b_tx1_seeds": dict(
        base="q0p5b",
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w64_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "l3_1b_tx1_seeds": dict(
        base="l3_1b",
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "p410m_urx1_5m": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        train_tokens=TOKENS_UR5M,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost", "mla_ours_unrope_cost", "decoupled_optwide_fisher_sigma_dkr64"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
    ),
    "p410m_gc": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm", "mla_w64_care_sigma"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "slm1p6b_gc": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("mla_w64_ours_unrope_cost", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm", "mla_w64_care_sigma"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "tl1b_gc": dict(
        base="tl1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost", "mla_w16_transmla_sigma", "mla_w16_mha2mla_s2norm"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "sm360m_gc": dict(
        base="sm360m",
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost", "mla_w16_transmla_sigma", "mla_w16_mha2mla_s2norm"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "olmo1b_gc": dict(
        base="olmo1b",
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w64_ours_unrope_cost", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm", "mla_w64_care_sigma"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "q0p5b_gc": dict(
        base="q0p5b",
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w64_ours_unrope_cost", "mla_w64_transmla_sigma", "mla_w64_mha2mla_s2norm", "mla_w64_care_sigma"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "l3_1b_gc": dict(
        base="l3_1b",
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        arms_only=("mla_w16_ours_unrope_cost", "mla_w16_transmla_sigma", "mla_w16_mha2mla_s2norm"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
        save_final=True,
    ),
    "p410m_gc5m": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_tokens=TOKENS_UR5M,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        arms_only=("pool", "gauge3B", "procrustes_simk", "optimal", "optwide", "decoupled_optwide_fisher_sigma_dkr64",
                   "mla_w64_ours_unrope_cost", "mla_ours_unrope_cost"),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        grad_clip=GRAD_CLIP_STANDARD,
    ),
    "slm1p6b_urx1_fi": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_fi",),
        arms_only=("mla_w64_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "slm1p6b_urx1_si": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_si",),
        arms_only=("mla_w64_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "p410m_urx1_fi": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_fi",),
        arms_only=("mla_w64_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "p410m_urx1_si": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_si",),
        arms_only=("mla_w64_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "tl1b_tx1_fi": dict(
        base="tl1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_fi",),
        arms_only=("mla_w16_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "tl1b_tx1_si": dict(
        base="tl1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_si",),
        arms_only=("mla_w16_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "sm360m_tx1_fi": dict(
        base="sm360m",
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_fi",),
        arms_only=("mla_w16_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "sm360m_tx1_si": dict(
        base="sm360m",
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_si",),
        arms_only=("mla_w16_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "olmo1b_tx1_fi": dict(
        base="olmo1b",
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w64_ours_unrope_cost_fi",),
        arms_only=("mla_w64_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "olmo1b_tx1_si": dict(
        base="olmo1b",
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w64_ours_unrope_cost_si",),
        arms_only=("mla_w64_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "q0p5b_tx1_fi": dict(
        base="q0p5b",
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w64_ours_unrope_cost_fi",),
        arms_only=("mla_w64_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "q0p5b_tx1_si": dict(
        base="q0p5b",
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w64_ours_unrope_cost_si",),
        arms_only=("mla_w64_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "l3_1b_tx1_fi": dict(
        base="l3_1b",
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_fi",),
        arms_only=("mla_w16_ours_unrope_cost_fi",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "l3_1b_tx1_si": dict(
        base="l3_1b",
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_si",),
        arms_only=("mla_w16_ours_unrope_cost_si",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        fisher_label_seeds=(20261014, 20262014, 20263014, 20264014, 20265014),
        t0_only=True,
    ),
    "slm1p6b_urx1_ii": dict(
        base="slm1p6b",
        hf_id="stabilityai/stablelm-2-1_6b", revision="f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef",
        family="llama_layout", model_type="stablelm", role="trained",
        L=24, dm=2048, H=32, Hkv=32, d_k=64, r=16, rope_base=10000, max_ctx=4096, vocab=100352,
        qkv_bias=True, o_bias=False, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="1143a0b36a2a29335a0387736659842689db19f0",
        teacher=dict(hf_id="stabilityai/stablelm-2-12b", revision="74dd3f536b71d18aa122a11af0d1b83105cb2d38",
                     dtype="float32", vocab=100352, tokenizer_oid="ab53dcfb233a308e5045eed71cfca31afdf32d74",
                     tokenizer_equal="functional"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_ii",),
        arms_only=("mla_w64_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "p410m_urx1_ii": dict(
        base="p410m",
        hf_id="EleutherAI/pythia-410m", revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        family="neox", model_type="gpt_neox", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=16, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=True, o_bias=True, norm_kind="ln_affine", tie_word_embeddings=False,
        tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc",
        teacher=dict(hf_id="EleutherAI/pythia-2.8b", revision="2a259cdd96a4beb1cdf467512e3904197345f6a9",
                     dtype="float32", vocab=50304, tokenizer_oid="f74dfbfab8f97770a87769c739fb080c21c8bacc"),
        kl_width_rule="argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        partial_unrope=True,
        extra_arms=("mla_w64_ours_unrope_cost_ii",),
        arms_only=("mla_w64_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "tl1b_tx1_ii": dict(
        base="tl1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        revision="59f6f375b26bde864a6ca194a9a3044570490064",
        family="llama_layout", model_type="llama", role="trained",
        L=22, dm=2048, H=32, Hkv=4, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=dict(hf_id="meta-llama/Llama-2-7b-hf", revision="01c7f73d771dfac7d292323805ebc428287df4f9",
                     dtype="float32", vocab=32000, tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
                     gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_ii",),
        arms_only=("mla_w16_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "sm360m_tx1_ii": dict(
        base="sm360m",
        hf_id="HuggingFaceTB/SmolLM-360M", revision="59f7ef243ee09a72cbc14cb054393a3e3b771d41",
        family="llama_layout", model_type="llama", role="trained",
        L=32, dm=960, H=15, Hkv=5, d_k=64, r=64, rope_base=10000, max_ctx=2048, vocab=49152,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd",
        teacher=dict(hf_id="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                     dtype="float32", vocab=49152, tokenizer_oid="f922b1797f0c88e71addc8393787831f2477a4bd"),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_ii",),
        arms_only=("mla_w16_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "olmo1b_tx1_ii": dict(
        base="olmo1b",
        hf_id="allenai/OLMo-1B-hf", revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
        family="llama_layout", model_type="olmo", role="trained",
        L=16, dm=2048, H=16, Hkv=16, d_k=128, r=128, rope_base=10000, max_ctx=2048, vocab=50304,
        qkv_bias=False, o_bias=False, norm_kind="ln_noaffine", tie_word_embeddings=True, clip_qkv=None,
        tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed",
        teacher=dict(hf_id="allenai/OLMo-7B-hf", revision="11fb3186a2e4f681edea621fa8b4345147a9db6a",
                     dtype="float32", vocab=50304, tokenizer_oid="21dabc8dd2144c1d63d5f10334b43d3470dd4aed"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w64_ours_unrope_cost_ii",),
        arms_only=("mla_w64_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "q0p5b_tx1_ii": dict(
        base="q0p5b",
        hf_id="Qwen/Qwen1.5-0.5B", revision="8f445e3628f3500ee69f24e1303c9f10f5342a39",
        family="llama_layout", model_type="qwen2", role="trained",
        L=24, dm=1024, H=16, Hkv=16, d_k=64, r=64, rope_base=1000000, max_ctx=32768, vocab=151936,
        qkv_bias=True, o_bias=False, norm_kind="rms", tie_word_embeddings=True,
        tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
        teacher=dict(hf_id="Qwen/Qwen1.5-4B", revision="a66363a0c24e2155c561e4b53c658b1d3965474e",
                     dtype="float32", vocab=151936, tokenizer_oid="33ea6c72ebb92a237fa2bdf26c5ff16592efcdae"),
        kl_width_rule="unrope_argmin", init_layout="B",
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w64_ours_unrope_cost_ii",),
        arms_only=("mla_w64_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "l3_1b_tx1_ii": dict(
        base="l3_1b",
        hf_id="meta-llama/Llama-3.2-1B", revision="4e20de362430cd3b72f300e6b0f18e50e7166e08", gated=True,
        family="llama_layout", model_type="llama", role="trained",
        L=16, dm=2048, H=32, Hkv=8, d_k=64, r=64, rope_base=500000, max_ctx=131072, vocab=128256,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=True, rope_scaling="llama3",
        tokenizer_oid="5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        teacher=dict(hf_id="meta-llama/Llama-3.1-8B", revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                     dtype="float32", vocab=128256, tokenizer_oid="f916e71031fa08f3c6ef1680a590c15b52d3cdd9",
                     tokenizer_equal="functional", gated=True),
        kl_width_rule="unrope_argmin", init_layout="B",
        w_compare=16,
        train_plan=TRAIN_PLAN_MLA50,
        init_plan=INIT_PLAN_MLA_ONLY,
        init_budgets=INIT_BUDGETS_TRAINED,
        extra_arms=("mla_w16_ours_unrope_cost_ii",),
        arms_only=("mla_w16_ours_unrope_cost_ii",),
        fisher_labels="model",
        fisher_exclude_top=0.01,
        t0_only=True,
    ),
    "l2_13b": dict(
        hf_id="meta-llama/Llama-2-13b-hf", revision="5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1", gated=True,
        family="llama_layout", model_type="llama", role="retired",
        L=40, dm=5120, H=40, Hkv=40, d_k=128, r=128, rope_base=10000, max_ctx=4096, vocab=32000,
        qkv_bias=False, o_bias=False, norm_kind="rms", tie_word_embeddings=False,
        tokenizer_oid="a6e931b92caff4c79c5c56282f1e89569a0ae558",
        teacher=None, kl_width_rule="unrope_argmin", init_layout="B",
    ),
}
ORDER = ("p410m", "slm1p6b", "tl1b", "sm360m", "mn4b", "olmo1b", "q0p5b", "l3_1b", "p12b",
         "p410m_ur", "slm1p6b_ur",
         "p410m_ur5m",
         "slm1p6b_ue",
         "slm1p6b_tm", "slm1p6b_tmi",
         "slm1p6b_uri",
         "slm1p6b_urt",
         "p410m_uri",
         "slm1p6b_urx5", "slm1p6b_urx2", "slm1p6b_urx1",
         "p410m_urx1", "tl1b_tx1", "sm360m_tx1", "olmo1b_tx1", "q0p5b_tx1", "l3_1b_tx1",
         "slm1p6b_urx1_seeds", "p410m_urx1_seeds", "tl1b_tx1_seeds", "sm360m_tx1_seeds",
         "olmo1b_tx1_seeds", "q0p5b_tx1_seeds", "l3_1b_tx1_seeds",
         "p410m_urx1_5m",
         "p410m_gc", "slm1p6b_gc", "tl1b_gc", "sm360m_gc", "olmo1b_gc", "q0p5b_gc", "l3_1b_gc",
         "p410m_gc5m",
         "slm1p6b_urx1_fi", "p410m_urx1_fi", "tl1b_tx1_fi", "sm360m_tx1_fi", "olmo1b_tx1_fi", "q0p5b_tx1_fi",
         "l3_1b_tx1_fi",
         "slm1p6b_urx1_si", "p410m_urx1_si", "tl1b_tx1_si", "sm360m_tx1_si", "olmo1b_tx1_si", "q0p5b_tx1_si",
         "l3_1b_tx1_si",
         "slm1p6b_urx1_ii", "p410m_urx1_ii", "tl1b_tx1_ii", "sm360m_tx1_ii", "olmo1b_tx1_ii", "q0p5b_tx1_ii",
         "l3_1b_tx1_ii")

CONSTANTS = dict(long=LONG, short=SHORT, control=CONTROL, pending=PENDING_ARMS, retired=RETIRED_ARMS,
                 compare=COMPARE, klw=KLW, dec_fixed=DEC_FIXED, w_compare=COMPARE_W, fixed_w=FIXED_W,
                 ours_full=OURS_FULL, ours_full_w64=OURS_FULL_W64, ours_full_kl=OURS_FULL_KL,
                 subst_full=SUBST_FULL, care=CARE, grouping=GROUPING, mla=MLA, decoupled=DECOUPLED,
                 tokens_long=TOKENS_LONG, tokens_short=TOKENS_SHORT, fracs=FRACS, trained=TRAINED_FRACS)

def registry_sha(tag):
    blob = json.dumps({"v": REGISTRY_VERSION, "tag": tag, "model": MODELS[tag], "constants": CONSTANTS,
                       "budgets": budgets(tag), "cells": cells(tag)},
                      sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def full_rope(m):
    return m["r"] == m["d_k"]

def is_gqa(m):
    return m["Hkv"] != m["H"]

def source_cache(m):
    return 2 * m["Hkv"] * m["d_k"]

def budgets(tag):
    m = MODELS[tag]
    S, unit = source_cache(m), 2 * m["d_k"]
    out = []
    for f in FRACS:
        B = int(round(f * S))
        out.append(dict(label=f"{f * 100:g}%", frac=f, B=B))
    for b in out:
        g = b["B"] / unit
        b["G"] = int(g) if g >= 1 and float(g).is_integer() else None
        b["star"] = b["G"] is None
        b["trained"] = b["frac"] in TRAINED_FRACS
        b["one_kv"] = b["B"] == unit
    return out

def _grouping_ok(m, G):
    if G is None:
        return False, "below one KV head or not a whole number of KV heads (no grouping exists)"
    if m["H"] % G:
        return False, f"{G} groups do not partition {m['H']} query heads"
    if is_gqa(m) and m["Hkv"] % G:
        return False, f"{G} groups are not a coarsening of the source's {m['Hkv']} KV groups"
    return True, ""

def _rot(m, arm, b):
    if arm in FIXED_W or arm in m.get("extra_arms", ()):
        w = comparison_width(m)
        return f"fixed:{w}", w
    if arm == KLW:
        return m["kl_width_rule"], ((m.get("regression") or {}).get("mla_w") or {}).get(b["G"])
    if arm in OURS_FULL_KL:
        return "unrope_argmin", None
    return None, None

def cells(tag):
    m = MODELS[tag]
    fisher_labels(m)
    fisher_exclude_top(m)
    fisher_label_seeds(m)
    t0o = t0_only(m)
    grad_clip(m)
    save_final(m)
    role, full, gqa = m["role"], full_rope(m), is_gqa(m)
    W = comparison_width(m)
    mla50 = m.get("train_plan") == TRAIN_PLAN_MLA50
    mla_only = m.get("init_plan") == INIT_PLAN_MLA_ONLY
    trained_only = m.get("init_budgets") == INIT_BUDGETS_TRAINED
    unrope_partial = bool(m.get("partial_unrope")) and not full
    out = []

    def add(arm, b, init, train=(), why="", degenerate=None, train_why=""):
        rot_rule, rot_w = _rot(m, arm, b) if init == "BUILD" else (None, None)
        arm = at_width(arm, W)
        out.append(dict(arm=arm, label=b["label"], B=b["B"], G=b["G"], star=b["star"], init=init,
                        why=why, train=tuple(train), train_why=train_why, degenerate=degenerate,
                        rot_rule=rot_rule, rot_w=rot_w))

    arms = list(LONG) + list(SHORT) + list(PENDING_ARMS)
    if full or unrope_partial:
        arms = arms + list(OURS_FULL)
    xa = tuple(m.get("extra_arms", ()))
    if any(a not in EXTRA_ARMS for a in xa) or len(set(xa)) != len(xa):
        raise ValueError(f"{tag}: extra_arms {xa} must be distinct members of EXTRA_ARMS {EXTRA_ARMS}")
    _xw = [a for a in xa if not a.startswith(f"mla_w{W}_")]
    if _xw:
        raise ValueError(f"{tag}: the extra arms {_xw} are not w = {W} arms; this model's comparison width is {W}")
    arms = arms + list(xa)
    for b in budgets(tag):
        for arm in arms:
            _long = arm in LONG or arm in OURS_FULL or arm in xa
            tokens = TOKENS_LONG if _long else TOKENS_SHORT
            if mla50:
                tokens = m.get("train_tokens", TOKENS_MLA50) if _long else ()
            elif "train_tokens" in m:
                tokens = tuple(t for t in tokens if t in m["train_tokens"])
            init, why, degenerate = "BUILD", "", None
            if (arm in FIXED_W or arm in xa) and b["B"] <= W:
                init, why = "NOT_REPRESENTABLE", (f"the fixed rotary width {W} leaves no latent at "
                                                  f"B = {b['B']} (D-R12, decided 2026-10-04)")
            elif arm in SUBST_FULL and full:
                init, why = ("SUBSTITUTED:" + "|".join(at_width(a, W) for a in SUBST_FULL[arm]),
                             "r = d_k: dt_mla_ours refuses (dt_mla_ours.py:334)")
            elif arm in SUBST_FULL and unrope_partial:
                init, why = ("SUBSTITUTED:" + "|".join(at_width(a, W) for a in SUBST_FULL[arm]),
                             "partial RoPE, overflow un-roped (@@UNROPE_PARTIAL@@, the authors 2026-10-12): dt_mla_ours drops "
                             "the rotary rows its cache does not hold; TransMLA, CARE and MHA2MLA un-rope them")
            elif arm in PENDING_ARMS:
                init, why = "PENDING:" + PENDING_ARMS[arm], "the arm itself waits on a decision"
            elif arm in GROUPING:
                ok, reason = _grouping_ok(m, b["G"])
                if not ok:
                    init, why = "NOT_REPRESENTABLE", reason
                    if b["G"] is None and b["B"] > 2 * m["d_k"]:
                        init, why = "PENDING:D-STAR", reason + "; asymmetric K/V groups might represent it"
            elif arm == "optwide" and b["one_kv"]:
                init, why = "DEDUP:optimal", "at one KV head optwide is optimal's construction"
            elif arm in CARE and gqa:
                init, why = "PENDING:D-CG1", "CARE's Σ-port equivalence is MHA-only (dt_mla_care.py:423)"
            elif arm in CARE and role == "t0" and m["dm"] >= 5000:
                init, why = "PENDING:D-L1", "CARE at 12-13B: ~30-40 h per cell on CPU float64 (port estimate)"
            if init == "BUILD" and full:
                if arm == "gauge3B":
                    degenerate = ("full RoPE: the key gauge has d_k parameters per head (MODEL_GRID_PLAN §4); "
                                  "its gain over pool on the keys is predicted small, not exactly zero")
                elif arm in DECOUPLED:
                    degenerate = "EXACT: full RoPE: the key share of the latent Gram is 0 (MODEL_GRID_PLAN §4)"
            if (mla_only and arm not in MLA_ONLY_KEEP and arm not in xa
                    and (init == "BUILD" or init.startswith("DEDUP:"))):
                init, why, degenerate = "OMITTED:MLA_ONLY", (
                    "MLA_ONLY decided 2026-10-08 (the authors): \"we're not running the other arms, just the MLA arms\"; "
                    f"would be {init}"), None
            if trained_only and not b["trained"] and (init == "BUILD" or init.startswith("DEDUP:")):
                init, why, degenerate = "OMITTED:TRAINED_ONLY", (
                    "TRAINED_ONLY decided 2026-10-11 (the authors): \"the init needs to reflect it and not build so many arms\"; "
                    f"{b['label']} is not trained; would be {init}"), None
            if (trained_only and arm in OURS_FULL and not arm.endswith("_" + d_a1_trains(m))
                    and (init == "BUILD" or init.startswith("DEDUP:"))):
                init, why, degenerate = "OMITTED:TRAINED_ONLY", (
                    "TRAINED_ONLY (the authors 2026-10-11, the modern pair): only the branches that train are built; this "
                    f"rotary allocation is t = 0 only under D-A1; would be {init}"), None
            if ("arms_only" in m and at_width(arm, W) not in m["arms_only"]
                    and (init == "BUILD" or init.startswith("DEDUP:"))):
                init, why, degenerate = "OMITTED:ARMS_ONLY", (
                    "ARMS_ONLY (the authors 2026-10-12, \"so that should be 6 runs in total\"): this tag builds only "
                    f"{', '.join(m['arms_only'])}; would be {init}"), None
            train, twhy = (), ""
            if init != "BUILD":
                twhy = "not built"
            elif role == "t0":
                twhy = "T0_ONLY: t = 0-only model"
            elif role.startswith(("open:", "pending:")):
                twhy = "PENDING:" + role.split(":", 1)[1]
            elif not b["trained"]:
                twhy = "T0_ONLY: 50% is t = 0 only"
            elif t0o:
                twhy = "T0_ONLY: t0_only (the true Fisher's seed spread, @@TFSEED@@)"
            elif arm in OURS_FULL and not arm.endswith("_" + d_a1_trains(m)):
                twhy = (f"T0_ONLY: D-A1 decided 2026-10-05 (the authors): the {d_a1_trains(m)} rotary allocation trains, "
                        f"this one is measured at t = 0 only")
            elif not tokens:
                twhy = ("T0_ONLY: MLA50 decided 2026-10-06 (the authors): only p410m trains the short group; this model "
                        "trains its MLA group at 50M only")
            else:
                train = tokens
                if m["teacher"] and m["teacher"].get("gated"):
                    twhy = "the teacher is GATED: its weights need access on the account before this trains"
            add(arm, b, init, train, why, degenerate, twhy)
    if role == "trained" and mla50:
        ctl_train, ctl_why = (), ("NOT_TRAINED: MLA50 decided 2026-10-06 (the authors): only p410m trains the control; its "
                                  "t = 0 is the run's reference (KL of the teacher to the original)")
    elif role == "trained":
        ctl_train, ctl_why = tuple(t for t in TOKENS_SHORT if t in m.get("train_tokens", TOKENS_SHORT)), ""
        if m["teacher"] and m["teacher"].get("gated"):
            ctl_why = "the teacher is GATED: its weights need access on the account before this trains"
    elif role.startswith(("open:", "pending:")):
        ctl_train, ctl_why = (), "PENDING:" + role.split(":", 1)[1]
    else:
        ctl_train, ctl_why = (), "T0_ONLY: t = 0-only model"
    out.append(dict(arm=CONTROL, label="src", B=source_cache(m), G=m["Hkv"], star=False, init="BUILD",
                    why="the unconverted model (no checkpoint: built in the run cell)",
                    train=ctl_train, train_why=ctl_why, degenerate=None, rot_rule=None, rot_w=None))
    if role.startswith("pending:"):
        d = "PENDING:" + role.split(":", 1)[1]
        for c in out:
            c["why"] = f"the whole model waits on {d[8:]}; would be {c['init']}" + (f" ({c['why']})" if c["why"] else "")
            c["init"], c["train"], c["train_why"] = d, (), d
            c["rot_rule"] = c["rot_w"] = None
    return out

def branch_counts(tag):
    n = {}
    for c in cells(tag):
        for t in c["train"]:
            n[t] = n.get(t, 0) + 1
    return n

def _print(tag):
    m = MODELS[tag]
    print(f"\n=== {tag}: registry sha {registry_sha(tag)[:16]}")
    print(f"    {m['hf_id']} @ {str(m['revision'])[:12]}  role={m['role']}  family={m['family']}  "
          f"H={m['H']} Hkv={m['Hkv']} d_k={m['d_k']} r={m['r']}  "
          f"teacher={m['teacher']['hf_id'] if m['teacher'] else None}")
    print("    budgets: " + ", ".join(f"{b['label']} B={b['B']} G={b['G']}{'✱' if b['star'] else ''}"
                                       f"{'' if b['trained'] else ' (t0)'}" for b in budgets(tag)))
    width = max(len(a) for a in LONG + SHORT + tuple(PENDING_ARMS) + OURS_FULL) + 1
    for c in cells(tag):
        tr = "+".join(f"{t // 1_000_000}M" for t in c["train"]) or "-"
        note = c["why"] or c["train_why"]
        if c["degenerate"]:
            note = (note + "; " if note else "") + "DEGENERATE: " + c["degenerate"]
        print(f"    {c['arm']:<{width}} {c['label']:>6} B={c['B']:<5} {c['init']:<26} train {tr:<7} {note}")
    print(f"    branches: { {f'{k // 1_000_000}M': v for k, v in sorted(branch_counts(tag).items())} }")

if __name__ == "__main__":
    import sys
    tags = sys.argv[1:] or list(ORDER)
    print(REGISTRY_VERSION)
    for t in tags:
        _print(t)
