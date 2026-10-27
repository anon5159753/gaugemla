import dt_mla_unrope_arms as _U

ARM = "mla_w64_ours_unrope_cost_fi"
ROTARY_ALLOC = "cost"
OUTPUT_METRIC = "identity"
WIDTH_RULE = "fixed:64"
SCHEMA = _U.SCHEMA
VERSION = _U.VERSION
ROTARY_MODE = _U.ROTARY_MODE
BIAS_SET = list(_U.BIAS_SET)
mla_scale_of = _U.mla_scale_of

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _U.build(ARM, ROTARY_ALLOC, True, model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm,
                    output_metric=OUTPUT_METRIC)

def self_check(wd):
    _U.self_check(ARM, wd)
