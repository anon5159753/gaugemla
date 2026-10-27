import dt_mla_unrope_arms as _U

ARM = "mla_ours_unrope_cost"
ROTARY_ALLOC = "cost"
WIDTH_RULE = "unrope_argmin"
SCHEMA = _U.SCHEMA
VERSION = _U.VERSION
ROTARY_MODE = _U.ROTARY_MODE
BIAS_SET = list(_U.BIAS_SET)
mla_scale_of = _U.mla_scale_of

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    return _U.build(ARM, ROTARY_ALLOC, False, model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm)

def self_check(wd):
    _U.self_check(ARM, wd)
