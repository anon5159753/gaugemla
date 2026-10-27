import dt_mla_ours as _base
import dt_mla_w64 as _w64

ARM = "mla_w64_ours_fisher_sigma"
SCHEMA = _base.SCHEMA
VERSION = _base.VERSION
ROTARY_MODE = _base.ROTARY_MODE
BIAS_SET = list(_base.BIAS_SET)
MLA_SCALE = _base.MLA_SCALE
mla_scale_of = _base.mla_scale_of
WIDTH_RULE = _w64.WIDTH_RULE

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    _w64.check_width(ARM, w)
    out = _w64.relabel(_base.build_weights(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm),
                       _base, ARM)
    self_check(out)
    return out

def self_check(wd):
    if wd["arm"] != ARM:
        raise AssertionError(f"arm {wd['arm']!r} != {ARM!r}")
    _base.self_check(dict(wd, arm=_base.ARM))
