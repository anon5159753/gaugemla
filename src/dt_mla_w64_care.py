import dt_mla_care as _base
import dt_mla_w64 as _w64

ARM = "mla_w64_care_sigma"
SCHEMA = _base.SCHEMA
VERSION = _base.VERSION
ROTARY_MODE = _base.ROTARY_MODE
BIAS_SET = list(_base.BIAS_SET)
mla_scale_of = _base.mla_scale_of
WIDTH_RULE = _w64.WIDTH_RULE

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    _w64.check_width(ARM, w)
    return _w64.relabel(_base.build_weights(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm),
                        _base, ARM)
