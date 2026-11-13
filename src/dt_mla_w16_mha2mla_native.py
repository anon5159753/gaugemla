import dt_mla_gqa_native as _base
import dt_mla_mha2mla as _mm
import dt_mla_w16 as _w16

ARM = "mla_w16_mha2mla_native_s2norm"
SCHEMA = _base.SCHEMA
VERSION = _mm.VERSION
ROTARY_MODE = "shared"
BIAS_SET = list(_mm.BIAS_SET)
mla_scale_of = _mm.mla_scale_of
WIDTH_RULE = _w16.WIDTH_RULE

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    _w16.check_width(ARM, w)
    return _base.build_mha2mla(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, arm=ARM)
