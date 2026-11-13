import dt_mla_care as _ca
import dt_mla_gqa_native as _base
import dt_mla_transmla as _tm
import dt_mla_w16 as _w16

ARM = "mla_w16_care_native_sigma"
SCHEMA = _base.SCHEMA
VERSION = _ca.VERSION
ROTARY_MODE = "shared"
BIAS_SET = list(_ca.BIAS_SET)
mla_scale_of = _tm.mla_scale_of
WIDTH_RULE = _w16.WIDTH_RULE

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    _w16.check_width(ARM, w)
    return _base.build_care(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm, arm=ARM)
