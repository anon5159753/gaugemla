import dt_mla_transmla as _base
import dt_mla_w16 as _w16

ARM = "mla_w16_transmla_sigma"
SCHEMA = _base.SCHEMA
VERSION = _base.VERSION
ROTARY_MODE = _base.ROTARY_MODE
BIAS_SET = list(_base.BIAS_SET)
mla_scale_of = _base.mla_scale_of
WIDTH_RULE = _w16.WIDTH_RULE

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    _w16.check_width(ARM, w)
    return _w16.relabel(_base.build_weights(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm),
                        _base, ARM)
