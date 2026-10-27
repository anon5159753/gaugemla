import dt_mla_ours

ARM = "mla_ours_fisher_sigma_eig"
SCHEMA = dt_mla_ours.SCHEMA
VERSION = "mla_ours_eig/1 (2026-10-01)"
ROTARY_MODE = dt_mla_ours.ROTARY_MODE
BIAS_SET = list(dt_mla_ours.BIAS_SET)
MLA_SCALE = dt_mla_ours.MLA_SCALE
mla_scale_of = dt_mla_ours.mla_scale_of
WIDTH_RULE = "eigen"

def build_weights(model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None):
    out = dt_mla_ours.build_weights(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm)
    if out["arm"] != dt_mla_ours.ARM:
        raise AssertionError(f"dt_mla_ours returned arm {out['arm']!r}, expected {dt_mla_ours.ARM!r}")
    out["arm"] = ARM
    out["version"] = VERSION
    out["provenance"]["text"] = (
        f"{ARM} ({VERSION}): GaugeMLA at the EIGEN-ENERGY widths (w = {w}, d_c = {d_c}; "
        f"DSTAR['mla'] proxy_m_by_layer, mu_m >= lam_(B-2m+1) + lam_(B-2m+2), half-up median). "
        f"Built by {dt_mla_ours.VERSION} unchanged: " + out["provenance"]["text"])
    self_check(out)
    return out

def self_check(wd):
    if wd["arm"] != ARM or wd["version"] != VERSION:
        raise AssertionError(f"arm / version {wd['arm']!r} / {wd['version']!r} != {ARM!r} / {VERSION!r}")
    dt_mla_ours.self_check(dict(wd, arm=dt_mla_ours.ARM))
