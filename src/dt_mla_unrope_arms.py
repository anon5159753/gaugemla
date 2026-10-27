import numpy as np

import dt_mla_unrope as _base
import dt_mla_w16 as _w16
import dt_mla_w64 as _w64
import gam_core as _gc

READOUT = "damped"
UNROPE = "all"
SCHEMA = _base.SCHEMA
VERSION = _base.VERSION
ROTARY_MODE = _base.ROTARY_MODE
BIAS_SET = list(_base.BIAS_SET)

def mla_scale_of(w, d_h):
    return float(d_h ** -0.5)

def _identity_fisher(model):
    L, H, d_k, d_v, _dm, _r = _gc.geometry(model)
    return (np.ascontiguousarray(np.broadcast_to(np.eye(d_k), (L, H, d_k, d_k)), dtype=np.float64),
            np.ascontiguousarray(np.broadcast_to(np.eye(d_v), (L, H, d_v, d_v)), dtype=np.float64))

def _identity_sigma(model):
    L, _H, _dk, _dv, dm, _r = _gc.geometry(model)
    eye = np.eye(dm + 1, dtype=np.float64)
    eye.flags.writeable = False
    return [eye] * L

def build(arm, rotary_alloc, fixed, model, *, w, d_c, G, fisher=None, sigma=None, s2norm=None,
          readout=READOUT, rotary_source="ours", output_metric="measured", input_metric="measured"):
    _fw = None
    if fixed:
        _fw = _w64 if fixed is True else {16: _w16}[fixed]
        _fw.check_width(arm, w)
    if output_metric not in ("measured", "identity"):
        raise ValueError(f"output_metric {output_metric!r} not in ('measured', 'identity')")
    if input_metric not in ("measured", "identity"):
        raise ValueError(f"input_metric {input_metric!r} not in ('measured', 'identity')")
    if output_metric == "identity":
        if fisher is not None:
            raise ValueError(f"{arm} builds with F := I; the measured Fisher must not reach it (adversarial C6)")
        fisher = _identity_fisher(model)
    if input_metric == "identity":
        if sigma is not None:
            raise ValueError(f"{arm} builds with Sigma := I; the measured Sigma must not reach it (adversarial C6)")
        sigma = _identity_sigma(model)
    out = _base.build_weights(model, w=w, d_c=d_c, G=G, fisher=fisher, sigma=sigma, s2norm=s2norm,
                              readout=readout, unrope=UNROPE, rotary_alloc=rotary_alloc, rotary_source=rotary_source)
    if out["arm"] != _base.ARM:
        raise AssertionError(f"dt_mla_unrope returned arm {out['arm']!r}, expected {_base.ARM!r}")
    out["arm"] = arm
    _rot = (f"rotary allocation {rotary_alloc!r} (D-A1)" if rotary_source == "ours"
            else f"rotary key TransMLA's (rotary_source {rotary_source!r}, @@UNROPE_TM@@)")
    if output_metric == "identity":
        out["provenance"]["fisher_sha256"] = None
    if input_metric == "identity":
        out["provenance"]["sigma_sha256"] = None
    if output_metric == input_metric == "identity":
        _rot += ", output metric F := I and input metric Sigma := I (no Fisher, no Sigma; @@INITABL@@ 2026-10-26)"
    elif output_metric == "identity":
        _rot += ", output metric F := I with Sigma measured (no Fisher; @@UNROPE_TM@@ 2026-10-14)"
    elif input_metric == "identity":
        _rot += ", input metric Sigma := I with F measured (no Sigma; @@INITABL@@ 2026-10-26)"
    out["provenance"]["text"] = (
        f"{arm}: {_base.ARM} built UNCHANGED ({_base.VERSION}) at read-out {readout!r}, unrope {UNROPE!r}, "
        f"{_rot}, w = {w}"
        + (f" (the fixed width, rule {_fw.WIDTH_RULE})" if fixed else " (the un-rope KL-objective width)")
        + ". -- " + out["provenance"]["text"])
    self_check(arm, out)
    return out

def self_check(arm, wd):
    if wd["arm"] != arm:
        raise AssertionError(f"arm {wd['arm']!r} != {arm!r}")
    _base.self_check(dict(wd, arm=_base.ARM))
