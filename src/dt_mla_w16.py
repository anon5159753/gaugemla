FIXED_W = 16
WIDTH_RULE = "fixed:16"

def check_width(arm, w):
    if w != FIXED_W:
        raise ValueError(f"{arm} is the fixed-width arm, built at w = {FIXED_W} only, got w = {w}. Its width "
                         f"comes from dt_feasibility_arms._mla_widths (MLA_WIDTH_RULE[{arm!r}] = "
                         f"{WIDTH_RULE!r}); the KL-objective width is a different arm.")

def relabel(out, base, arm):
    if out["arm"] != base.ARM:
        raise AssertionError(f"{base.__name__} returned arm {out['arm']!r}, expected {base.ARM!r}")
    if out["w"] != FIXED_W:
        raise AssertionError(f"{base.__name__} returned w = {out['w']}, expected {FIXED_W}")
    out["arm"] = arm
    out["provenance"]["text"] = (
        f"{arm}: {base.ARM} built UNCHANGED ({base.VERSION}) at the FIXED rotary width w = {FIXED_W}, "
        f"d_c = {out['d_c']} (rule {WIDTH_RULE}: the MLA arms compared at one width, 16 on the GQA models, "
        f"the authors 2026-10-05). Any width wording below is the base module's own. -- " + out["provenance"]["text"])
    return out
