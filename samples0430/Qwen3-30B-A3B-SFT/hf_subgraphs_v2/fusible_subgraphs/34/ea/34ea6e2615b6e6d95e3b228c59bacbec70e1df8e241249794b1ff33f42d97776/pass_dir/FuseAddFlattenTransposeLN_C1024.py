import torch
from pass_dir.ln_shared_kernel import launch_fused_add_ln


# ── Pattern: fuse add + flatten + transpose + LN + two transpose(0,1) ──────────
def pattern(conv_out, x, weight, bias):
    add_out   = conv_out + x
    flat_out  = add_out.flatten(2)
    trans_out = flat_out.transpose(1, 2)
    norm_out  = torch.nn.functional.layer_norm(trans_out, (1024,), weight, bias, 1e-05)
    t1 = norm_out.transpose(0, 1)
    t2 = norm_out.transpose(0, 1)
    return (trans_out, t2, t1)


def replacement_args(conv_out, x, weight, bias):
    return (x, conv_out, weight, bias)


# ── Kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def _fused_1024(x, conv_out, weight, bias):
    trans, y = launch_fused_add_ln(x, conv_out, weight, bias)
    return (trans, y, y)


def replacement_func():
    return _fused_1024