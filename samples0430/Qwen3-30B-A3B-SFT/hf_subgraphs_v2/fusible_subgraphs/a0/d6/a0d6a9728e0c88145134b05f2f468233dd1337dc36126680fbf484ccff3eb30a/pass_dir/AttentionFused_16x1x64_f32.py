import torch
from pass_dir.shared_attn_kernel import attn_fused_dispatch


# ── D=64: 2-arg, starts from dropout_out (float32) ───────────────────────────

def pattern(dropout_out, value):
    bmm_1 = torch.bmm(dropout_out, value)
    tmp_4   = bmm_1.view(1, 16, 1, 64)
    tmp_5   = tmp_4.transpose(1, 2)
    tmp_6   = tmp_5.reshape(1, 1, 1024)
    return tmp_6


def replacement_args(dropout_out, value):
    return (dropout_out, value, "route_weighted_64")


def replacement_func():
    return attn_fused_dispatch