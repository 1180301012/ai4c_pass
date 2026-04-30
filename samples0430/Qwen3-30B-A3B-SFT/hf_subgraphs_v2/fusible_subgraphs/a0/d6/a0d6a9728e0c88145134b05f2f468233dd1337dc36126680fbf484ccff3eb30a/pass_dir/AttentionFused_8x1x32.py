import torch
from pass_dir.shared_attn_kernel import attn_fused_dispatch


# ── D=32: 2-arg, starts from dropout_out ─────────────────────────────────────
# dropout_out = model's tmp_2 (softmax→dropout output; p=0.0 so == softmax output)
# dropout_out shape [1,8,1,32], value [8,1,32] → output [1,1,256]

def pattern(dropout_out, value):
    bmm_1 = torch.bmm(dropout_out, value)
    tmp_4   = bmm_1.view(1, 8, 1, 32)
    tmp_5   = tmp_4.transpose(1, 2)
    tmp_6   = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(dropout_out, value):
    return (dropout_out, value, "route_weighted_32")


def replacement_func():
    return attn_fused_dispatch