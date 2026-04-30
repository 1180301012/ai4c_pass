import torch
from pass_dir.shared_fused_gelu_transpose_add_layernorm import fused_dispatch


def pattern(x, residual):
    tmp_5 = torch.nn.functional.gelu(x)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = residual + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False)
    return tmp_8


def replacement_args(x, residual):
    return (x, residual, "epilogue_p01")


def replacement_func():
    return fused_dispatch