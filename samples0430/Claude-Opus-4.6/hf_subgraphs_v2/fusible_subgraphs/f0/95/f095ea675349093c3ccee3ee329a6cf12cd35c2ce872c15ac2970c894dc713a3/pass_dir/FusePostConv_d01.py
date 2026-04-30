import torch
import triton
import triton.language as tl


def pattern(conv_result, residual):
    sliced = conv_result[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    gelu_out = torch.nn.functional.gelu(sliced)
    transposed = gelu_out.transpose(1, 2)
    added = residual + transposed
    return added


def replacement_args(conv_result, residual):
    return (conv_result, residual)


from pass_dir.FusePostConv_d005 import fused_gelu_transpose_add


def replacement_func():
    return fused_gelu_transpose_add