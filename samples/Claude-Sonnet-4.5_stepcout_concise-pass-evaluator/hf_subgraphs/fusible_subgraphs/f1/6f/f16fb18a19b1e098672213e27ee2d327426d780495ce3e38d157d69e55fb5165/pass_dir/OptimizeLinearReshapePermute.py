import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Simple pattern: just match the full computation without trying to handle unbind specially
    """
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_5 = tmp_3[0, :, :, :, :]
    tmp_6 = tmp_3[1, :, :, :, :]
    tmp_7 = tmp_3[2, :, :, :, :]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def fused_qkv_baseline(weight, input):
    """
    Just use torch operations for now to verify pattern matching works
    """
    tmp_1 = torch.nn.functional.linear(input, weight, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_5 = tmp_3[0, :, :, :, :]
    tmp_6 = tmp_3[1, :, :, :, :]
    tmp_7 = tmp_3[2, :, :, :, :]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)

def replacement_func():
    return fused_qkv_baseline