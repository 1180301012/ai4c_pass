import torch

# Pattern for face-parsing_start422_end426_52 with batch=32
def pattern(in_0, in_1):
    tmp_0 = in_1.view(32, -1, 5, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 320, 32, 32)
    return tmp_1, tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def view_transpose_native_2(in_1):
    return in_1.view(32, -1, 5, 64).transpose(1, 2)

@torch.fx.wrap
def permute_reshape_native_2(in_0):
    return in_0.permute(0, 2, 1).reshape(32, 320, 32, 32)

def fused_replacement_2(in_0, in_1):
    out1 = view_transpose_native_2(in_1)
    out2 = permute_reshape_native_2(in_0)
    return out1, out2

def replacement_func():
    return fused_replacement_2