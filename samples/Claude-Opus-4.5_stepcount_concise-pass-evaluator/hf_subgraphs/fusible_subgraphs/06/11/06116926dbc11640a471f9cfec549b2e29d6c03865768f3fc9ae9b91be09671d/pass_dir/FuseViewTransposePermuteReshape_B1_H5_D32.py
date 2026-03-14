import torch

# Pattern for nvidia_mit-b0_start222_end226_44 with batch=1
def pattern(in_0, in_1):
    tmp_0 = in_1.view(1, -1, 5, 32)
    tmp_1 = tmp_0.transpose(1, 2)
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(1, 160, 32, 32)
    return tmp_1, tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def view_transpose_native_4(in_1):
    return in_1.view(1, -1, 5, 32).transpose(1, 2)

@torch.fx.wrap
def permute_reshape_native_4(in_0):
    return in_0.permute(0, 2, 1).reshape(1, 160, 32, 32)

def fused_replacement_4(in_0, in_1):
    out1 = view_transpose_native_4(in_1)
    out2 = permute_reshape_native_4(in_0)
    return out1, out2

def replacement_func():
    return fused_replacement_4