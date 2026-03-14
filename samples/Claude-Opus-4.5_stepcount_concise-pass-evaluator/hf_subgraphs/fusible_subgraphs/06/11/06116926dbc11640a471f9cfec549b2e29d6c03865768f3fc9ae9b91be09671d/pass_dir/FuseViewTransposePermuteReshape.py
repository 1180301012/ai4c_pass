import torch

# Pattern matching function - specifically for face-parsing_start46_end50_12 with batch=32
def pattern(in_0, in_1):
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    return tmp_1, tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Use scripted function to minimize Python overhead
@torch.jit.script
def _transform_impl(in_0: torch.Tensor, in_1: torch.Tensor):
    out1 = in_1.view(32, -1, 1, 64).transpose(1, 2)
    out2 = in_0.permute(0, 2, 1).reshape(32, 64, 128, 128)
    return out1, out2

@torch.fx.wrap
def combined_transform(in_0, in_1):
    return _transform_impl(in_0, in_1)

def fused_replacement(in_0, in_1):
    result = combined_transform(in_0, in_1)
    return result[0], result[1]

def replacement_func():
    return fused_replacement