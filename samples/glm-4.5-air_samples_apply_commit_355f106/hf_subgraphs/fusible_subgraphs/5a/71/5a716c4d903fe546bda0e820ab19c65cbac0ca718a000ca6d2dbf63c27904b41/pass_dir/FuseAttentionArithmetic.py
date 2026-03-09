import torch

def pattern(in_0, in_1, in_2, in_3):
    # Fuse: ((in_0 + in_3 + in_2) / 8.0) + in_1
    # This corresponds to the arithmetic sequence:
    # tmp_0 = in_0 + in_3
    # tmp_1 = tmp_0 + in_2
    # tmp_2 = tmp_1 / 8.0
    # tmp_3 = tmp_2 + in_1
    
    # Pattern must return what the original would return before softmax
    # So we return tmp_3 which feeds into softmax
    tmp_3 = ((in_0 + in_3 + in_2) / 8.0) + in_1
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@torch.fx.wrap
def fused_attention_arithmetic(in_0, in_1, in_2, in_3):
    # Use fused operations with PyTorch's built-in optimizations
    # This is simpler and avoids Triton kernel overhead
    out = (in_0 + in_2 + in_3).div(8.0, rounding_mode=None).add(in_1)
    return out

def replacement_func():
    return fused_attention_arithmetic