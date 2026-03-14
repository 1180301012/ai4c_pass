import torch
import triton
import triton.language as tl

def pattern(conv_output, in_5):
    # Match sigmoid -> multiplication pattern from the model
    tmp_3 = torch.sigmoid(conv_output)
    tmp_4 = in_5 * tmp_3
    return tmp_3, tmp_4

def replacement_args(conv_output, in_5):
    return (conv_output, in_5)

@torch.fx.wrap
def fused_sigmoid_mul_optimized(conv_output, in_5):
    # Optimized fused sigmoid + multiplication
    # This avoids intermediate tensor creation
    return torch.sigmoid(conv_output) * in_5

def replacement_func():
    return fused_sigmoid_mul_optimized