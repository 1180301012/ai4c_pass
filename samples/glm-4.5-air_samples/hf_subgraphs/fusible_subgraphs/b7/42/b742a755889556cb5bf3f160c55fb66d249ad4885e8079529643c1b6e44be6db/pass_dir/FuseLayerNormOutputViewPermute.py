import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the sequence: permute -> view after layer norm
    # x has shape [1, 576, 384] from layer norm output
    x_permute = x.permute(0, 2, 1)  # [1, 384, 576]
    x_view = x_permute.view(1, 384, 24, 24)  # [1, 384, 24, 24]
    return x_view

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_layer_norm_output_transform(x):
    # Direct reshape from [1, 576, 384] to [1, 384, 24, 24]
    # This is more efficient than separate permute + view
    return x.reshape(1, 384, 24, 24)

def replacement_func():
    return fused_layer_norm_output_transform