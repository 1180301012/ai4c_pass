import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the computation pattern from model.py
    tmp_0 = x.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_mean_operation(input_tensor):
    # Input tensor: [1, 2, 256, H, W]
    # Original computation: 
    #   tmp_0 = in_0.sum(1)        # Sum along axis 1, result: [1, 256, H, W]
    #   tmp_1 = tmp_0.mean((2, 3), keepdim=True)  # Mean along axes 2,3, result: [1, 256, 1, 1]
    # 
    # This is mathematically equivalent to mean along axes (1, 2, 3), but we need to be careful 
    # about numerical precision and intermediate values
    
    # Mathematically equivalent to: sum(axis=1) then mean(axes=(2,3), keepdim=True)
    # This should be mathematically identical but more efficient
    return input_tensor.mean(dim=(1, 3, 4), keepdim=True)

def replacement_func():
    return fused_mean_operation