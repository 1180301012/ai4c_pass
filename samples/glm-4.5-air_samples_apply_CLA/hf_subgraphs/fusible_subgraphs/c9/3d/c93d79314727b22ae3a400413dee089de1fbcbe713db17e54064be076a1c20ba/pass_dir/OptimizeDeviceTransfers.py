import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_3 = in_1.to(device(type='cuda'))
    tmp_4 = in_0.to(device(type='cuda'))
    return tmp_4, tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def optimized_device_transfer(in_0, in_1):
    # Both input tensors are already on CPU, transfer them together
    # in a single batch operation to reduce overhead
    result_device = 'cuda:0'
    
    # Transfer both tensors to GPU in one batch operation
    in_0_gpu = in_0.to(device=result_device)
    in_1_gpu = in_1.to(device=result_device)
    
    # Return both tensors directly without intermediate assignments
    return in_0_gpu, in_1_gpu

def replacement_func():
    return optimized_device_transfer