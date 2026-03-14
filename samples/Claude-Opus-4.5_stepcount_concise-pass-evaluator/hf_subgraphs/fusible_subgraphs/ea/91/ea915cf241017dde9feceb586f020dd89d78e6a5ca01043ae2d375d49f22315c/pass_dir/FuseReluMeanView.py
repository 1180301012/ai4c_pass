import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    tmp_1 = tmp_0.mean((2, 3))
    tmp_2 = tmp_1.view(1, 1, -1)
    return tmp_2

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def mean_kernel_fast(
    in_ptr,
    out_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * spatial_size
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size
    
    x = tl.load(in_ptr + base_offset + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    mean_val = sum_val / spatial_size
    tl.store(out_ptr + pid, mean_val)

@torch.fx.wrap  
def fused_mean_view(tmp_0):
    # Directly compute mean and reshape - minimize ops
    out_mean = tmp_0.mean((2, 3))
    return out_mean.view(1, 1, -1)

def replacement_func():
    return fused_mean_view