import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """Exact pattern matching the original computation without redundant checks and assertions"""
    # Indexing operation
    tmp_1 = in_0[slice(None, None, None), in_2]
    
    # Get size of dimension 1 (only once, eliminating redundant checks and assertions)
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    
    # Concatenate tensors
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    
    # Direct computation of ones tensor size instead of torch.sym_sum
    # The constant 128 will be parameterized in the kernel wrapper
    tmp_10 = tmp_2 + 128  # Replaces torch.sym_sum([128, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=torch.device('cuda'))
    
    return (tmp_9, tmp_11)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_ones_kernel(
    ones_ptr,
    ones_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for creating ones tensor"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ones_size
    tl.store(ones_ptr + offsets, 1.0, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, constant):
    """Optimized kernel wrapper that eliminates redundant operations and uses optimized ones creation"""
    # Perform the indexing operation (necessary for correctness)
    tmp_1 = in_0[slice(None, None, None), in_2]
    
    # Get size (eliminating redundant checks and assertions)
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    
    # Concatenate tensors using standard torch.cat (already efficient)
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    
    # Optimized ones tensor creation using Triton
    total_size = tmp_2.item() + constant
    tmp_11 = torch.empty((total_size,), dtype=torch.float32, device='cuda')
    
    # Launch optimized ones kernel
    BLOCK_SIZE = 1024
    grid_size = (math.ceil(total_size / BLOCK_SIZE),)
    
    optimized_ones_kernel[grid_size](
        tmp_11,
        total_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_9, tmp_11

def replacement_func():
    """Return the optimized kernel wrapper"""
    return lambda in_0, in_1, in_2: kernel_wrapper(in_0, in_1, in_2, 128)