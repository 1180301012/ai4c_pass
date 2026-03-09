import torch
from torch import device
import triton
import triton.language as tl
import math

def pattern(in_0):
    """Pattern that matches arange(0, 64) with bool conversion"""
    tmp_1 = torch.arange(0, 64)
    tmp_2 = in_0.to(dtype=torch.bool)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    """Extract input tensor for the replacement"""
    return (in_0,)

@triton.jit
def fused_kernel_64(out_arange_ptr, out_bool_ptr, in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused kernel for arange(0, 64) and bool conversion"""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Process arange output (fixed size 64)
    arange_block_start = pid * BLOCK_SIZE
    arange_offsets = arange_block_start + tl.arange(0, BLOCK_SIZE)
    arange_mask = arange_offsets < 64
    tl.store(out_arange_ptr + arange_offsets, arange_offsets, mask=arange_mask)
    
    # Process bool conversion
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    bool_result = x != 0
    tl.store(out_bool_ptr + offsets, bool_result, mask=mask)

@torch.fx.wrap
def optimized_fusion_64(in_0):
    """Optimized function for pattern with size 64"""
    device = in_0.device
    arange_out = torch.empty(64, dtype=torch.int64, device=device)
    bool_out = torch.empty(in_0.numel(), dtype=torch.bool, device=device)
    
    BLOCK_SIZE = 1024
    n_elements = in_0.numel()
    
    # Grid for size=64 arange + input processing
    grid_size = max(math.ceil(64 / BLOCK_SIZE), math.ceil(n_elements / BLOCK_SIZE))
    
    fused_kernel_64[(grid_size,)](
        out_arange_ptr=arange_out,
        out_bool_ptr=bool_out,
        in_ptr=in_0,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return arange_out, bool_out

def replacement_func():
    """Returns the optimized function"""
    return optimized_fusion_64