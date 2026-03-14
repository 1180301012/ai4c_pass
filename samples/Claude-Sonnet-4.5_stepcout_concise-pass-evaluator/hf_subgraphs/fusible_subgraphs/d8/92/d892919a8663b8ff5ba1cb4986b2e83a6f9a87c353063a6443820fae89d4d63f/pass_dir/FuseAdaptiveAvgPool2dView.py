import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching adaptive_avg_pool2d + view"""
    tmp_8 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_9 = tmp_8.view(x.shape[0], x.shape[1])
    return tmp_9

def replacement_args(x):
    return (x,)

@triton.jit
def fused_adaptive_avg_pool2d_view_kernel(
    in_ptr,
    out_ptr,
    N, C, H, W,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for adaptive_avg_pool2d + view"""
    pid = tl.program_id(0)
    
    # Each program handles one (N, C) output element
    n_idx = pid // C
    c_idx = pid % C
    
    if n_idx >= N or c_idx >= C:
        return
    
    # Calculate the base offset for this (N, C) slice
    base_offset = n_idx * C * H * W + c_idx * H * W
    
    # Accumulate sum over H*W spatial dimensions
    accumulator = 0.0
    for hw_idx in range(0, HW, BLOCK_SIZE):
        offsets = hw_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        
        load_offsets = base_offset + offsets
        values = tl.load(in_ptr + load_offsets, mask=mask, other=0.0)
        accumulator += tl.sum(values)
    
    # Average over spatial dimensions
    avg = accumulator / HW
    
    # Store result
    out_offset = n_idx * C + c_idx
    tl.store(out_ptr + out_offset, avg)

@torch.fx.wrap
def fused_adaptive_avg_pool2d_view(x):
    """Wrapper for fused adaptive_avg_pool2d + view kernel"""
    N, C, H, W = x.shape
    HW = H * W
    
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    
    grid = (N * C,)
    BLOCK_SIZE = 256
    
    fused_adaptive_avg_pool2d_view_kernel[grid](
        x,
        out,
        N, C, H, W,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_adaptive_avg_pool2d_view