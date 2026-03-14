import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: Add 2 tensors (with 0) and compute mean over spatial dimensions
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)
    """
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add2_mean_kernel(
    in0_ptr, in1_ptr, sum_ptr, mean_ptr,
    B, C, H, W,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Add 2 tensors elementwise and compute spatial mean
    Input shape: [B, C, H, W]
    Output: sum [B, C, H, W], mean [B, C, 1, 1]
    """
    # Each program handles one (batch, channel) slice
    bc_idx = tl.program_id(0)
    
    # Compute spatial sum for this (batch, channel)
    spatial_sum = 0.0
    
    # Process spatial dimensions in blocks
    for block_start in range(0, HW, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        
        # Base pointer for this (batch, channel, spatial_location)
        base_idx = bc_idx * HW + offsets
        
        # Load from both inputs
        in0 = tl.load(in0_ptr + base_idx, mask=mask, other=0.0)
        in1 = tl.load(in1_ptr + base_idx, mask=mask, other=0.0)
        
        # Compute sum
        result = in0 + in1
        
        # Store sum result
        tl.store(sum_ptr + base_idx, result, mask=mask)
        
        # Accumulate for mean
        spatial_sum += tl.sum(result)
    
    # Store mean
    mean_val = spatial_sum / HW
    tl.store(mean_ptr + bc_idx, mean_val)


@torch.fx.wrap
def fused_add2_mean(in_0, in_1):
    """Wrapper for fused add2 + mean kernel"""
    B, C, H, W = in_0.shape
    HW = H * W
    
    # Output tensors
    sum_out = torch.empty_like(in_0)
    mean_out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with one program per (batch, channel)
    num_programs = B * C
    BLOCK_SIZE = 1024
    
    fused_add2_mean_kernel[(num_programs,)](
        in_0, in_1, sum_out, mean_out,
        B, C, H, W,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (sum_out, mean_out)


def replacement_func():
    return fused_add2_mean