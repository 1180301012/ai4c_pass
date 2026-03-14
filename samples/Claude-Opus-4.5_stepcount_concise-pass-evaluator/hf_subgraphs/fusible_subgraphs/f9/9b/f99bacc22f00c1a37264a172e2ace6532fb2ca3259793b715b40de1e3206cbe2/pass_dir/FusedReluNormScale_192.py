import torch
import triton
import triton.language as tl

# Scale factor for HW=192 (16x12)
SCALE_FACTOR = 0.07216878364870322

def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_relu_norm_scale_kernel_192(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    num_pairs,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    
    if pid >= num_pairs:
        return
    
    # Compute offset for this (batch, channel) pair
    base_offset = pid * HW
    
    # Load in_0 (scalar parameter)
    g = tl.load(in_0_ptr)
    
    # Load input data and apply ReLU
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    x = tl.load(in_1_ptr + base_offset + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)  # ReLU
    
    # Compute L2 norm (sum of squares then sqrt)
    sum_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(sum_sq)
    
    # Scale by constant and clamp
    scaled_norm = norm * 0.07216878364870322
    clamped_norm = tl.maximum(scaled_norm, 1e-05)
    
    # Normalize and scale by g
    result = (x / clamped_norm) * g
    
    # Store result
    tl.store(out_ptr + base_offset + offsets, result, mask=mask)

@torch.fx.wrap
def fused_relu_norm_scale_192(in_0, in_1):
    B, C, H, W = in_1.shape
    HW = H * W
    
    # Output shape is [B, C, HW]
    out = torch.empty((B, C, HW), device=in_1.device, dtype=in_1.dtype)
    
    # Determine block size (must be power of 2 >= HW)
    BLOCK_SIZE = triton.next_power_of_2(HW)
    
    # Grid: one block per (batch, channel) pair
    num_pairs = B * C
    grid = (num_pairs,)
    
    fused_relu_norm_scale_kernel_192[grid](
        in_0,
        in_1,
        out,
        num_pairs,
        HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out,)

def replacement_func():
    return fused_relu_norm_scale_192