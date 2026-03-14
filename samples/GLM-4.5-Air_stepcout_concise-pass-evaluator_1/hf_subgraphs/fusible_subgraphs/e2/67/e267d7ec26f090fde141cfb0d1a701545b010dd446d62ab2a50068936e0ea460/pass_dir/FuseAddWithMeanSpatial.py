import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern to match: Three-input addition followed by spatial mean computation
    """
    # Exact match: tmp_0 = in_1 + in_2
    tmp_0 = in_1 + in_2
    # Exact match: tmp_0 += in_0 (in-place addition)
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_add_mean_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    out_ptr, mean_ptr,
    B, C, H, W,
    num_inputs: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one spatial position across all batch channels
    pid = tl.program_id(0)
    
    # Determine spatial position
    spatial_idx = pid
    h = spatial_idx // W
    w = spatial_idx % W
    
    # Ensure we're within bounds
    if h >= H or w >= W:
        return
    
    # Load all inputs for this spatial position across all batch channels
    offsets = tl.arange(0, B * C)
    base_offset0 = h * W * C + w * C
    base_offset1 = h * W * C + w * C
    base_offset2 = h * W * C + w * C
    
    mask = offsets < B * C
    
    # Load inputs
    if num_inputs == 3:
        in0_vals = tl.load(in0_ptr + base_offset0 + offsets, mask=mask, other=0.0)
        in1_vals = tl.load(in1_ptr + base_offset1 + offsets, mask=mask, other=0.0)
        in2_vals = tl.load(in2_ptr + base_offset2 + offsets, mask=mask, other=0.0)
        # Add all three inputs
        sum_vals = in0_vals + in1_vals + in2_vals
    else:
        in0_vals = tl.load(in0_ptr + base_offset0 + offsets, mask=mask, other=0.0)
        in1_vals = tl.load(in1_ptr + base_offset1 + offsets, mask=mask, other=0.0)
        # Add two inputs (in_1 is added first with 0, then in_0)
        sum_vals = in0_vals + in1_vals
    
    # Store output
    tl.store(out_ptr + base_offset0 + offsets, sum_vals, mask=mask)
    
    # Compute mean by reading back the sum
    if mask[0]:  # Only compute mean for valid positions
        # Sum over all batch and channel for this spatial position
        spatial_sum = tl.sum(sum_vals)
        mean_val = spatial_sum / (B * C)
        tl.store(mean_ptr + pid, mean_val)

@torch.fx.wrap
def fused_add_mean_forward(in_0, in_1, in_2):
    # Get tensor shapes
    B, C, H, W = in_0.shape
    
    # Check input shapes match
    if in_1.shape != (B, C, H, W):
        raise ValueError(f"in_1 shape {in_1.shape} doesn't match in_0 shape {in_0.shape}")
    if in_2.shape != (B, C, H, W):
        raise ValueError(f"in_2 shape {in_2.shape} doesn't match in_0 shape {in_0.shape}")
    
    # Create output tensors
    out = torch.empty_like(in_0)
    mean = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid size (total spatial positions)
    spatial_positions = H * W
    BLOCK_SIZE_M = 1024  # Number of spatial positions per program
    
    # Launch kernel
    grid = (triton.cdiv(spatial_positions, BLOCK_SIZE_M),)
    
    fused_add_mean_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out_ptr=out,
        mean_ptr=mean,
        B=B, C=C, H=H, W=W,
        num_inputs=3,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=1024,
    )
    
    return out, mean

def replacement_func():
    return fused_add_mean_forward