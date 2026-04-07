import torch
import triton
import triton.language as tl

# Pattern matching function for transpose
def pattern(x):
    """Match transpose operation swapping last two dimensions"""
    result = x.transpose(-2, -1)
    return result

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized transpose kernel with better memory access patterns
@triton.jit
def efficient_transpose_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Highly efficient transpose kernel for swapping last two dimensions"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.all(~mask):
        return
    
    # Efficient coordinate calculation for 4D tensor transpose [B,C,H,W] -> [B,C,W,H]
    idx = offsets
    
    # Convert linear index to output coordinates [B, C, W, H]
    output_b = idx // (C * W * H)
    remainder = idx % (C * W * H)
    output_c = remainder // (W * H)
    remainder = remainder % (W * H)
    output_w = remainder // H
    output_h = remainder % H
    
    # Calculate original index [B, C, H, W] that maps to output
    original_idx = output_b * (C * H * W) + output_c * (H * W) + output_h * W + output_w
    
    # Load from original position and store to output position
    x = tl.load(x_ptr + original_idx, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

# Alternative transpose using load/store coalescing optimization
@triton.jit
def coalesced_transpose_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Memory-coalesced transpose kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate coordinates assuming output layout [B, C, W, H]
    total_per_b = W * H  # Elements per batch item in transposed output
    offset_in_b = offsets % total_per_b
    batch_idx = offsets // total_per_b
    
    if batch_idx >= B * C:
        return  # Out of bounds
    
    batch_item = batch_idx % C
    batch_group = batch_idx // C
    
    # Convert within-batch offset [W, H] to original [H, W]
    w = offset_in_b // H
    h = offset_in_b % H
    
    # Calculate original index
    original_idx = batch_group * (C * H * W) + batch_item * (H * W) + h * W + w
    
    # Coalesced memory access
    x = tl.load(x_ptr + original_idx, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def triton_efficient_transpose(x):
    """Efficient transpose using Triton kernel"""
    B, C, H, W = x.shape
    n_elements = x.numel()
    
    # Optimize block size for transpose operation
    BLOCK_SIZE = 1024
    
    # Calculate number of programs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with transposed shape [B, C, W, H]
    out = torch.empty((B, C, W, H), dtype=x.dtype, device=x.device)
    
    # Launch coalesced transpose kernel
    coalesced_transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        B=B, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_efficient_transpose