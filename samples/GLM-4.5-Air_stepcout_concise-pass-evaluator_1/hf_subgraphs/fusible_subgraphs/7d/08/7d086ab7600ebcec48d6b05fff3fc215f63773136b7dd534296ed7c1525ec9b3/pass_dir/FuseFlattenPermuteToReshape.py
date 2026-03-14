import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: flatten(2) followed by permute(0, 2, 1)
# Input: [B, C, H, W]
# flatten(2) -> [B, C, H*W]
# permute(0, 2, 1) -> [B, H*W, C]
def pattern(x):
    tmp = x.flatten(2)
    out = tmp.permute(0, 2, 1)
    return out

# Argument extraction function
def replacement_args(x):
    return (x,)


# Optimized Triton kernel that uses contiguous memory layout for better GPU performance
# This kernel performs a fused flatten + permute with optimized memory access pattern
@triton.jit
def fused_reshape_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input - memory coalesced access from [B, C, H, W]
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute output indices for [B, H*W, C] layout
    # Original input index: [b, c, h, w] -> flat = b*C*H*W + c*H*W + h*W + w
    # Target output index: [b, hw, c] where hw = h*W + w
    # -> flat = b*H*W*C + hw*C + c
    
    b = offsets // (C * H * W)
    c_h_w = offsets % (C * H * W)
    c = c_h_w // (H * W)
    hw = c_h_w % (H * W)
    
    # Compute output flat index
    out_flat = b * (H * W * C) + hw * C + c
    
    # Store to output with optimized write pattern
    tl.store(output_ptr + out_flat, x, mask=mask)


@torch.fx.wrap
def triton_optimized_reshape(x):
    """
    Optimized reshape from [B, C, H, W] to [B, H*W, C].
    Uses single kernel launch with optimized block size for better GPU utilization.
    """
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    
    # Allocate output tensor
    out = torch.empty((B, H * W, C), dtype=x.dtype, device=x.device)
    
    # Use larger block size for better GPU occupancy with these tensor sizes
    # For small tensors (like [1, 256, 8, 8] = 16384 elements), use smaller blocks
    # For larger tensors, use larger blocks
    if n_elements < 65536:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 4096
    
    # Calculate number of programs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized kernel
    fused_reshape_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return triton_optimized_reshape


# ============================================
# THE PASS CONFIGURATION FILE (sorted_output_pass_rule_names.json) SHOULD CONTAIN:
# ["FuseFlattenPermuteToReshape"]
# ============================================