import torch
import triton
import triton.language as tl

# Fixed block size using constexpr
BLOCK_SIZE = tl.constexpr(1024)

@triton.jit
def fused_sigmoid_mul_kernel(
    # Input pointers
    sigmoid_ptr, in_2_ptr,
    # Output pointer
    out_ptr,
    # Shape info
    B, C, H, W,
    # Strides for sigmoid [B, C, 1, 1]
    sigmoid_stride_b, sigmoid_stride_c,
    # Strides for in_2 [B, C, H, W]
    in_2_stride_b, in_2_stride_c, in_2_stride_h, in_2_stride_w,
    # Number of elements
    N_elements: tl.constexpr,
):
    """
    Fused sigmoid multiplication kernel.
    
    sigmoid has shape [B, C, 1, 1] (per-channel scalar)
    in_2 has shape [B, C, H, W]
    output has shape [B, C, H, W]
    
    Grid: (num_programs,)
    Each program handles BLOCK_SIZE elements
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements
    
    # Calculate b, c, h, w from flat offsets
    b = offsets // (C * H * W)
    c = (offsets // (H * W)) % C
    h = (offsets // W) % H
    w = offsets % W
    
    # Compute offsets
    sigmoid_offset = b * sigmoid_stride_b + c * sigmoid_stride_c
    in_2_offset = b * in_2_stride_b + c * in_2_stride_c + h * in_2_stride_h + w * in_2_stride_w
    out_offset = in_2_offset
    
    # Load values
    sigmoid_val = tl.load(sigmoid_ptr + sigmoid_offset)
    in_2_val = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # Multiply
    out_val = sigmoid_val * in_2_val
    
    # Store
    tl.store(out_ptr + out_offset, out_val, mask=mask)


def pattern(in_2, in_1):
    """
    Match the sigmoid * tensor multiplication pattern.
    This is a simpler pattern that matches the mul operation with broadcasting.
    """
    return in_2 * in_1


def replacement_args(in_2, in_1):
    """Extract arguments needed for the replacement kernel."""
    return (in_2, in_1)


@torch.fx.wrap
def fused_sigmoid_mul_wrapper(in_2, in_1):
    """
    Wrapper function that fuses the sigmoid multiplication.
    Uses Triton for the element-wise multiplication with broadcasting.
    """
    B, C, H, W = in_2.shape
    N_elements = B * C * H * W
    
    # Allocate output tensor
    out = torch.empty_like(in_2)
    
    # Calculate strides for sigmoid [B, C, 1, 1]
    sigmoid_stride_b = C
    sigmoid_stride_c = 1
    
    # Calculate strides for in_2 [B, C, H, W]
    in_2_stride_b = C * H * W
    in_2_stride_c = H * W
    in_2_stride_h = W
    in_2_stride_w = 1
    
    # Configure grid
    BLOCK = 1024
    num_programs = (N_elements + BLOCK - 1) // BLOCK
    
    # Launch Triton kernel for multiplication
    grid = (num_programs,)
    
    fused_sigmoid_mul_kernel[grid](
        in_1, in_2, out,
        B, C, H, W,
        sigmoid_stride_b, sigmoid_stride_c,
        in_2_stride_b, in_2_stride_c, in_2_stride_h, in_2_stride_w,
        N_elements
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_sigmoid_mul_wrapper