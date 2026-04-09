import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    """Pattern matching for batch normalization operation"""
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    """Extract arguments for batch normalization optimization"""
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_kernel(
    x_ptr,                    # Input tensor pointer
    running_mean_ptr,         # Running mean pointer
    running_var_ptr,          # Running variance pointer
    weight_ptr,               # Weight pointer (gamma)
    bias_ptr,                 # Bias pointer (beta)
    out_ptr,                  # Output tensor pointer
    N,                        # Batch size
    C,                        # Number of channels
    H,                        # Height
    W,                        # Width
    eps: tl.constexpr,        # Epsilon for numerical stability
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channel dimension
    BLOCK_SIZE_HW: tl.constexpr  # Block size for spatial dimensions
):
    # Calculate program IDs
    c_idx = tl.program_id(0)  # Channel program
    hw_idx = tl.program_id(1)  # Spatial program
    
    # Calculate offsets
    c_offset = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    hw_offset = hw_idx * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    h_offset = hw_offset // W
    w_offset = hw_offset % W
    
    # Create masks
    c_mask = c_offset < C
    hw_mask = hw_offset < (H * W)
    
    # Load running stats and parameters
    running_mean = tl.load(running_mean_ptr + c_offset, mask=c_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_offset, mask=c_mask, other=1.0)
    weight = tl.load(weight_ptr + c_offset, mask=c_mask, other=1.0)
    bias = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0)
    
    # Load input data spatially
    x_ptrs = x_ptr + (c_offset[:, None] * H * W + h_offset[None, :] * W + w_offset[None, :])
    x_data = tl.load(x_ptrs, mask=c_mask[:, None] & hw_mask[None, :], other=0.0)
    
    # Apply batch normalization
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    normalized = (x_data - running_mean[:, None]) * inv_std[:, None] * weight[:, None] + bias[:, None]
    
    # Store output
    out_ptrs = out_ptr + (c_offset[:, None] * H * W + h_offset[None, :] * W + w_offset[None, :])
    tl.store(out_ptrs, normalized, mask=c_mask[:, None] & hw_mask[None, :])

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    """Optimized batch normalization using Triton kernel"""
    # Get tensor dimensions
    N, C, H, W = x.shape
    
    # Choose optimal block sizes (must be powers of 2) - conservative approach
    BLOCK_SIZE_C = 1
    temp_c = min(64, C)
    while BLOCK_SIZE_C * 2 <= temp_c:
        BLOCK_SIZE_C *= 2
    
    BLOCK_SIZE_HW = 1024  # Fixed power of 2 for spatial dimensions
    
    # Calculate grid size
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid = (grid_c, grid_hw)
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    batch_norm_kernel[grid](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        eps=0.001,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    
    return output

def replacement_func():
    """Return the optimized batch normalization function"""
    return optimized_batch_norm