import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_4, in_3):
    # This matches the addition + dropout pattern
    # tmp_3 = in_4 + in_3
    tmp_3 = in_4 + in_3
    # tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    tmp_3 = None
    return tmp_4

# Argument extraction function
def replacement_args(in_4, in_3):
    return (in_4, in_3)

# Triton kernel for fused addition + dropout2d
@triton.jit
def fused_add_dropout_kernel(
    x1_ptr,           # First input tensor [B, C, H, W]
    x2_ptr,           # Second input tensor [B, C, H, W]
    output_ptr,       # Output tensor [B, C, H, W]
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # Number of elements per program
):
    # Calculate grid position
    pid = tl.program_id(0)  # Program ID
    num_programs = tl.num_programs(0)
    
    # Each program handles BLOCK_SIZE_N elements
    block_start = pid * BLOCK_SIZE_N
    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)
    total_elements = batch_size * channels * height * width
    
    # Mask to handle out-of-bounds elements
    mask = offsets < total_elements
    
    # Load input elements (using strides to handle 4D tensor)
    def load_4d_tensor(ptr, offset, batch_size, channels, height, width):
        # Calculate 4D coordinates from flattened offset
        flat_idx = offset
        b = flat_idx // (channels * height * width)
        flat_idx = flat_idx % (channels * height * width)
        c = flat_idx // (height * width)
        flat_idx = flat_idx % (height * width)
        h = flat_idx // width
        w = flat_idx % width
        
        # Ensure coordinates are within bounds
        b = min(b, batch_size - 1)
        c = min(c, channels - 1)
        h = min(h, height - 1)
        w = min(w, width - 1)
        
        # Return linear address and bounds check
        linear_addr = ptr + b * channels * height * width + c * height * width + h * width + w
        bounds_ok = (b < batch_size) & (c < channels) & (h < height) & (w < width)
        return linear_addr, bounds_ok
    
    x1_addr, x1_bounds_ok = load_4d_tensor(x1_ptr, offsets, batch_size, channels, height, width)
    x2_addr, x2_bounds_ok = load_4d_tensor(x2_ptr, offsets, batch_size, channels, height, width)
    
    # Combine masks
    element_mask = mask & x1_bounds_ok & x2_bounds_ok
    
    # Load input values with masking
    x1 = tl.load(x1_addr, mask=element_mask, other=0.0)
    x2 = tl.load(x2_addr, mask=element_mask, other=0.0)
    
    # Perform addition
    sum_val = x1 + x2
    
    # Apply dropout during training mode - since training=False in original,
    # we would normally skip dropout, but let's implement for generality.
    # For training=False (inference), dropout is a no-op
    if dropout_p > 0.0:
        # Generate random values - in real implementation we'd use proper RNG
        # For now, create a simple approximation for demonstration
        # Note: This is a simplified dropout implementation
        scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 1.0
        
        # Simple hash-based pseudo-random value for demonstration
        # In practice, use proper CUDA RNG or triton's random utilities
        random_vals = tl.sin(offsets * 12.9898 + 78.233) * 43758.5453
        random_vals = random_vals - tl.floor(random_vals)
        
        # Apply dropout mask
        keep_mask = random_vals > dropout_p
        
        # Apply dropout scaling
        out = sum_val * keep_mask * scale
    else:
        # No dropout needed
        out = sum_val
    
    # Store result
    if mask.any():
        tl.store(output_ptr + offsets, out, mask=element_mask)

# Kernel wrapper function
@torch.fx.wrap
def fused_add_dropout2d(x1, x2, dropout_p=0.1, training=False):
    # If training=False, dropout is a no-op, just do addition
    if not training:
        return x1 + x2
    
    batch_size, channels, height, width = x1.shape
    total_elements = batch_size * channels * height * width
    
    # Ensure output tensor has correct shape
    output = torch.empty_like(x1)
    
    # Launch Triton kernel
    BLOCK_SIZE_N = 1024  # Number of elements per program
    
    grid = (total_elements + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_add_dropout_kernel[grid](
        x1_ptr=x1,
        x2_ptr=x2,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        dropout_p=dropout_p,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_add_dropout2d