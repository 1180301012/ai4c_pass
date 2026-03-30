import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_dropout_kernel(
    x_ptr,
    out_relu_ptr,
    out_dropout_ptr,
    n_elements,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + 2D Dropout kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU (in-place semantics)
    relu_out = tl.where(x > 0, x, 0.0)
    
    # Apply Dropout - only during training, we're using deterministic evaluation
    # For evaluation mode, dropout is effectively a no-op, but we'll keep the structure
    # Use manual random number generation for deterministic behavior
    dropout_seed = tl.program_id(0)
    offsets_with_seed = offsets + dropout_seed * n_elements
    
    # Simple hash-based pseudo-random for dropout mask
    random_vals = tl.offsets_and_seed_to_mask(offsets_with_seed % (1 << 30), p, 0.5)
    dropout_mask = tl.where(random_vals > p, 1.0, 0.0)
    
    dropout_out = relu_out * dropout_mask
    
    # Store both outputs
    tl.store(out_relu_ptr + offsets, relu_out, mask=mask)
    tl.store(out_dropout_ptr + offsets, dropout_out, mask=mask)

@triton.jit
def fused_relu_dropout_2d_kernel(
    x_ptr,
    out_relu_ptr,
    out_dropout_ptr,
    batch_size,
    channels,
    height,
    width,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """2D-aware fused ReLU + Dropout kernel with proper spatial dropout semantics"""
    # Calculate total elements
    n_elements = batch_size * channels * height * width
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_out = tl.where(x > 0, x, 0.0)
    
    # For dropout2d, we need to dropout entire feature maps (channels)
    # Convert linear offset to 4D coordinates
    linear_idx = offsets
    batch_idx = linear_idx // (channels * height * width)
    remaining = linear_idx % (channels * height * width)
    channel_idx = remaining // (height * width)
    spatial_idx = remaining % (height * width)
    
    # Generate dropout masks per channel
    dropout_seed = (tl.program_id(0) * 17 + batch_idx * 7 + channel_idx * 3) & 0xFFFF
    dropout_val = (dropout_seed * 9301 + 49297) % 233280  # Simple PRNG
    dropout_mask = 1.0 if dropout_val / 233280.0 > p else 0.0
    
    dropout_out = relu_out * dropout_mask
    
    # Store both outputs
    tl.store(out_relu_ptr + offsets, relu_out, mask=mask)
    tl.store(out_dropout_ptr + offsets, dropout_out, mask=mask)

@triton.jit
def optimized_dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple optimized dropout kernel - just identity for eval mode"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    
    # Load input data and store it back (identity operation for eval mode)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_dropout(x, p=0.1):
    """Optimized dropout implementation - returns dropout_out only"""
    n_elements = x.numel()
    
    # Use optimal block size for maximum occupancy
    BLOCK_SIZE = 1024  # Good balance for most tensor sizes
    
    # Calculate grid size efficiently
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out_dropout = torch.empty_like(x)
    
    # Launch simplified kernel
    optimized_dropout_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out_dropout,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_dropout

def pattern(x):
    """Simple dropout pattern that matches"""
    return torch.nn.functional.dropout2d(x, 0.1, False, False)

def replacement_args(in_0):
    """Extract arguments for fused kernel"""
    return (in_0,)

def replacement_func():
    """Return optimized dropout function"""
    return optimized_dropout