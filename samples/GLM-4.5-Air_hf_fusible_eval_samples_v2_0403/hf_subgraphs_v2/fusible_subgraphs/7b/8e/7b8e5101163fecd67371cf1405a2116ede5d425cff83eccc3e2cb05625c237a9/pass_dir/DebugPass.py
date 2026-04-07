import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern: Simple element-wise addition"""
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Addition kernel for bias addition"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the feature map
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load bias value (scalar)
    bias = tl.load(bias_ptr + 0)
    
    # Calculate addition
    out = x + bias
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def full_computation_kernel(
    bias_ptr,
    scale_ptr,
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for ReLU * scale + bias"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load bias and scale (they're scalars)
    bias = tl.load(bias_ptr + 0)
    scale = tl.load(scale_ptr + 0)
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: ReLU * scale + bias
    relu_x = tl.maximum(x, 0.0)
    out = relu_x * scale + bias
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@triton.jit
def apply_padding_kernel(
    input_ptr,
    output_ptr,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    """Apply padding (0,1,0,1) using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate output coordinates
    total_output_elements = batch * channels * (height + 1) * (width + 1)
    mask = offsets < total_output_elements
    
    # Convert to coordinates
    out_coords = offsets
    batch_idx = out_coords // (channels * (height + 1) * (width + 1))
    remaining = out_coords % (channels * (height + 1) * (width + 1))
    channel_idx = remaining // ((height + 1) * (width + 1))
    remaining = remaining % ((height + 1) * (width + 1))
    h_idx = remaining // (width + 1)
    w_idx = remaining % (width + 1)
    
    # Calculate input coordinates (skip padding area)
    input_h = h_idx
    input_w = w_idx
    input_coords = batch_idx * (channels * height * width) + channel_idx * (height * width) + input_h * width + input_w
    
    # Load input with bounds checking, pad with 0.0
    input_vals = tl.load(input_ptr + input_coords, mask=(input_coords < batch * channels * height * width), other=0.0)
    
    # Store output
    tl.store(output_ptr + out_coords, input_vals, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    """Optimized addition using Triton"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    add_kernel[(num_programs,)](
        x_ptr=x,
        bias_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_add