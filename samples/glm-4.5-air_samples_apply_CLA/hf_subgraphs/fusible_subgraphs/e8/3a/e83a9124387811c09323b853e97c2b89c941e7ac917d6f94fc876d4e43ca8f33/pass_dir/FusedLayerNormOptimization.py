import torch
import triton
import triton.language as tl

# Pattern matching for LayerNorm computation: (in_3 + in_2) * in_1 + in_0
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.heuristics({"BLOCK_SIZE": lambda args: 1024})
@triton.jit
def fused_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    bias_size: tl.constexpr,
    weight_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors with optimized memory coalescing
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized broadcasting with shared memory access patterns
    if bias_size == 1:
        bias = tl.load(bias_ptr + 0)
    else:
        # Use modulo optimization with stride considerations
        bias_offsets = offsets % bias_size
        bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
    
    if weight_size == 1:
        weight = tl.load(weight_ptr + 0)
    else:
        weight_offsets = offsets % weight_size
        weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    
    # Fused computation with minimal intermediate results
    tl.store(out_ptr + offsets, (y + x) * weight + bias, mask=mask)

@torch.fx.wrap
def fused_layernorm_forward(bias, weight, x, y):
    # Determine tensor size and launch grid
    n_elements = x.numel()
    
    # Handle different tensor shapes by flattening for optimal memory access
    if x.ndim > 1:
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
    else:
        x_flat = x
        y_flat = y
    
    out_flat = torch.empty_like(x_flat)
    
    # Intelligent block size selection based on tensor characteristics
    # For very small tensors, use smaller blocks to reduce overhead
    if n_elements < 5000:
        BLOCK_SIZE = 64
    elif n_elements < 50000:
        BLOCK_SIZE = 256
    elif n_elements < 500000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate optimal grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Ensure we don't oversubscribe GPU resources
    max_grid_size = 65535  # Maximum grid size for CUDA
    grid_size = min(grid_size, max_grid_size)
    
    # Launch kernel with autotuned configuration
    grid = (grid_size,)
    
    fused_layernorm_kernel[grid](
        bias_ptr=bias,
        weight_ptr=weight,
        x_ptr=x_flat,
        y_ptr=y_flat,
        out_ptr=out_flat,
        n_elements=n_elements,
        bias_size=bias.numel(),
        weight_size=weight.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Restore original shape if needed
    if x.ndim > 1:
        out = out_flat.reshape(x.shape)
    else:
        out = out_flat
    
    return out

# Replacement function
def replacement_func():
    return fused_layernorm_forward