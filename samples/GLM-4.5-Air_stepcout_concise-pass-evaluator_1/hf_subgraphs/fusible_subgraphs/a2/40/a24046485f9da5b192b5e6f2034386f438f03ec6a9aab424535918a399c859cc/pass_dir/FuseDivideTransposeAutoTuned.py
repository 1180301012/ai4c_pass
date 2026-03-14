import torch
import triton
import triton.language as tl

# Pattern matching function - must match exact computation structure from model.py
def pattern(in_0):
    """
    Pattern to match: division by constant followed by transpose of last two dimensions.
    
    This matches the computation:
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)
    """
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    """Extract the input tensor argument for the fused kernel."""
    return (in_0,)

# Constants
DIVISOR = 1.6817928305074292

# Highly optimized kernel with autotuning for best performance
@triton.autotune(
    configs=[
        triton.Config(kwargs={'num_warps': 4, 'num_stages': 1}),
        triton.Config(kwargs={'num_warps': 8, 'num_stages': 1}),
        triton.Config(kwargs={'num_warps': 16, 'num_stages': 1}),
        triton.Config(kwargs={'num_warps': 8, 'num_stages': 2}),
        triton.Config(kwargs={'num_warps': 16, 'num_stages': 2}),
        triton.Config(kwargs={'num_warps': 32, 'num_stages': 2}),
    ],
    key=['stride_inner', 'stride_last'],
)
@triton.jit
def fused_divide_transpose_kernel(
    input_ptr,
    output_ptr,
    stride_inner,
    stride_last,
    DIVISOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Autotuned fused kernel that performs element-wise division and transpose.
    
    Automatically finds optimal warp count and pipeline stages for each tensor shape.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Create offset range for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (stride_inner * stride_last)
    
    # Convert to 2D coordinates for the last two dimensions
    inner = offsets // stride_last
    last = offsets % stride_last
    
    # Apply bounds checking directly in coordinates
    inner_mask = inner < stride_inner
    last_mask = last < stride_last
    
    # Combined mask for all bounds checking
    combined_mask = mask & inner_mask & last_mask
    
    # Transpose coordinates for output
    transposed_inner = last
    transposed_last = inner
    
    # Load input data with original coordinates and bounds checking
    input_ptrs = input_ptr + (inner * stride_inner + last)
    input_vals = tl.load(input_ptrs, mask=combined_mask, other=0.0)
    
    # Apply division with fused operation
    result_vals = input_vals / DIVISOR
    
    # Store to output with transposed coordinates and bounds checking
    output_ptrs = output_ptr + (transposed_inner * stride_inner + transposed_last)
    tl.store(output_ptrs, result_vals, mask=combined_mask)

@torch.fx.wrap
def fused_divide_transpose_kernel_wrapper(input_tensor):
    """
    Wrapper function that sets up and launches the autotuned fused kernel.
    
    The autotuner will automatically find the best configuration for each input shape.
    """
    # Get input shape - the last two dimensions will be transposed
    # Input shape: [..., inner_dim, last_dim]
    output_shape = input_tensor.shape[:-2] + input_tensor.shape[-2:][::-1]
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Compute strides for the input tensor (row-major order)
    strides = []
    current_stride = 1
    for dim in reversed(input_tensor.shape):
        strides.append(current_stride)
        current_stride *= dim
    strides = list(reversed(strides))
    
    # Extract strides for the last two dimensions
    stride_inner = strides[-2]  # Stride for the inner dimension (penultimate)
    stride_last = strides[-1]   # Stride for the last dimension
    
    # Use moderate block size with autotuning finding optimal configuration
    BLOCK_SIZE = 1024
    
    # Calculate grid size for the last two dimensions
    total_elements = stride_inner * stride_last
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with autotuning - autotuner finds best configuration
    fused_divide_transpose_kernel[(grid_size,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        stride_inner=stride_inner,
        stride_last=stride_last,
        DIVISOR=DIVISOR,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function - must return a zero-arg function that returns the kernel wrapper
def replacement_func():
    """Returns the autotuned kernel wrapper function."""
    return fused_divide_transpose_kernel_wrapper