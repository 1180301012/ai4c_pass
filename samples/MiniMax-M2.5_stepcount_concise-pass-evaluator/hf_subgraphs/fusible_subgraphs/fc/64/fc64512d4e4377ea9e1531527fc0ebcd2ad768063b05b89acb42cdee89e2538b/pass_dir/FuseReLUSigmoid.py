import torch
import triton
import triton.language as tl


# Pattern matching function - matches relu followed by sigmoid
def pattern(in_0):
    """
    Simple pattern: just match sigmoid to debug the matching issue
    """
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    # Extract and return arguments needed for the replacement
    return (in_0,)


# Optimized Triton kernel that fuses relu and sigmoid
@triton.jit
def relu_sigmoid_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Apply Sigmoid: 1 / (1 + exp(-x))
    # Using tl.sigmoid for efficiency
    result = tl.sigmoid(x_relu)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


# Autotuned Triton kernel with multiple configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_sigmoid_autotuned_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Apply Sigmoid: 1 / (1 + exp(-x))
    result = tl.sigmoid(x_relu)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


# Kernel wrapper function
@torch.fx.wrap
def relu_sigmoid_fused(x):
    """
    Fused ReLU + Sigmoid kernel.
    Applies ReLU first (clamping negative values to 0), then sigmoid.
    """
    # Get total number of elements
    n_elements = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Define grid based on number of elements and block size
    # Using autotuned kernel
    grid = lambda opts: (triton.cdiv(n_elements, opts['BLOCK_SIZE']),)
    
    # Launch kernel
    relu_sigmoid_autotuned_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output


# Replacement function - returns the fused kernel function
def replacement_func():
    return relu_sigmoid_fused