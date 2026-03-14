import torch
import triton
import triton.language as tl

# Pattern matching function to identify the mathematical operations sequence
def pattern(x):
    """
    Matches the sequence: sigmoid() - 0.25 * PI
    """
    tmp_5 = x.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel that fuses sigmoid, subtraction, and multiplication
@triton.jit
def fused_math_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Fused mathematical operations: sigmoid(x) - 0.25 * PI
    Optimized for good performance with enhanced memory coalescing
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input with cache hints for better performance
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused mathematical operations: sigmoid(x) - 0.25 * PI
    # Use optimized sigmoid computation
    sigmoid_x = tl.sigmoid(x)  # Use Triton's built-in sigmoid for better performance
    
    # Pre-compute constants as compile-time constants for efficiency
    pi_value = 3.141592653589793
    sub_value = 0.25
    
    # Fused operations: (sigmoid(x) - 0.25) * PI
    # Vectorized computation for better performance
    result = (sigmoid_x - sub_value) * pi_value
    
    # Store result with cache hints
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper function (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_math_ops(x):
    """
    Hybrid fused mathematical operations: sigmoid(x) - 0.25 * PI
    
    For small tensors, uses PyTorch's built-in operations for better performance.
    For larger tensors, uses optimized Triton kernel.
    """
    N = x.numel()
    
    # Use different strategies based on tensor size to avoid overhead
    if N <= 4096:  # Very small tensors - use PyTorch operations
        # For very small tensors, kernel launch overhead outweighs benefits
        out = (x.sigmoid() - 0.25) * 3.141592653589793
        return out
    elif N <= 16384:  # Small tensors - use optimized Triton with small blocks
        return _triton_math_ops(x, N, BLOCK_SIZE=128)
    elif N <= 65536:  # Medium tensors - use optimized Triton with medium blocks
        return _triton_math_ops(x, N, BLOCK_SIZE=256)
    elif N <= 262144:  # Large tensors - use optimized Triton with large blocks
        return _triton_math_ops(x, N, BLOCK_SIZE=512)
    else:  # Very large tensors - use optimized Triton with largest blocks
        return _triton_math_ops(x, N, BLOCK_SIZE=1024)

def _triton_math_ops(x, N, BLOCK_SIZE):
    """Helper function to launch Triton kernel with specific block size"""
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape and dtype as input
    out = torch.empty_like(x)
    
    # Launch the kernel with optimal configuration
    fused_math_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_math_ops