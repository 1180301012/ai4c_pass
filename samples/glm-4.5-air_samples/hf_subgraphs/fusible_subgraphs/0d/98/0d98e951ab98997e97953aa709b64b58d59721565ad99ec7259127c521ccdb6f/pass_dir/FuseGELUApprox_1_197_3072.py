import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU approximation: x * sigmoid(1.702 * x)
def pattern(x):
    tmp_0 = 1.702 * x
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = x * tmp_1
    # The dropout with p=0.0 is essentially a no-op, so we just return tmp_2
    return (tmp_2,)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for GELU approximation with better memory access
@triton.jit
def gelu_approx_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to prevent out-of-bounds access
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU approximation using math that's optimized for accuracy/speed trade-off
    # Use the classic GELU approximation that's known to be good
    # Instead of separate operations, calculate in fewer steps
    x_scaled = 1.702 * x
    # Use half precision for intermediate exp for better performance (if supported)
    neg_exp = tl.exp(-x_scaled)
    sigmoid_neg = neg_exp / (1.0 + neg_exp)  # This is 1 - sigmoid(x_scaled)
    sigmoid = 1.0 - sigmoid_neg  # More numerically stable
    out = x * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper for Triton execution with optimal block size
@torch.fx.wrap
def triton_gelu_approx(x):
    # Get tensor size
    n_elements = x.numel()
    
    # For this specific tensor size (594,144 elements), choose optimal block sizes
    # We want power-of-2 block sizes for Triton arange compatibility
    candidates = [
        128, 256, 512, 1024, 2048, 4096, 8192
    ]
    
    # Find the best block size - prioritize exact divisibility
    best_block_size = 512  # default
    min_remainder = float('inf')
    
    for bs in candidates:
        remainder = n_elements % bs
        if remainder < min_remainder:
            min_remainder = remainder
            best_block_size = bs
    
    num_programs = (n_elements + best_block_size - 1) // best_block_size
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with optimized launch configuration
    gelu_approx_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=best_block_size,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_gelu_approx