import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation from model.py
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused GeLU computation with manual optimization
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with optimized memory access patterns
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants for GeLU approximation - direct values for compilation compatibility
    cubic_coeff = 0.044715
    tanh_coeff = 0.7978845608028654  # sqrt(2/pi)
    linear_coeff = 0.5
    
    # Optim computation pipeline using native Triton operations
    x_cubed = x * x * x  # More efficient than torch.pow
    cubic_term = x_cubed * cubic_coeff
    inner_linear = x + cubic_term
    tanh_input = inner_linear * tanh_coeff
    
    # Optimized tanh using exponential identity
    exp_2x = tl.exp(2.0 * tanh_input)
    tanh_output = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Final fused computation
    gelu_inner = 1.0 + tanh_output
    out = x * gelu_inner * linear_coeff
    
    # Vectorized store operation
    tl.store(out_ptr + offsets, out, mask=mask)

# Optimized kernel wrapper with adaptive block sizing
@torch.fx.wrap
def fused_gelu(x):
    # Get tensor properties
    n_elements = x.numel()
    dtype = x.dtype
    device = x.device
    
    # Smart adaptive block sizing for different tensor sizes
    if n_elements < 1 * 1024 * 1024:  # < 1M elements
        BLOCK_SIZE = 128
    elif n_elements < 10 * 1024 * 1024:  # < 10M elements
        BLOCK_SIZE = 256
    elif n_elements < 100 * 1024 * 1024:  # < 100M elements
        BLOCK_SIZE = 512
    else:  # > 100M elements
        BLOCK_SIZE = 1024
    
    # Calculate optimal grid size efficiently
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with proper type
    out = torch.empty_like(x)
    
    # Launch optimized kernel
    gelu_kernel[(num_programs, 1, 1)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function that returns the kernel wrapper (no arguments)
def replacement_func():
    return fused_gelu