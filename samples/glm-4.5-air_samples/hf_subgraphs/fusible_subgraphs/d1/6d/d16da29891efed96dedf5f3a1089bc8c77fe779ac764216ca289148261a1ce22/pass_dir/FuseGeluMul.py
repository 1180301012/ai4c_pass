import torch
import triton
import triton.language as tl

# Pattern matching for GELU followed by element-wise multiplication
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    return tmp_1

# Extract arguments for the replacement function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for fused GELU + multiplication
@triton.jit
def gelu_mul_kernel(
    x_ptr,      # Input tensor 1 (for GELU)
    y_ptr,      # Input tensor 2 (for multiplication)  
    out_ptr,    # Output tensor
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU using exp to avoid tanh dependency
    # GELU(x) = 0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # Using tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    sqrt_2_over_pi = tl.sqrt(2.0 / 3.141592653589793)
    x_cubed = x * x * x
    inner = x + 0.044715 * x_cubed
    tanh_arg = sqrt_2_over_pi * inner
    
    # Implement tanh using exponential for better compatibility
    exp_2z = tl.exp(2.0 * tanh_arg)
    tanh_val = (exp_2z - 1.0) / (exp_2z + 1.0)
    
    gelu_val = x * 0.5 * (1.0 + tanh_val)
    
    # Element-wise multiplication
    out = gelu_val * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_gelu_mul(x, y):
    # Get tensor properties
    N = x.numel()
    
    # Optimal block size for GELU computation
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    gelu_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_gelu_mul