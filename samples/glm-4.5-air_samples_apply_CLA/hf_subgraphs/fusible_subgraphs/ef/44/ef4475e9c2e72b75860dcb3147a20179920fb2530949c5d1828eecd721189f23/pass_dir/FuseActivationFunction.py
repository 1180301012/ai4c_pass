import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the exact computation from model.py
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton with overflow protection
@triton.jit
def fused_activation_kernel(
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
    
    # Load input data with vectorized memory access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused activation function with all constants inlined
    # result = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    x3 = x * x * x  # More efficient than pow(x, 3.0)
    tmp = x + 0.044715 * x3
    tmp = 0.7978845608028654 * tmp
    
    # Implement tanh using exp: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # Clamp input to exp to prevent overflow
    tmp_clamped = tl.where(tmp > 50.0, 50.0, tl.where(tmp < -50.0, -50.0, tmp))
    exp_tmp = tl.exp(tmp_clamped)
    exp_neg_tmp = tl.exp(-tmp_clamped)
    tanh_tmp = (exp_tmp - exp_neg_tmp) / (exp_tmp + exp_neg_tmp)
    
    result = 0.5 * x * (1.0 + tanh_tmp)
    
    # Store result with vectorized memory access
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch fx.wrap)
@torch.fx.wrap
def fused_activation(x):
    # Get tensor info
    N = x.numel()
    
    # Heuristic to choose optimal block size based on tensor size and dimensions
    total_elements = N
    if total_elements > 200_000_000:  # Very large tensors (>200M elements)
        BLOCK_SIZE = 4096
    elif total_elements > 100_000_000:  # Large tensors (>100M elements)
        BLOCK_SIZE = 2048
    elif total_elements > 10_000_000:  # Medium tensors (>10M elements)
        BLOCK_SIZE = 1024
    else:  # Regular tensors
        BLOCK_SIZE = 256 if total_elements < 1_000_000 else 512
    
    # Calculate number of programs needed
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with optimized parameters
    fused_activation_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_activation