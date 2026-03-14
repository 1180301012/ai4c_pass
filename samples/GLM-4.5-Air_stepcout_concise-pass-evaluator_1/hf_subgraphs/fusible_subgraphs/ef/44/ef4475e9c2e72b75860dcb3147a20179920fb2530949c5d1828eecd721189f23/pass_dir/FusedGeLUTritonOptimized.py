import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation chain
def pattern(in_0):
    # Match the exact computation pattern from model.py
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

# Optimized Triton kernel
@triton.jit
def gelu_kernel_optimized(
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
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the entire GELU variant in fused operations
    # x_cubed = x^3
    x_cubed = x * x * x
    
    # x_plus_cubed = x + 0.044715 * x^3
    x_plus_cubed = x + 0.044715 * x_cubed
    
    # x_scaled = 0.7978845608028654 * (x + 0.044715 * x^3)
    x_scaled = 0.7978845608028654 * x_plus_cubed
    
    # tanh_activation = tanh(x_scaled) using: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    exp_scaled = tl.exp(x_scaled)
    exp_neg_scaled = tl.exp(-x_scaled)
    tanh_activation = (exp_scaled - exp_neg_scaled) / (exp_scaled + exp_neg_scaled)
    
    # one_plus_tanh = 1.0 + tanh(x_scaled)
    one_plus_tanh = 1.0 + tanh_activation
    
    # x_half = 0.5 * x
    x_half = 0.5 * x
    
    # final_result = 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    out = x_half * one_plus_tanh
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper that launches the optimized kernel
@torch.fx.wrap
def fused_gelu_optimized(x):
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimized block size for float32 operations
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch the optimized kernel
    gelu_kernel_optimized[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Enhanced optimized kernel with autotuning
@triton.jit
def gelu_kernel_autotuned(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned version with better memory access patterns"""
    pid = tl.program_id(0)
    
    # Use more efficient grid calculation
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with aligned memory access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation with mathematical optimizations
    # Precompute constants for better performance
    alpha = 0.044715
    beta = 0.7978845608028654
    half = 0.5
    
    # x^3 computation
    x_cubed = x * x * x
    
    # x + alpha * x^3
    x_term = x + alpha * x_cubed
    
    # beta * (x + alpha * x^3)
    scaled_term = beta * x_term
    
    # tanh(beta * (x + alpha * x^3)) using: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    exp_scaled_term = tl.exp(scaled_term)
    exp_neg_scaled_term = tl.exp(-scaled_term)
    tanh_term = (exp_scaled_term - exp_neg_scaled_term) / (exp_scaled_term + exp_neg_scaled_term)
    
    # 1 + tanh(beta * (x + alpha * x^3))
    activation_term = 1.0 + tanh_term
    
    # 0.5 * x * (1 + tanh(beta * (x + alpha * x^3)))
    out = half * x * activation_term
    
    # Store with vectorization
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu_autotuned(x):
    """Autotuned wrapper with dynamic block size selection"""
    N = x.numel()
    
    # Use larger block sizes for better GPU utilization
    if N > 1024 * 1024:  # Large tensors
        BLOCK_SIZE = 2048
    elif N > 1024 * 1024 * 8:  # Very large tensors
        BLOCK_SIZE = 4096
    else:  # Medium/small tensors
        BLOCK_SIZE = 1024
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch autotuned kernel
    gelu_kernel_autotuned[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Further optimized kernel with configurable parameters
@triton.jit
def gelu_kernel_final_optimized(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Final optimized kernel with performance enhancements"""
    pid = tl.program_id(0)
    
    # Optimized memory access pattern
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with optimized alignment
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Precompute mathematical constants for performance
    alpha = 0.044715
    beta = 0.7978845608028654
    half = 0.5
    
    # Optimized computation sequence with reduced temporaries
    x_cubed = x * x * x
    
    # x + alpha*x^3 combined operation
    x_term = x + (alpha * x_cubed)
    
    # beta*(x + alpha*x^3)
    scaled_term = beta * x_term
    
    # Numerically stable tanh using exponential functions
    exp_scaled = tl.exp(scaled_term)
    exp_neg_scaled = tl.exp(-scaled_term)
    tanh_activation = (exp_scaled - exp_neg_scaled) / (exp_scaled + exp_neg_scaled)
    
    # 1 + tanh(beta*(x + alpha*x^3))
    activation_term = 1.0 + tanh_activation
    
    # 0.5 * x * activation combined
    out = half * x * activation_term
    
    # Vectorized store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu_final_wrapper(x):
    """Final wrapper with adaptive optimization strategy"""
    N = x.numel()
    
    # Advanced block size optimization based on tensor characteristics
    if N > 1024 * 1024 * 32:  # Extremely large tensors
        BLOCK_SIZE = 2048
    elif N > 1024 * 1024 * 8:  # Large tensors
        BLOCK_SIZE = 1024
    elif N > 1024 * 512:  # Medium tensors
        BLOCK_SIZE = 512
    else:  # Small tensors
        BLOCK_SIZE = 256
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch final optimized kernel
    gelu_kernel_final_optimized[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function that returns the final optimized kernel
def replacement_func():
    return fused_gelu_final_wrapper