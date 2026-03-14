import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation from model.py
def pattern(in_0):
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_1 = None
    tmp_3 = 1.0 + tmp_2
    tmp_2 = None
    tmp_4 = tmp_0 * tmp_3
    tmp_0 = tmp_3 = None
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized GELU kernel using Triton with autotuning
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: GELU(x) = x * 0.5 * (1.0 + erf(x / sqrt(2)))
    # Use compiler constants for better performance
    sqrt2_inv = 0.7071067811865476  # 1.0 / sqrt(2) ≈ 0.7071067811865476
    half = 0.5
    
    # Compute the fused GELU operation in optimal order for numerical stability
    x_scaled = x * sqrt2_inv
    erf_result = tl.erf(x_scaled)
    gelu = x * (half + half * erf_result)
    
    # Store the result
    tl.store(out_ptr + offsets, gelu, mask=mask)

# Maximum performance GELU kernel with vectorized constants
@triton.jit
def gelu_kernel_dynamic(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    # Optimized GPU execution pipeline
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Vectorized memory access with optimal alignment
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    
    # Pre-vectorized constants for maximum throughput
    SQRT2_INV = 0.7071067811865476
    HALF = 0.5
    ONE = 1.0
    
    # Compute GELU in mathematically equivalent form: 0.5 * x * (1.0 + erf(x * SQRT2_INV))
    
    # Step 1: Vectorized scaling operation
    x_scaled = x * SQRT2_INV
    
    # Step 2: Parallel erf computation (inherently serial but necessary)
    erf_result = tl.erf(x_scaled)
    
    # Step 3: Optimized arithmetic for FMA utilization
    one_plus_erf = ONE + erf_result
    gelu_result = x * HALF * one_plus_erf
    
    # Store result with vectorized memory access
    tl.store(out_ptr + offset, gelu_result, mask=mask)

# Ultra-optimized kernel wrapper with advanced memory management
@torch.fx.wrap
def gelu_kernel_wrapper_optimized(x):
    n_elements = x.numel()
    
    # Memory optimization: avoid unnecessary copy if already 1D
    if x.dim() == 1:
        x_flat = x
        out_flat = torch.empty_like(x, device=x.device, dtype=x.dtype)
        need_reshape = False
    else:
        x_flat = x.flatten()
        out_flat = torch.empty_like(x_flat, device=x.device, dtype=x.dtype)
        need_reshape = True
    
    # Advanced block size optimization based on tensor size and hardware
    if n_elements <= 50000:
        # Ultra-small tensors: minimize kernel launch overhead
        BLOCK_SIZE = 4096  # Very large blocks for fewer programs
    elif n_elements <= 200000:
        # Small tensors: good balance of parallelism and overhead
        BLOCK_SIZE = 2048
    elif n_elements <= 2000000:
        # Medium tensors: balance performance and parallelism
        BLOCK_SIZE = 1024
    elif n_elements <= 10000000:
        # Large tensors: moderate parallelism
        BLOCK_SIZE = 512  
    else:
        # Very large tensors: optimal parallelism
        BLOCK_SIZE = 256
    
    # Calculate optimal number of programs for full GPU utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_warps = 1  # Single warp for maximum throughput per SM
    
    # Launch ultra-optimized kernel
    gelu_kernel_dynamic[(num_programs,)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Reshape only if necessary to avoid extra memory operations
    if need_reshape:
        if x.dim() == 3:
            return out_flat.reshape(x.shape)
        else:
            return out_flat.view(x.shape)
    else:
        return out_flat

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return gelu_kernel_wrapper_optimized