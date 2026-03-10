import torch
import triton
import triton.language as tl

# Pattern matching function for the scaled error function activation
def pattern(in_0):
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return tmp_4

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused scaled error function activation
@triton.jit
def scaled_erf_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused scaled error function activation
    # y = 0.5 * x * (1.0 + erf(x / √2))
    # Precomputed constant for better performance
    erf_input = x * 0.7071067811865476  # x / √2
    erf_result = tl.erf(erf_input)
    # Fuse operations to reduce register pressure and improve performance
    final_result = (x * 0.5) * (1.0 + erf_result)
    
    # Store result
    tl.store(out_ptr + offsets, final_result, mask=mask)

# Kernel wrapper function with dynamic block size selection
@torch.fx.wrap
def scaled_erf_activation(in_0):
    N = in_0.numel()
    
    # Optimized block sizes for different tensor sizes with better GPU occupancy
    if N <= 65536:  # Small tensors
        BLOCK_SIZE = 256
    elif N <= 262144:  # Medium tensors
        BLOCK_SIZE = 1024
    else:  # Large tensors
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    scaled_erf_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return scaled_erf_activation