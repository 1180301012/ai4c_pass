import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU activation
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

# Triton kernel for fused GELU - fixed block size
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
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = x * 0.7071067811865476
    erf_val = tl.math.erf(x_scaled)
    result = 0.5 * x * (1.0 + erf_val)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_gelu(x):
    N = x.numel()
    out = torch.empty_like(x)
    
    # Use fixed block size of 4096 with 8 warps - good balance
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gelu_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_gelu