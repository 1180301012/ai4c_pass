import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must match the exact computation structure including return
def pattern(in_0):
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3  # Match the final return value

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel - using sigmoid with efficient operations
@triton.jit
def fused_sigmoidmul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: x * sigmoid(scale * x)
    # Use more efficient operations sequence
    scaled_x = scale * x
    # Use sigmoid directly - it's optimized in Triton
    sigmoid_x = tl.sigmoid(scaled_x)
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Optimized kernel wrapper with minimal launch overhead
@torch.fx.wrap
def fused_sigmoidmul(x, scale=1.702):
    N = x.numel()
    
    # Use largest possible block sizes to minimize launch overhead
    # For our tensor (605,184 elements), we want to minimize kernel launch costs
    if N < 16384:
        BLOCK_SIZE = 1024
    elif N < 65536:
        BLOCK_SIZE = 2048
    elif N < 262144:
        BLOCK_SIZE = 4096
    elif N < 1048576:
        # For our case (605,184): use 8192 for minimal launch overhead
        # 605,184 / 8192 = 74 programs, very efficient for GPU
        BLOCK_SIZE = 8192
    else:
        BLOCK_SIZE = 16384
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_sigmoidmul_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - must return a function reference with no arguments
def replacement_func():
    return fused_sigmoidmul