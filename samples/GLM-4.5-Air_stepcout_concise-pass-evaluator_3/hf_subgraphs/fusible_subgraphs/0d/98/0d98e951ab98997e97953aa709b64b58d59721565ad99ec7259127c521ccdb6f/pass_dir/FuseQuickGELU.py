import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation
def pattern(in_0):
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return (tmp_3,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized quick GELU kernel with better performance
@triton.jit
def quick_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    const_1_702: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with vectorized access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized computation: fuse operations to minimize registers
    # Use intermediate var to reduce redundant computations
    linear = const_1_702 * x
    sigmoid_val = tl.sigmoid(linear)
    out = x * sigmoid_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)
    
    # Compiler should optimize register usage automatically

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_quick_gelu(x):
    N = x.numel()
    
    # Use moderate block size that's efficient and quick to compile
    BLOCK_SIZE = 1024  # Good compromise between efficiency and compilation time
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, device=x.device, dtype=x.dtype)
    
    quick_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        const_1_702=1.702,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_quick_gelu