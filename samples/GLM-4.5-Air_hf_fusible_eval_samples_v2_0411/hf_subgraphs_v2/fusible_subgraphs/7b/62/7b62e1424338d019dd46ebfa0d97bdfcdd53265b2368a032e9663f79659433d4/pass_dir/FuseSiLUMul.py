import torch
import triton
import triton.language as tl

# Pattern matching function - matches SiLU followed by multiplication
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    return tmp_1

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Triton kernel for fused SiLU + multiplication
@triton.jit
def fused_silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused SiLU * y: (sigmoid(x) * x) * y
    sigmoid_x = tl.sigmoid(x)
    silu_x = sigmoid_x * x
    out = silu_x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Fused kernel wrapper with optimized block sizes
@torch.fx.wrap
def fused_silu_mul_op(in_0, in_1):
    N = in_0.numel()
    
    # Optimized block sizes for fused operations
    if N < 10000:
        BLOCK_SIZE = 2048    # Larger blocks for reduced overhead
    elif N < 500000:
        BLOCK_SIZE = 4096    # Optimal for medium-large tensors
    else:
        BLOCK_SIZE = 8192    # Very large blocks for huge tensors
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    # Launch fused kernel
    fused_silu_mul_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_silu_mul_op