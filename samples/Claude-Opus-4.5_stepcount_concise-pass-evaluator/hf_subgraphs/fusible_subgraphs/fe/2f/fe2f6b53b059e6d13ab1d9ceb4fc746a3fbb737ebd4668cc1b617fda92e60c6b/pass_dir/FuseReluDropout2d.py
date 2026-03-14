import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation in model.py
def pattern(in_0):
    """
    Match ReLU followed by dropout2d pattern.
    Both tmp_0 (relu output) and tmp_1 (dropout output) are returned.
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for ReLU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
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
    
    # ReLU: max(0, x)
    zero = tl.zeros_like(x)
    out = tl.maximum(x, zero)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_dropout(in_0):
    """
    Optimized fused ReLU + dropout2d (training=False).
    Since training=False, dropout is identity, so we just compute ReLU.
    Returns (relu_out, relu_out) since both outputs are the same.
    """
    N = in_0.numel()
    
    # Allocate output tensor
    out = torch.empty_like(in_0)
    
    # Calculate grid
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    relu_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=N,
    )
    
    # Return both outputs (they're the same since dropout with training=False is identity)
    return (out, out)

# Replacement function - returns the wrapper function
def replacement_func():
    return fused_relu_dropout