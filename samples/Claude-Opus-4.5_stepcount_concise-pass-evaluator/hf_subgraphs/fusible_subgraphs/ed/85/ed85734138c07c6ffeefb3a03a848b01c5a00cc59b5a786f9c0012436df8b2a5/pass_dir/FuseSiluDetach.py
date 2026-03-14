import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized SiLU kernel using Triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_kernel(
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
    
    # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_detach(in_0, in_1, in_2):
    # For SiLU computation on in_0
    N = in_0.numel()
    
    # Allocate output tensor for SiLU result
    silu_out = torch.empty_like(in_0)
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    silu_kernel[grid](
        x_ptr=in_0,
        out_ptr=silu_out,
        n_elements=N,
    )
    
    # Detach operations - these create views without gradient tracking
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = silu_out.detach()
    
    return (tmp_1, tmp_2, tmp_3, silu_out)

# Replacement function - returns the optimized function
def replacement_func():
    return fused_silu_detach