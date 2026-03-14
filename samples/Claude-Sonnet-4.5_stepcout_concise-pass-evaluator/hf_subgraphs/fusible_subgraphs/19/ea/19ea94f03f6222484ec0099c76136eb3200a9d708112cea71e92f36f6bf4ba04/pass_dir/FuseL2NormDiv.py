import torch
import triton
import triton.language as tl

def pattern(in_1):
    """
    Pattern to match L2 normalization: norm + division
    """
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def l2_norm_vectorized_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized L2 normalization kernel with coalesced memory access.
    """
    row_idx = tl.program_id(0)
    
    # Row offset
    row_offset = row_idx * n_cols
    
    # Load data with vectorized access
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(input_ptr + row_offset + cols, mask=mask, other=0.0, eviction_policy="evict_last")
    
    # Compute L2 norm: sqrt(sum(x^2))
    norm = tl.sqrt(tl.sum(x * x))
    
    # Normalize and store with vectorized access
    result = x / norm
    tl.store(output_ptr + row_offset + cols, result, mask=mask, eviction_policy="evict_last")

@torch.fx.wrap
def fused_l2_norm_triton(x):
    """
    Fused L2 normalization using Triton with vectorized memory access.
    """
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    
    BLOCK_SIZE = 2048
    grid = (n_rows,)
    
    l2_norm_vectorized_kernel[grid](
        input_ptr=x,
        output_ptr=y,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
        num_stages=2,
    )
    
    return y

def replacement_func():
    return fused_l2_norm_triton