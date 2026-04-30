import torch
import triton
import triton.language as tl


@triton.jit
def fused_cumsum_mul_sub_add_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: (cumsum(input, dim=1) * input - 1 + 2).long()
    Optimized for small tensors with minimal overhead.
    """
    pid = tl.program_id(0)
    start_offset = pid * BLOCK_SIZE
    
    if start_offset >= n_elements:
        return
    
    # Load and compute in one pass
    offs = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input_ptr + offs, mask=mask, other=0).to(tl.int64)
    
    # Cumsum along axis 0
    cs = tl.cumsum(x, 0)
    
    # Compute: cs * x - 1 + 2 = cs * x + 1
    result = cs * x + 1
    
    tl.store(output_ptr + offs, result, mask=mask)


@torch.fx.wrap
def fused_cumsum_mul_sub_add(x):
    """
    Fused function that computes: (cumsum(x, dim=1) * x - 1 + 2).long()
    Input: x with shape [1, 13], dtype int64
    Output: same shape, dtype int64
    """
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Use block size larger than n_elements for single program execution
    BLOCK_SIZE = 16
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_cumsum_mul_sub_add_kernel[grid](
        x, out, n_elements, BLOCK_SIZE
    )
    
    return out


def pattern(in_0):
    """
    Match the pattern: cumsum + multiply + subtract + cast + add
    """
    tmp_0 = in_0
    tmp_1 = torch.cumsum(tmp_0, dim=1)
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_cumsum_mul_sub_add