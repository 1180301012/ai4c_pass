import torch
import triton
import triton.language as tl


@triton.jit
def optimized_slice_transpose_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized slice + transpose kernel.
    input: [1, B, N, K]
    output: [1, B, K, N-1]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offs = tl.arange(0, BLOCK_SIZE)
    total = B * K * (N - 1)
    mask = offs < (total - block_start)
    
    base_offsets = block_start + offs
    
    b = base_offsets // (K * (N - 1))
    rem = base_offsets % (K * (N - 1))
    k_h = rem // (N - 1)
    n = rem % (N - 1)
    
    # Slice: n -> n+1, then transpose: swap k_h and n
    input_offsets = b * N * K + (n + 1) * K + k_h
    
    vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + base_offsets, vals, mask=mask)


@torch.fx.wrap
def optimized_slice_transpose(v):
    """
    Optimized slice + transpose using Triton.
    """
    B = v.shape[1]
    N = v.shape[2]
    K = v.shape[3]
    
    output = torch.empty((1, B, K, N - 1), dtype=v.dtype, device=v.device)
    
    total_elements = B * K * (N - 1)
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_slice_transpose_kernel[(num_programs,)](
        v, output, B, N, K, BLOCK_SIZE
    )
    
    return output


def pattern(v):
    """
    Pattern to match slice + transpose operation.
    """
    tmp_2 = v[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(v):
    """Extract arguments needed for the replacement function."""
    return (v,)


def replacement_func():
    """
    Returns a replacement function.
    """
    return optimized_slice_transpose