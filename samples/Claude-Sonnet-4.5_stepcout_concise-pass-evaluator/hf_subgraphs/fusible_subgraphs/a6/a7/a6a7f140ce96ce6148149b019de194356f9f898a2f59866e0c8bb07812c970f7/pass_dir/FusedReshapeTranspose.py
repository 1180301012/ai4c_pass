import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Match reshape + transpose pattern on input
    """
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)
    tmp_6 = tmp_5.transpose(2, 3)
    return tmp_6

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_reshape_transpose_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused reshape + transpose kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Decode output linear index to [b, i, k, j, l, c]
    rem = offsets
    c = rem % 96
    rem = rem // 96
    l = rem % 7
    rem = rem // 7
    j = rem % 7
    rem = rem // 7
    k = rem % 19
    rem = rem // 19
    i = rem % 19
    
    # Map to input coordinates
    in_row = i * 7 + j
    in_col = k * 7 + l
    
    # Compute input linear index: (1, 133, 133, 96)
    in_idx = in_row * 133 * 96 + in_col * 96 + c
    
    # Load and store
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_reshape_transpose(in_0):
    """
    Fused reshape + transpose operation
    """
    # Output shape: (1, 19, 19, 7, 7, 96)
    out = torch.empty((1, 19, 19, 7, 7, 96), device=in_0.device, dtype=in_0.dtype)
    
    N = out.numel()
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    fused_reshape_transpose_kernel[grid](
        in_0,
        out,
        N,
    )
    
    return out

def replacement_func():
    return fused_reshape_transpose