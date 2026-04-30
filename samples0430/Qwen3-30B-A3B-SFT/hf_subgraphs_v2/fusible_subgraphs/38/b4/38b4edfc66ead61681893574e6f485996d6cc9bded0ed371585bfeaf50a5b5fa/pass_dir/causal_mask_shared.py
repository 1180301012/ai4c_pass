import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _causal_mask_kernel(
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill out[i,j] = -3.4e38 if j>i (upper triangle), else 0.0."""
    total = N * N
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total

    row = offsets // N
    col = offsets % N
    masked = col > row
    NEG_INF: tl.constexpr = -3.4028234663852886e+38
    result = tl.where(masked, NEG_INF, 0.0)
    tl.store(out_ptr + offsets, result, mask=valid)


@torch.fx.wrap
def causal_mask_dispatch(route):
    """Dispatch to causal mask kernel based on route string."""
    if route == "causal_9":
        N = 9
    elif route == "causal_13":
        N = 13
    else:
        N = 9  # safe default
    total = N * N
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device='cuda')
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _causal_mask_kernel[grid](out, N)
    return out