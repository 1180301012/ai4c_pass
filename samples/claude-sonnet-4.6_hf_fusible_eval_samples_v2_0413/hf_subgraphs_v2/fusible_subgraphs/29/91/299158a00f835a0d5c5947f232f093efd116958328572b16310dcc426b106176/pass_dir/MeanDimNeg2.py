import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Mean over dim -2 for 3-D tensors [B, S, D] → [B, D].
#
# Key optimisation vs earlier drafts:
#   • eviction_policy="evict_first" on every load so we stream data through
#     the L1/L2 caches without polluting them (matches what PyTorch's ATen
#     streaming-reduce kernel does internally with the .cs cache-control).
#   • Grid is (B, cdiv(D, BLOCK_D)) — one program per (batch-row, D-tile).
#   • Autotune includes both fine-grained (BLOCK_D=32, 1 warp) and coarser
#     (BLOCK_D=128/256, 4-8 warps) configs so the tuner can pick the shape
#     that best saturates the memory pipeline for each (B, S, D) triple.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # fine-grained — maps well to "one warp per 32 output elements"
        triton.Config({'BLOCK_D': 32},  num_warps=1, num_stages=4),
        triton.Config({'BLOCK_D': 32},  num_warps=1, num_stages=3),
        triton.Config({'BLOCK_D': 64},  num_warps=2, num_stages=4),
        triton.Config({'BLOCK_D': 64},  num_warps=2, num_stages=3),
        # medium
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8, num_stages=3),
        # coarse — fewer, larger blocks; best for small B
        triton.Config({'BLOCK_D': 512}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_D': 512}, num_warps=16, num_stages=3),
    ],
    key=['B', 'S', 'D'],
)
@triton.jit
def _mean_neg2_kernel(
    x_ptr,
    out_ptr,
    B,
    S,
    D,
    stride_xb,
    stride_xs,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b     = tl.program_id(0)
    d_blk = tl.program_id(1)

    d_off  = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)
    base = x_ptr + b * stride_xb + d_off

    for s in range(S):
        # evict_first: don't pollute caches with data read only once
        x = tl.load(base + s * stride_xs,
                    mask=d_mask, other=0.0,
                    eviction_policy="evict_first")
        acc = acc + x.to(tl.float32)

    result = acc * (1.0 / S)

    if IS_FP16:
        result = result.to(tl.float16)
    elif IS_BF16:
        result = result.to(tl.bfloat16)

    tl.store(out_ptr + b * D + d_off, result, mask=d_mask)


@torch.fx.wrap
def triton_mean_neg2(x):
    B = x.shape[0]
    S = x.shape[1]
    D = x.shape[2]

    out = torch.empty((B, D), dtype=x.dtype, device=x.device)

    IS_FP16 = x.dtype == torch.float16
    IS_BF16 = x.dtype == torch.bfloat16

    grid = lambda meta: (B, triton.cdiv(D, meta['BLOCK_D']))

    _mean_neg2_kernel[grid](
        x, out,
        B, S, D,
        x.stride(0), x.stride(1),
        IS_FP16, IS_BF16,
    )

    return out


def pattern(x):
    return x.mean(-2)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_neg2