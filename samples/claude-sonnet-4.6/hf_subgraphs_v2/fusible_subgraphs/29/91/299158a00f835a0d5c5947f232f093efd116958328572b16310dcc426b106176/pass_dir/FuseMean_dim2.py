import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # num_warps = BLOCK_H/32  → exactly 1 element per thread, zero idle threads
        # BLOCK_H=64: 448/64=7 blocks per row, zero H-masking waste
        triton.Config({'BLOCK_H':  64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_H':  64}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_H':  64}, num_warps=2, num_stages=8),
        # BLOCK_H=128: 4 blocks per row (one partially masked), higher throughput/block
        triton.Config({'BLOCK_H': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_H': 128}, num_warps=4, num_stages=8),
        # BLOCK_H=256: 2 blocks per row, best SM occupancy for large B (fits 8 blocks/SM)
        triton.Config({'BLOCK_H': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_H': 256}, num_warps=8, num_stages=8),
    ],
    key=['SEQ_LEN', 'HIDDEN'],
)
@triton.jit
def _mean_dim1_kernel(
    x_ptr, out_ptr,
    B, SEQ_LEN, HIDDEN,
    BLOCK_H: tl.constexpr,
):
    """
    Loop over SEQ_LEN with software pipelining.
    Grid: (B, ceil(HIDDEN / BLOCK_H))
    num_warps = BLOCK_H/32 → every thread active, no idle warps.
    """
    b       = tl.program_id(0)
    h_block = tl.program_id(1)

    h_offs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < HIDDEN

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    x_base = x_ptr + b * SEQ_LEN * HIDDEN

    for s in range(SEQ_LEN):
        v = tl.load(x_base + s * HIDDEN + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        acc += v

    tl.store(out_ptr + b * HIDDEN + h_offs, acc / SEQ_LEN, mask=h_mask)


@torch.fx.wrap
def triton_mean_dim2(x):
    B       = x.shape[0]
    SEQ_LEN = x.shape[1]
    HIDDEN  = x.shape[2]

    out = torch.empty((B, HIDDEN), dtype=x.dtype, device=x.device)

    grid = lambda meta: (B, triton.cdiv(HIDDEN, meta['BLOCK_H']))

    _mean_dim1_kernel[grid](
        x, out,
        B, SEQ_LEN, HIDDEN,
    )

    return out


def pattern(x):
    return x.mean(-2)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_dim2