import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},   num_warps=2),
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 64},   num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 128},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16, num_stages=2),
    ],
    key=['HW', 'BC'],
)
@triton.jit
def avgpool2d_kernel(
    x_ptr, out_ptr,
    HW, BC,
    BLOCK_HW: tl.constexpr,
):
    # 2-D grid: dim0 = BC (b*c index), dim1 = HW tile index
    bc_idx = tl.program_id(0)
    hw_blk = tl.program_id(1)

    hw_base = hw_blk * BLOCK_HW
    offsets = hw_base + tl.arange(0, BLOCK_HW)
    mask    = offsets < HW

    x_base = bc_idx * HW
    x      = tl.load(x_ptr + x_base + offsets, mask=mask, other=0.0)
    total  = tl.sum(tl.where(mask, x, 0.0), axis=0)

    tl.atomic_add(out_ptr + bc_idx, total / HW)


# ── Pattern: adaptive_avg_pool2d(x, 1) ────────────────────────────────────────

def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, 1)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def triton_avgpool(x):
    B  = x.shape[0]
    C  = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    BC = B * C

    pooled = torch.zeros((B, C), dtype=x.dtype, device=x.device)

    avgpool2d_kernel[lambda meta: (BC, triton.cdiv(HW, meta['BLOCK_HW']))](
        x, pooled, HW, BC,
    )
    return pooled


def replacement_func():
    return triton_avgpool