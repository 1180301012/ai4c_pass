import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Matches avg_pool2d with kernel=2, stride=2, padding=0, ceil_mode=True,
    count_include_pad=False, divisor_override=None.
    """
    return torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 16},  num_warps=4),
        triton.Config({'BLOCK_OW': 32},  num_warps=4),
        triton.Config({'BLOCK_OW': 64},  num_warps=8),
        triton.Config({'BLOCK_OW': 128}, num_warps=8),
    ],
    key=['OW', 'OH'],
)
@triton.jit
def _avg_pool2x2_kernel(
    x_ptr,
    out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_OW: tl.constexpr,
):
    """
    Grid: (N * C * OH, ceil(OW / BLOCK_OW))
    kernel=2, stride=2, padding=0, count_include_pad=False.
    Each output[n, c, oh, ow] = mean of input[n, c, 2*oh:2*oh+2, 2*ow:2*ow+2]
    """
    nco_pid = tl.program_id(0)  # encodes (n * C + c) * OH + oh
    ow_pid  = tl.program_id(1)

    oh  = nco_pid % OH
    nc  = nco_pid // OH
    c   = nc % C
    n   = nc // C

    ow_offsets = ow_pid * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask = ow_offsets < OW

    ih0 = oh * 2
    iw0 = ow_offsets * 2          # shape: (BLOCK_OW,)

    in_base = (n * C + c) * IH * IW

    # 2×2 window loads (strided in W, so accesses at iw0 and iw0+1)
    x00 = tl.load(x_ptr + in_base + ih0       * IW + iw0,     mask=mask, other=0.0)
    x01 = tl.load(x_ptr + in_base + ih0       * IW + iw0 + 1, mask=mask, other=0.0)
    x10 = tl.load(x_ptr + in_base + (ih0 + 1) * IW + iw0,     mask=mask, other=0.0)
    x11 = tl.load(x_ptr + in_base + (ih0 + 1) * IW + iw0 + 1, mask=mask, other=0.0)

    avg = (x00 + x01 + x10 + x11) * 0.25

    out_base = (n * C + c) * OH * OW
    tl.store(out_ptr + out_base + oh * OW + ow_offsets, avg, mask=mask)


@torch.fx.wrap
def _avg_pool_2x2(x):
    """
    Triton-based 2×2 average pooling with stride 2.
    Equivalent to avg_pool2d(x, 2, 2, 0, ceil_mode=True, count_include_pad=False).
    """
    x = x.contiguous()
    N, C, IH, IW = x.shape
    # ceil_mode=True: output size = ceil(dim / stride)
    OH = (IH + 1) // 2
    OW = (IW + 1) // 2
    out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)

    grid = lambda meta: (N * C * OH, triton.cdiv(OW, meta['BLOCK_OW']))
    _avg_pool2x2_kernel[grid](
        x, out,
        N, C, IH, IW, OH, OW,
    )
    return out


def replacement_func():
    return _avg_pool_2x2