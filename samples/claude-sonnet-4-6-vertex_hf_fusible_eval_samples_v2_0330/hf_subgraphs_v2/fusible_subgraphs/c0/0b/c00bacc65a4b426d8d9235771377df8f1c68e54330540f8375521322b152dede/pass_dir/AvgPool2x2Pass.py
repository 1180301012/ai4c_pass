import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)


def replacement_args(x):
    return (x,)


@triton.jit
def _avgpool2x2_fwd(
    x_ptr,
    out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_OW: tl.constexpr,
):
    nco_pid = tl.program_id(0)
    ow_pid  = tl.program_id(1)

    oh  = nco_pid % OH
    nc  = nco_pid // OH

    ow_offsets = ow_pid * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask = ow_offsets < OW

    ih0 = oh * 2
    iw0 = ow_offsets * 2

    in_base = nc * IH * IW

    x00 = tl.load(x_ptr + in_base + ih0       * IW + iw0,     mask=mask, other=0.0)
    x01 = tl.load(x_ptr + in_base + ih0       * IW + iw0 + 1, mask=mask, other=0.0)
    x10 = tl.load(x_ptr + in_base + (ih0 + 1) * IW + iw0,     mask=mask, other=0.0)
    x11 = tl.load(x_ptr + in_base + (ih0 + 1) * IW + iw0 + 1, mask=mask, other=0.0)

    avg = (x00 + x01 + x10 + x11) * 0.25

    out_base = nc * OH * OW
    tl.store(out_ptr + out_base + oh * OW + ow_offsets, avg, mask=mask)


@torch.fx.wrap
def _avg_pool_2x2(x):
    x = x.contiguous()
    N, C, IH, IW = x.shape
    OH = (IH + 1) // 2
    OW = (IW + 1) // 2
    out = x.new_empty(N, C, OH, OW)

    BLOCK_OW = 32
    grid = (N * C * OH, triton.cdiv(OW, BLOCK_OW))
    _avgpool2x2_fwd[grid](
        x, out,
        N, C, IH, IW, OH, OW,
        BLOCK_OW=BLOCK_OW,
    )
    return out


def replacement_func():
    return _avg_pool_2x2