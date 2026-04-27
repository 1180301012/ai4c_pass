import torch
import triton
import triton.language as tl


@triton.jit
def _avgpool_flatten_kernel(
    x_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Grid: (C,)  [B=1 always for these graphs]
    x   : [1, C, H, W]  contiguous  -- input is already relu-applied
    out : [1, C]         contiguous
    Computes: out[c] = mean(x[c,:,:])
    """
    c    = tl.program_id(0)
    base = c * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += v

    total = tl.sum(acc, axis=0) / HW
    tl.store(out_ptr + c, total)


@torch.fx.wrap
def _fused_avgpool_flatten(x):
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    _avgpool_flatten_kernel[(B * C,)](x, out, C=C, HW=HW, BLOCK_HW=256)
    return out


# ---------------------------------------------------------------------------
# Pattern: adaptive_avg_pool2d(x, 1) -> flatten(_, 1, -1)
# (input x is the relu-already-applied activation; relu node stays in graph)
# ---------------------------------------------------------------------------
def pattern(x):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(x):
    return (x,)


def replacement_func():
    return _fused_avgpool_flatten