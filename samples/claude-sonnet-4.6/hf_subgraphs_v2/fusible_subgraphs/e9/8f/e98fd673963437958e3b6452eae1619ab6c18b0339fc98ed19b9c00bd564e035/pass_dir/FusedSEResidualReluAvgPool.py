import operator
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Full-fusion kernel: sigmoid(in2)*in1 + in0 -> relu -> global-avg-pool
# Grid: (C,)   in0,in1:[1,C,H,W]  in2:[1,1,C]  out:[1,C]
# ---------------------------------------------------------------------------
@triton.jit
def _full_fused_kernel(
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    c    = tl.program_id(0)
    raw  = tl.load(in2_ptr + c).to(tl.float32)
    scale = 1.0 / (1.0 + tl.exp(-raw))

    base = c * HW
    acc  = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x0 = tl.load(in0_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(in1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        v  = tl.maximum(x1 * scale + x0, 0.0)
        acc += v

    total = tl.sum(acc, axis=0) / HW
    tl.store(out_ptr + c, total)


@torch.fx.wrap
def _fused_all(in_0, in_1, in_2):
    B, C, H, W = in_1.shape
    HW = H * W
    out = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)
    _full_fused_kernel[(B * C,)](in_0, in_1, in_2, out, C=C, HW=HW, BLOCK_HW=256)
    return out


# ---------------------------------------------------------------------------
# Pattern: sigmoid + view + expand_as + mul
# (iadd omitted – will rely on FusedReluAvgPoolFlatten for the rest)
# ---------------------------------------------------------------------------
def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def _se_scale_kernel(in1_ptr, in2_ptr, out_ptr, C, HW, BLOCK_HW: tl.constexpr):
    c    = tl.program_id(0)
    raw  = tl.load(in2_ptr + c).to(tl.float32)
    scale = 1.0 / (1.0 + tl.exp(-raw))
    base = c * HW
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x1 = tl.load(in1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + offs, x1 * scale, mask=mask)


@torch.fx.wrap
def _fused_se_scale(in_1, in_2):
    B, C, H, W = in_1.shape
    out = torch.empty_like(in_1)
    _se_scale_kernel[(B * C,)](in_1, in_2, out, C=C, HW=H * W, BLOCK_HW=256)
    return out


def replacement_func():
    return _fused_se_scale