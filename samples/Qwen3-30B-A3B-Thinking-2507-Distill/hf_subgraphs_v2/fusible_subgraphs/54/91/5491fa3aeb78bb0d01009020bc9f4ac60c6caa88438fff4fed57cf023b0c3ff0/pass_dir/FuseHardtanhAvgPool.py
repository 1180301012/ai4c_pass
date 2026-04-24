import torch
import triton
import triton.language as tl


def _next_power_of_2(n):
    p = 1
    while p < n:
        p <<= 1
    return max(p, 16)


@triton.jit
def hardtanh_avgpool_kernel(
    x_ptr,
    out_ptr,
    HW,
    BC,
    BLOCK_HW: tl.constexpr,
):
    """1D fused kernel: hardtanh [0,6] + global average pool, one program per (b,c)."""
    pid  = tl.program_id(0)
    base = pid * HW

    offsets = tl.arange(0, BLOCK_HW)
    mask    = offsets < HW

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    x = tl.where(x < 0.0, 0.0, x)
    x = tl.where(x > 6.0, 6.0, x)

    tl.store(out_ptr + pid, tl.sum(x, axis=0) / HW)


@torch.fx.wrap
def fused_hardtanh_avgpool(x):
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)

    BLOCK_HW = _next_power_of_2(HW)
    BLOCK_HW = min(BLOCK_HW, 512)

    hardtanh_avgpool_kernel[(BC,)](
        x, out, HW, BC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_hardtanh_avgpool