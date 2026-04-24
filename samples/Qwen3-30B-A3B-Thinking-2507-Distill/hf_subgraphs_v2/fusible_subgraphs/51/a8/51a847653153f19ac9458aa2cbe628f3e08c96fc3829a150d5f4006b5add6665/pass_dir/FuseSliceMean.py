import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=1),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _mean_kernel_slice(
    in_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * HW

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    vals = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals)
    mean_f32 = total / HW

    tl.store(out_ptr + pid, mean_f32)


@torch.fx.wrap
def fused_mean_only(tmp_1):
    """Replace: tmp_2 = tmp_1.mean((2, 3), keepdim=True)  — fallback."""
    B, C, H, W = tmp_1.shape
    HW = H * W
    BC = B * C
    out_mean = torch.empty((B, C, 1, 1), dtype=tmp_1.dtype, device=tmp_1.device)
    _mean_kernel_slice[(BC,)](tmp_1, out_mean, HW)
    return out_mean


def pattern(tmp_1):
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_2


def replacement_args(tmp_1):
    return (tmp_1,)


def replacement_func():
    return fused_mean_only