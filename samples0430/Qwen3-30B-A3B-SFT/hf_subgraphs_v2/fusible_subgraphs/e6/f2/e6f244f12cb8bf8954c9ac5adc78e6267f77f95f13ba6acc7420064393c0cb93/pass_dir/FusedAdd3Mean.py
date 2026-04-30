import torch
import triton
import triton.language as tl
import operator

torch.fx.wrap(operator.iadd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add3_mean_kernel(
    in1_ptr,
    in2_ptr,
    in3_ptr,
    out_sum_ptr,
    out_mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """Fused (in1 + in2 + in3) + spatial mean for one (B,C) slice."""
    bc_idx = tl.program_id(0)
    base_offset = bc_idx * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    x1 = tl.load(in1_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in2_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(in3_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x1 + x2 + x3

    # Write sum
    tl.store(out_sum_ptr + base_offset + offsets, x.to(in1_ptr.dtype.element_ty), mask=mask)

    # Reduce to mean
    total = tl.sum(x, axis=0)
    mean_val = total / HW
    tl.store(out_mean_ptr + bc_idx, mean_val.to(out_mean_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_add3_mean(in_0, in_1, in_2):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    out_sum = torch.empty_like(in_0)
    out_mean = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    _fused_add3_mean_kernel[(BC,)](
        in_0, in_1, in_2, out_sum, out_mean,
        HW=HW,
    )

    return out_sum, out_mean


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 + in_1
    tmp_1 = operator.iadd(tmp_0, in_2)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_add3_mean