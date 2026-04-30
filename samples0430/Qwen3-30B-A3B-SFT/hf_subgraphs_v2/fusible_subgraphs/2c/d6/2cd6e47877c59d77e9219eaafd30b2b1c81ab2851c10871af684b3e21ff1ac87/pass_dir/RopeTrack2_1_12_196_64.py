"""
Pass: Fuse k_rope track for shape (1, 12, 197, 64).
Covers: float32/9 and float16/2 eva02_base (S=196, H=12).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S'],
)
@triton.jit
def _rope2_1_12_196_64_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    S: tl.constexpr, K: tl.constexpr,
):
    head_idx = tl.program_id(0)
    s_idx    = tl.program_id(1)
    D        = K >> 1
    base    = head_idx * (S + 1) * K + s_idx * K
    d_off   = tl.arange(0, D)

    xe = tl.load(x_ptr + base + (2 * d_off)).to(tl.float32)
    xo = tl.load(x_ptr + base + (2 * d_off + 1)).to(tl.float32)
    ce = tl.load(cos_ptr + s_idx * K + (2 * d_off)).to(tl.float32)
    co = tl.load(cos_ptr + s_idx * K + (2 * d_off + 1)).to(tl.float32)
    se = tl.load(sin_ptr + s_idx * K + (2 * d_off)).to(tl.float32)
    so = tl.load(sin_ptr + s_idx * K + (2 * d_off + 1)).to(tl.float32)

    ye = xe * ce - xo * se
    yo = xo * co + xe * so

    tl.store(out_ptr + base + (2 * d_off),   ye.to(tl.float16), mask=tl.full((D,), True, dtype=tl.int1))
    tl.store(out_ptr + base + (2 * d_off + 1), yo.to(tl.float16), mask=tl.full((D,), True, dtype=tl.int1))


@torch.fx.wrap
def _rope2_1_12_196_64(in_0, in_4, in_6):
    H, S, K = 12, 196, 64
    out = torch.empty_like(in_6)
    _rope2_1_12_196_64_kernel[(H, S + 1)](in_4, in_0, in_0, out, S=S, K=K)
    return out


def pattern(in_0, in_4, in_6):
    tmp_13 = in_0.tensor_split(2, -1)
    tmp_14 = tmp_13[0]
    tmp_15 = tmp_13[1]
    tmp_12 = in_4[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[(Ellipsis, slice(1, None, 2))]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[(Ellipsis, slice(None, None, 2))]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, 12, 196, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_13[0], tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(in_6)
    return tmp_25


def replacement_args(in_0, in_4, in_6):
    return (in_0, in_4, in_6)


def replacement_func():
    return _rope2_1_12_196_64