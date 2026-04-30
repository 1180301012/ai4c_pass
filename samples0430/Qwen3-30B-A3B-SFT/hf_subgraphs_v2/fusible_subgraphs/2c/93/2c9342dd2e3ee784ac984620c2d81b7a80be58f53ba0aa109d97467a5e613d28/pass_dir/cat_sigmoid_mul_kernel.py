"""
Shared Triton kernel: fuses cat([in_3, in_4, conv_out], dim=2) +
                      sigmoid - 0.25 * pi  ->  single output tensor.

Input layout (contiguous):
  in_3  : [N, 1, S1]   S1 = 6400
  in_4  : [N, 1, S2]   S2 = 1600
  conv  : [N, 1, S3]   S3 =  400  (viewed conv2d output)
Output layout (contiguous):
  out   : [N, 1, S1+S2+S3] = [N, 1, 8400]

1-D flat grid. S1/S2/S3 tl.constexpr → strength-reduced div/mod by 8400.
N_total is runtime → stable Triton cache key, enables cross-graph caching.
"""

import torch
import triton
import triton.language as tl

_S1    = 6400
_S2    = 1600
_S3    = 400
_TOTAL = _S1 + _S2 + _S3   # 8400
_BLOCK = 1024
_NWARPS = 4


@triton.jit
def _cat_sig_sub_mul_kernel(
    in3_ptr,
    in4_ptr,
    conv_ptr,
    out_ptr,
    N_total,                    # runtime (stable Triton cache key)
    S1:       tl.constexpr,   # 6400
    S2:       tl.constexpr,   # 1600
    S3:       tl.constexpr,   #  400
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total

    total = S1 + S2 + S3
    batch = offsets // total
    seg   = offsets % total

    is_s0 = seg < S1
    is_s1 = (seg >= S1) & (seg < S1 + S2)

    c0 = tl.where(is_s0, seg, 0)
    c1 = tl.where(is_s1, seg - S1, 0)
    c2 = tl.where(seg >= S1 + S2, seg - S1 - S2, 0)

    x0 = tl.load(in3_ptr + batch * S1 + c0, mask=mask & is_s0,         other=0.0)
    x1 = tl.load(in4_ptr + batch * S2 + c1, mask=mask & is_s1,         other=0.0)
    x2 = tl.load(conv_ptr + batch * S3 + c2, mask=mask & ~is_s0 & ~is_s1, other=0.0)

    x = tl.where(is_s0, x0, tl.where(is_s1, x1, x2))

    x_f32  = x.to(tl.float32)
    result = (tl.sigmoid(x_f32) - 0.25) * 3.141592653589793

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_cat_sigmoid_multiply(in_3, in_4, conv_out):
    """
    Fused replacement for:
        tmp_4 = torch.cat([in_3, in_4, conv_out], 2)
        tmp_5 = tmp_4.sigmoid()
        tmp_6 = tmp_5 - 0.25
        tmp_7 = tmp_6 * 3.141592653589793
    """
    N = in_3.shape[0]
    N_total = N * _TOTAL

    out = torch.empty((N, 1, _TOTAL), dtype=in_3.dtype, device=in_3.device)

    _cat_sig_sub_mul_kernel[(triton.cdiv(N_total, _BLOCK),)](
        in_3, in_4, conv_out, out,
        N_total,
        S1=_S1, S2=_S2, S3=_S3,
        BLOCK_SIZE=_BLOCK,
        num_warps=_NWARPS,
    )

    return out