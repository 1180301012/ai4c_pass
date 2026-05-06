"""
Pass: FuseQRoPE_cl16_s256_d64
Matches the QRoPE+cat+type_as subgraph where reshape shape is (1, 16, 256, 64).
Target: eva02_large_patch14_224 graphs (H=16, S=256, D=64).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel_q16s256(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    H,
    S,
    BLOCK_D: tl.constexpr,
):
    h       = tl.program_id(0)
    s       = tl.program_id(1)
    d_start = tl.program_id(2) * BLOCK_D

    d_offs   = tl.arange(0, BLOCK_D)
    d_global = d_start + d_offs

    cos_val = tl.load(cos_ptr + s * BLOCK_D + d_offs)
    sin_val = tl.load(sin_ptr + s * BLOCK_D + d_offs)
    x       = tl.load(x_ptr   + h * S * BLOCK_D + s * BLOCK_D + d_global)

    x_even = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + d_global,
                     mask=d_global % 2 == 0, other=0.0)
    x_odd  = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + d_global,
                     mask=d_global % 2 == 1, other=0.0)

    rope_q1 = x_even * sin_val + x_odd * cos_val
    rope_q2 = x_odd  * cos_val - x_even * sin_val

    output_base = h * S * BLOCK_D + s * BLOCK_D
    tl.store(out_q_ptr + output_base + tl.arange(0, BLOCK_D // 2),
             rope_q1, mask=tl.arange(0, BLOCK_D // 2) < BLOCK_D // 2)
    tl.store(out_q_ptr + output_base + BLOCK_D // 2 + tl.arange(0, BLOCK_D // 2),
             rope_q2, mask=tl.arange(0, BLOCK_D // 2) < BLOCK_D // 2)


def pattern(in_2, in_3, in_5, in_6):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 16, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    return tmp_10


def replacement_args(in_2, in_3, in_5, in_6):
    return (in_2, in_3, in_5, in_6)


@torch.fx.wrap
def _qr_t16s256(x, cos, sin, cls_tk, ref):
    H = x.shape[1]
    S = x.shape[2]
    D = x.shape[3]
    rope_out = torch.empty((1, H, S, D), dtype=ref.dtype, device=ref.device)
    out      = torch.empty((1, H, S + 1, D), dtype=ref.dtype, device=ref.device)
    x_flat   = x.view(H, S, D)
    out_flat = out.view(H, S + 1, D)
    grid = (H, S, triton.cdiv(D, 64))
    _rope_kernel_q16s256[grid](x_flat, cos, sin, rope_out, H, S, BLOCK_D=64)
    out[:, :, 0, :] = cls_tk.view(H, D)
    out[:, :, 1:, :] = rope_out.view(H, S, D)
    return out


def replacement_func():
    return _qr_t16s256