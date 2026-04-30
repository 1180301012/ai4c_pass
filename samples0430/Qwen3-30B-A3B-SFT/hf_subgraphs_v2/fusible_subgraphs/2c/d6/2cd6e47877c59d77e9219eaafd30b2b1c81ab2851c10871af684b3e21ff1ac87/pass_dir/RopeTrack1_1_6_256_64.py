"""
Pass: Fuse q_rope track for shape (1, 6, 256, 64).
Covers: float16/9 eva02_small (S=256, H=6).
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
def _rope_track1_kernel(
    x_ptr,      # [1, H, S, K]  -- x (q_input)
    cos_ptr,    # [S, K]         -- cos embeddings
    sin_ptr,    # [S, K]         -- sin embeddings
    cls_ptr,    # [1, H, 1, K]   -- CLS token embeddings (q_prefix)
    out_ptr,    # [1, H, S, K]   -- output
    S: tl.constexpr,   # sequence length (e.g. 256)
    K: tl.constexpr,   # head dim (64)
):
    """
    For each (head, s):
      out[head, s, :] = x[head, s, :] * cos[s, :] + rotated(x[head, s, :]) * sin[s, :]
    For s=0, the CLS token from cls_ptr is used directly.
    """
    head_idx = tl.program_id(0)
    s_idx    = tl.program_id(1)
    D        = K >> 1   # 32 pairs

    # Base offset for x and output
    base    = head_idx * S * K + s_idx * K
    d_off   = tl.arange(0, D)   # 0..31

    # Load x even/odd elements
    xe = tl.load(x_ptr + base + (2 * d_off)).to(tl.float32)
    xo = tl.load(x_ptr + base + (2 * d_off + 1)).to(tl.float32)

    # Load cos/sin (same structure: [S, K])
    ce = tl.load(cos_ptr + s_idx * K + (2 * d_off)).to(tl.float32)
    co = tl.load(cos_ptr + s_idx * K + (2 * d_off + 1)).to(tl.float32)
    se = tl.load(sin_ptr + s_idx * K + (2 * d_off)).to(tl.float32)
    so = tl.load(sin_ptr + s_idx * K + (2 * d_off + 1)).to(tl.float32)

    # RoPE: [xe, xo] -> [xe*ce - xo*se, xo*co + xe*so]
    ye = xe * ce - xo * se
    yo = xo * co + xe * so

    # Load/store CLS for s=0
    cls_base = head_idx * K
    cxe = tl.load(cls_ptr + cls_base + (2 * d_off)).to(tl.float32)
    cxo = tl.load(cls_ptr + cls_base + (2 * d_off + 1)).to(tl.float32)
    cye = cxe * ce - cxo * se
    cyo = cxo * co + cxe * so

    out_mask  = tl.full((D,), True, dtype=tl.int1)
    out_ven   = tl.where(s_idx == 0, cye.to(tl.float16), ye)
    out_odd   = tl.where(s_idx == 0, cyo.to(tl.float16), yo)

    tl.store(out_ptr + base + (2 * d_off),   out_ven,   mask=out_mask)
    tl.store(out_ptr + base + (2 * d_off + 1), out_odd, mask=out_mask)


@torch.fx.wrap
def _rope_track1_1_6_256_64(in_1, in_2, in_3, in_5, in_6):
    H, S, K = 6, 256, 64
    out = torch.empty_like(in_6)    # [1, 6, 256, 64] same dtype
    _rope_track1_kernel[(H, S)](
        in_3, in_1, in_5, in_2, out,
        S=S, K=K,
    )
    return out


def pattern(in_1, in_2, in_3, in_5, in_6):
    tmp_1  = in_3 * in_1
    tmp_2  = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3  = -tmp_2
    tmp_4  = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5  = torch.stack([tmp_3, tmp_4], -1)
    tmp_6  = tmp_5.reshape((1, 6, 256, 64))
    tmp_7  = tmp_6 * in_5
    tmp_8  = tmp_1 + tmp_7
    tmp_9  = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    return tmp_10


def replacement_args(in_1, in_2, in_3, in_5, in_6):
    return (in_1, in_2, in_3, in_5, in_6)


def replacement_func():
    return _rope_track1_1_6_256_64