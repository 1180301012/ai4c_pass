"""
Fuses the Rotary Position Embedding (RoPE) computation on key_states
into a single Triton kernel.

Pattern matches:
  tmp_0 = in_2 * in_1                        # key * cos  [1,1,3,256]
  tmp_1 = in_2[..., :128]                    # first half
  tmp_2 = in_2[..., 128:]                    # second half
  tmp_3 = -tmp_2                             # negate
  tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)  # rotated  [1,1,3,256]
  tmp_5 = tmp_4 * in_4                       # rotated * sin
  tmp_6 = tmp_0 + tmp_5                      # RoPE result [1,1,3,256]

Returns: tmp_6
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: flat 1-D approach over all N_ELEMENTS = 1*1*3*256 = 768
# ---------------------------------------------------------------------------
@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,   # 768
    N_HALF:     tl.constexpr,   # 128
    N_DIM:      tl.constexpr,   # 256
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # dimension index within the last axis
    d = offsets % N_DIM

    # Load x, cos, sin
    x  = tl.load(x_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(cos_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(sin_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)

    # Rotation partner: d<N_HALF → partner = offset+N_HALF; else offset-N_HALF
    d_shift  = tl.where(d < N_HALF, N_HALF, -N_HALF)
    x_p = tl.load(x_ptr + offsets + d_shift, mask=mask, other=0.0).to(tl.float32)
    x_rot = tl.where(d < N_HALF, -x_p, x_p)

    out = (x * c + x_rot * sn).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_rope(in_2, in_1, in_4):
    """
    in_2 : key_states [1,1,3,256] bfloat16
    in_1 : cos        [1,1,3,256] bfloat16
    in_4 : sin        [1,1,3,256] bfloat16
    Returns: [1,1,3,256] bfloat16
    """
    N_ELEMENTS = 768    # 1*1*3*256
    N_HALF     = 128
    N_DIM      = 256
    BLOCK_SIZE = 256

    out = torch.empty_like(in_2)

    num_programs = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE   # 3

    rope_kernel[(num_programs,)](
        x_ptr      = in_2,
        cos_ptr    = in_1,
        sin_ptr    = in_4,
        out_ptr    = out,
        N_ELEMENTS = N_ELEMENTS,
        N_HALF     = N_HALF,
        N_DIM      = N_DIM,
        BLOCK_SIZE = BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_2, in_1, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[..., :128]
    tmp_2 = in_2[..., 128:]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)


def replacement_func():
    return fused_rope