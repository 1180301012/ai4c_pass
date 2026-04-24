"""
Fuse the bicubic-interpolation chain in path 2 of the YOLOS model.

Pattern inputs: in_6  [4, 1, 236, 32]
Pattern outputs: tmp_35  [4, 1, 225, 32]

Since bicubic with scale=1 and align_corners=False is an identity, the whole
chain reduces to a simple permutation:
  output[b, 0, d, c] = in_6[b, 0, d//15, d%15, c]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _bicubic_permute_kernel(
    in_ptr,   # [4, 1, 236, 32]
    out_ptr,  # [4, 1, 225, 32]
    D,        # 225
    W_OUT,    # 15
    C,        # 32
    BLOCK_C: tl.constexpr,
):
    """
    Grid: (B * D,) = (4 * 225,) = 900 programs.
    Each program handles one (b, d) row and all C=32 channels.

    Read  : in_ptr  + b*236*C + (d//W_OUT)*W_OUT*C + (d%W_OUT)*C + c  (contiguous in c)
    Write : out_ptr + b*D*C       + d*C                  + c            (contiguous in c)
    """
    pid = tl.program_id(0)
    b   = pid // D
    d   = pid  % D

    c_offs = tl.arange(0, BLOCK_C)

    h = d // W_OUT
    w = d  % W_OUT
    row_offset = h * W_OUT * C + w * C

    vals = tl.load(in_ptr  + b * (236 * C) + row_offset + c_offs)
    tl.store(out_ptr + b * (D * C)         + d * C        + c_offs, vals)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def bicubic_path_permute(in_6):
    B = in_6.shape[0]
    C = in_6.shape[3]
    D = 225
    W_OUT = 15

    out = torch.empty((B, 1, D, C), dtype=in_6.dtype, device=in_6.device)
    grid = (B * D,)
    _bicubic_permute_kernel[grid](
        in_6, out,
        D, W_OUT, C,
        BLOCK_C=C,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_6):
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(in_6):
    return (in_6,)


def replacement_func():
    return bicubic_path_permute