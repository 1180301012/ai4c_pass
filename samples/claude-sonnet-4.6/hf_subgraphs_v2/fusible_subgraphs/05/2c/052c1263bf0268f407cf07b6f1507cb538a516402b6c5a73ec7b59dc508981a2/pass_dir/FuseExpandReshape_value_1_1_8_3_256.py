"""
Fuses the expand+reshape pattern from [1,1,3,256] → [1,8,3,256]
into a single Triton kernel.

This pattern matches TWICE in the model:
  1) For tmp_6 (RoPE key result) → tmp_9
  2) For in_5 (value_states)     → tmp_12

Pattern matched:
  tmp_a = x[:, :, None, :, :]          # unsqueeze dim-2 → [1,1,1,3,256]
  tmp_b = tmp_a.expand(1, 1, 8, 3, 256) # broadcast       [1,1,8,3,256]
  tmp_c = tmp_b.reshape(1, 8, 3, 256)   # reshape         [1,8,3,256]

Returns: tmp_c
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# Flat 1-D loop over all N_DST = 1*8*3*256 = 6144 output elements.
# For element at flat index i in [1,8,3,256]:
#   src_idx = i % (N_SEQ * N_DIM)   maps to [1,1,3,256]
# ---------------------------------------------------------------------------
@triton.jit
def expand_flat_kernel(
    src_ptr,
    dst_ptr,
    N_SRC:      tl.constexpr,   # 768  = 1*3*256
    N_DST:      tl.constexpr,   # 6144 = 1*8*3*256
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_DST

    src_idx = offsets % N_SRC
    vals = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def expand_1_1_to_1_8_3_256(x):
    """
    x   : [1,1,3,256] bfloat16
    Returns: [1,8,3,256] bfloat16
    """
    N_SRC      = 768    # 1*3*256
    N_DST      = 6144   # 1*8*3*256
    BLOCK_SIZE = 1024

    out = torch.empty((1, 8, 3, 256), dtype=x.dtype, device=x.device)

    num_programs = (N_DST + BLOCK_SIZE - 1) // BLOCK_SIZE   # 6

    expand_flat_kernel[(num_programs,)](
        src_ptr    = x,
        dst_ptr    = out,
        N_SRC      = N_SRC,
        N_DST      = N_DST,
        BLOCK_SIZE = BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(x):
    tmp_a = x[:, :, None, :, :]
    tmp_b = tmp_a.expand(1, 1, 8, 3, 256)
    tmp_c = tmp_b.reshape(1, 8, 3, 256)
    return tmp_c


def replacement_args(x):
    return (x,)


def replacement_func():
    return expand_1_1_to_1_8_3_256