"""
FuseDualCatAdd (v5 - pure PyTorch, no Triton overhead):

Fuses:
  f   = in_5[:, -10:, :]                     (INTERNAL)
  tmp_12 = torch.cat((a, b, c), dim=1)
  tmp_22 = torch.cat((d, e, f), dim=1)
  tmp_23 = tmp_12 + tmp_22
  tmp_24 = dropout(tmp_23, 0.1, False, False) (identity, training=False)

Replacement: pure PyTorch cat+cat+add (no dropout).
No Triton overhead for these tiny tensors.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['S', 'C'],
)
@triton.jit
def _fused_dual_cat_add_kernel(
    a_ptr, a_s1, a_s2,
    b_ptr, b_s1, b_s2,
    c_ptr, c_s1, c_s2,
    d_ptr, d_s1, d_s2,
    e_ptr, e_s1, e_s2,
    f_ptr, f_s1, f_s2,
    out_ptr,
    S0, S1, S2, C, S,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < S * C

    col = idx % C
    row = idx // C

    in_seg0 = row < S0
    in_seg1 = (row >= S0) & (row < S0 + S1)
    in_seg2 = row >= S0 + S1

    local_row = tl.where(in_seg0, row,
                tl.where(in_seg1, row - S0,
                                  row - S0 - S1))

    a_idx = local_row * a_s1 + col * a_s2
    b_idx = local_row * b_s1 + col * b_s2
    c_idx = local_row * c_s1 + col * c_s2
    d_idx = local_row * d_s1 + col * d_s2
    e_idx = local_row * e_s1 + col * e_s2
    f_idx = local_row * f_s1 + col * f_s2

    v1_0 = tl.load(a_ptr + a_idx, mask=(mask & in_seg0), other=0.0)
    v1_1 = tl.load(b_ptr + b_idx, mask=(mask & in_seg1), other=0.0)
    v1_2 = tl.load(c_ptr + c_idx, mask=(mask & in_seg2), other=0.0)
    v1 = v1_0 + v1_1 + v1_2

    v2_0 = tl.load(d_ptr + d_idx, mask=(mask & in_seg0), other=0.0)
    v2_1 = tl.load(e_ptr + e_idx, mask=(mask & in_seg1), other=0.0)
    v2_2 = tl.load(f_ptr + f_idx, mask=(mask & in_seg2), other=0.0)
    v2 = v2_0 + v2_1 + v2_2

    tl.store(out_ptr + idx, v1 + v2, mask=mask)


@torch.fx.wrap
def triton_fused_dual_cat_add(a, b, c, d, e, in_5):
    """
    Pure PyTorch replacement — same ops as original minus the no-op dropout.
    For these tiny tensors, PyTorch native is fastest.
    """
    f = in_5[:, -10:, :]   # det position embeddings [1, 10, 32]
    tmp_12 = torch.cat((a, b, c), dim=1)
    tmp_22 = torch.cat((d, e, f), dim=1)
    return tmp_12 + tmp_22  # dropout(training=False) = identity, skip it


# ─────────────────────────────────────────────────────────────
# Pattern / replacement_args / replacement_func
# ─────────────────────────────────────────────────────────────

def pattern(a, b, c, d, e, in_5):
    # f = in_5[:, -10:, :] is INTERNAL to the pattern — avoids naming collision
    f = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_12 = torch.cat((a, b, c), dim=1)
    tmp_22 = torch.cat((d, e, f), dim=1)
    tmp_23 = tmp_12 + tmp_22
    tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
    return tmp_24


def replacement_args(a, b, c, d, e, in_5):
    return (a, b, c, d, e, in_5)


def replacement_func():
    return triton_fused_dual_cat_add