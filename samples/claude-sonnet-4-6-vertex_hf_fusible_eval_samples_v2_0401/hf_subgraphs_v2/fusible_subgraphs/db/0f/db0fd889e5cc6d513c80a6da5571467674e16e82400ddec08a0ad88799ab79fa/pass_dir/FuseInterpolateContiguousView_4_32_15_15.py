"""
Pass: FuseInterpolateContiguousView_4_32_15_15

Fuses the chain (in_6 position-embedding path), starting from tmp_30:
  tmp_31 = interpolate(tmp_30, (15,15), bicubic, False)    # [4,32,15,15] -> [4,32,15,15] NO-OP!
  tmp_32 = tmp_31.flatten(2)                               # -> [4, 32, 225]
  tmp_33 = tmp_32.transpose(1, 2)                          # -> [4, 225, 32]
  tmp_34 = tmp_33.contiguous()                             # -> [4, 225, 32]
  tmp_35 = tmp_34.view(4, 1, 225, 32)                      # -> [4, 1, 225, 32]

Since the input is already (15×15), bicubic interpolation is identity.
We replace the whole chain with a Triton transpose kernel:
  [4, 32, 15, 15] (contiguous) → [4, 1, 225, 32] (contiguous)
by doing output[b,0,s,c] = input[b,c,s//15,s%15].

This pass runs FIRST so that the shorter pattern in
FuseInterpolateIdentity_1_32_15_15 does not accidentally match in_6's chain.
"""

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: [B, C, H, W] contiguous → [B, 1, H*W, C] contiguous
# B=4, C=32, H=W=15  → 28 800 elements total
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['B', 'C', 'S'],
)
@triton.jit
def _transpose_bchw_to_b1sc_kernel(
    input_ptr,
    output_ptr,
    B,
    C,
    S,          # H * W
    BLOCK_SIZE: tl.constexpr,
):
    """
    input  layout: [B, C, S] contiguous  (HW flattened)
      linear idx = b*(C*S) + c*S + s
    output layout: [B, S, C] contiguous  (= [B, 1, S, C] viewed as [B,S,C])
      linear idx = b*(S*C) + s*C + c

    output[b, s, c] = input[b, c, s]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * C * S
    mask = offsets < total

    # Decompose OUTPUT linear index: [B, S, C]
    out_c = offsets % C
    tmp   = offsets // C
    out_s = tmp % S
    out_b = tmp // S

    # Map to INPUT linear index: [B, C, S]
    in_idx = out_b * (C * S) + out_c * S + out_s

    x = tl.load(input_ptr + in_idx, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fuse_interpolate_contiguous_view_4_32_15_15(tmp_30):
    """
    tmp_30: [4, 32, 15, 15] – may be non-contiguous if it came from
            a view on a transposed tensor; we call .contiguous() first.
    Returns: [4, 1, 225, 32] fully contiguous.
    """
    B = 4
    C = 32
    S = 15 * 15   # 225
    N = B * C * S  # 28 800

    x = tmp_30.contiguous()                    # [4, 32, 15, 15] contiguous
    inp = x.view(B, C, S)                      # [4, 32, 225]
    out = torch.empty((B, S, C), dtype=x.dtype, device=x.device)  # [4, 225, 32]

    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _transpose_bchw_to_b1sc_kernel[grid](
        input_ptr=inp,
        output_ptr=out,
        B=B,
        C=C,
        S=S,
    )
    # Reshape [4, 225, 32] → [4, 1, 225, 32]
    return out.view(B, 1, S, C)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement_args / replacement_func
# ──────────────────────────────────────────────────────────────────────────────

def pattern(tmp_30):
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(tmp_30):
    return (tmp_30,)


def replacement_func():
    return fuse_interpolate_contiguous_view_4_32_15_15