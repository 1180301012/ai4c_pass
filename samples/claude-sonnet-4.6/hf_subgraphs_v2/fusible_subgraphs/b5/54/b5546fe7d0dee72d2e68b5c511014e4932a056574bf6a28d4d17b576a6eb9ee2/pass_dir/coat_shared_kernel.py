"""
Shared Triton kernels for CoAT factored-attention CRPE post-processing fusion.

Split into two passes to avoid F.pad matching issues:

Pass A: cat([in2, in3, conv]) → reshape → transpose(-1,-2) → in6 * transposed
  Input:  in2 [1,2C,H,W], in3 [1,3C,H,W], conv [1,3C,H,W], in6 [1,8,HW,C]
  Output: mul_result [1,8,HW,C]
  For each (head, s, c):
    d = head*C + c
    cat_val = in2/in3/conv[0, d, s]  (based on channel range)
    out[0, head, s, c] = in6[0, head, s, c] * cat_val

Pass B: scale*in4 + padded → transpose(1,2) → reshape
  Input:  padded [1,8,HW+1,C], in4 [1,8,HW+1,C]
  Output: result [1,HW+1,8*C]
  For each output pos (seq, d) where d = head*C + c:
    in4_idx = head*(HW+1)*C + seq*C + c
    out[0, seq, d] = scale * in4[in4_idx] + padded[in4_idx]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass A kernel: fuses cat + (implicit reshape/transpose) + elementwise mul
# ---------------------------------------------------------------------------
@triton.jit
def _coat_fuse_A_kernel(
    in2_ptr, in3_ptr, conv_ptr, in6_ptr, out_ptr,
    HW, C,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    output[0, head, s, c] = in6[0, head, s, c] * cat_val(d=head*C+c, s)
    output layout: [1, 8, HW, C]  →  index = head*HW*C + s*C + c
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # output is [8, HW, C]; decode (head, s, c) from flat index
    HC = HW * C
    head = offsets // HC
    rem  = offsets  % HC
    s    = rem // C
    c    = rem  % C
    d    = head * C + c      # channel index in the concatenated view

    # Load in6[0, head, s, c]  → stride: head*HW*C + s*C + c = offsets
    in6_val = tl.load(in6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # cat source: d in [0,2C) → in2; [2C,5C) → in3; [5C,8C) → conv
    C2 = 2 * C
    C5 = 5 * C

    in2_val  = tl.load(in2_ptr  + d            * HW + s,
                       mask=(mask & (d < C2)),            other=0.0).to(tl.float32)
    in3_val  = tl.load(in3_ptr  + (d - C2)     * HW + s,
                       mask=(mask & (d >= C2) & (d < C5)), other=0.0).to(tl.float32)
    conv_val = tl.load(conv_ptr + (d - C5)     * HW + s,
                       mask=(mask & (d >= C5)),            other=0.0).to(tl.float32)

    cat_val = tl.where(d < C2, in2_val,
              tl.where(d < C5, in3_val, conv_val))

    out_val = in6_val * cat_val
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def coat_fuse_A_dispatch(in_2, in_3, conv_out, in_6, route):
    """
    route: unused (kept for routing API compatibility)
    Computes: in6 * cat(in2, in3, conv_out) after reshape/transpose
    Output: [1, 8, HW, C]
    """
    device = in_6.device
    dtype  = in_6.dtype

    # in_6: [1, 8, HW, C]
    B, heads, HW, C = in_6.shape
    total_elements = heads * HW * C

    out = torch.empty_like(in_6)

    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _coat_fuse_A_kernel[grid](
        in_2.contiguous(), in_3.contiguous(), conv_out.contiguous(),
        in_6.contiguous(), out,
        HW, C,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pass B kernel: fuses scale*in4 + padded + transpose(1,2) + reshape
# ---------------------------------------------------------------------------
@triton.jit
def _coat_fuse_B_kernel(
    padded_ptr, in4_ptr, out_ptr,
    HW, C, scale,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Input layout (both padded and in4): [1, 8, HW+1, C]  head-major
      index: head*(HW+1)*C + seq*C + c
    Output layout: [1, HW+1, 8*C]  seq-major
      index: seq*(8*C) + head*C + c  = seq*D + d
    For output position (seq, d):
      head = d // C,  c = d % C
      input_idx = head*(HW+1)*C + seq*C + c
      out = scale * in4[input_idx] + padded[input_idx]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    D   = 8 * C
    SEQ = HW + 1

    # Decode output position
    seq  = offsets // D
    d    = offsets  % D
    head = d // C
    c    = d  % C

    # Input index (head-major layout)
    in_idx = head * SEQ * C + seq * C + c

    in4_val    = tl.load(in4_ptr    + in_idx, mask=mask, other=0.0).to(tl.float32)
    padded_val = tl.load(padded_ptr + in_idx, mask=mask, other=0.0).to(tl.float32)

    out_val = scale * in4_val + padded_val
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def coat_fuse_B_dispatch(padded, in_4, route):
    """
    route: string scale value, e.g. "0.22941573387056177"
    Computes: (scale * in4 + padded), then transpose(1,2) + reshape → [1, HW+1, 8*C]
    """
    scale = float(route)
    device = in_4.device
    dtype  = in_4.dtype

    # in_4: [1, 8, HW+1, C]
    B, heads, SEQ_plus_1, C = in_4.shape
    HW = SEQ_plus_1 - 1

    total_elements = SEQ_plus_1 * heads * C

    out = torch.empty((B, SEQ_plus_1, heads * C), dtype=dtype, device=device)

    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _coat_fuse_B_kernel[grid](
        padded.contiguous(), in_4.contiguous(), out,
        HW, C, scale,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out