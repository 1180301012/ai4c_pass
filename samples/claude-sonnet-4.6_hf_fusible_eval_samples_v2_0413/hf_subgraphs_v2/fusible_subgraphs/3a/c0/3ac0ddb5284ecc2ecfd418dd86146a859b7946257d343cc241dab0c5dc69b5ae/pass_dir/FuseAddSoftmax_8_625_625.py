"""
Fused pass: broadcast-add + softmax for shape (8, 625, 625).
Handles float32 graph.

Pattern:
  tmp_0 = in_1 + in_0                          # [1,8,625,625] + [1,1,625,625]
  tmp_1 = tmp_0.view(8, 625, 625)
  tmp_2 = softmax(tmp_1, dim=-1)
  tmp_3 = tmp_2.view(1, 8, 625, 625)
  tmp_4 = tmp_3.view(8, 625, 625)
  tmp_5 = dropout(tmp_4, p=0.0, training=False)  # identity
  return (tmp_5, tmp_3)
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: fused broadcast-add + numerically-stable row-softmax
# ---------------------------------------------------------------------------

@triton.jit
def _fused_add_softmax_625_625(
    in0_ptr,          # [1, 1, SEQ_LEN, WIDTH]  (broadcasts over heads)
    in1_ptr,          # [1, NUM_HEADS, SEQ_LEN, WIDTH]
    out_ptr,          # [NUM_HEADS, SEQ_LEN, WIDTH]  (output)
    seq_len,          # scalar: number of positions (625)
    width,            # scalar: row width (625)
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_W: tl.constexpr,   # must be >= width, must be power-of-2
):
    # 2-D grid: axis-0 = head index, axis-1 = position index
    head = tl.program_id(0)
    h    = tl.program_id(1)

    # Column offsets & mask  (handles non-power-of-2 widths)
    cols = tl.arange(0, BLOCK_W)
    mask = cols < width

    # ---- Load in_1[0, head, h, :] ----------------------------------------
    in1_row = tl.load(
        in1_ptr + (head * seq_len + h) * width + cols,
        mask=mask, other=float('-inf'),
    )

    # ---- Load in_0[0, 0, h, :] (broadcast over head dimension) ----------
    in0_row = tl.load(
        in0_ptr + h * width + cols,
        mask=mask, other=0.0,
    )

    # ---- Fused add, upcast to fp32 for stable softmax --------------------
    x = in1_row.to(tl.float32) + in0_row.to(tl.float32)

    # ---- Numerically-stable softmax over the row -------------------------
    x_max  = tl.max(x, axis=0)          # scalar max (padded slots are -inf → ignored)
    x      = x - x_max
    x_exp  = tl.exp(x)
    x_sum  = tl.sum(tl.where(mask, x_exp, 0.0), axis=0)
    x_soft = x_exp / x_sum

    # ---- Cast back to input dtype & store --------------------------------
    if IS_BF16:
        x_out = x_soft.to(tl.bfloat16)
    elif IS_FP16:
        x_out = x_soft.to(tl.float16)
    else:
        x_out = x_soft   # float32

    tl.store(
        out_ptr + (head * seq_len + h) * width + cols,
        x_out,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_softmax_625_625(in_0, in_1):
    """
    in_0 : [1, 1, 625, 625]
    in_1 : [1, 8, 625, 625]
    returns: (tmp_5 [8,625,625], tmp_3 [1,8,625,625])
    """
    NUM_HEADS = 8
    SEQ_LEN   = 625
    WIDTH     = 625
    BLOCK_W   = 1024   # next power-of-2 >= 625

    out = torch.empty(NUM_HEADS, SEQ_LEN, WIDTH,
                      dtype=in_1.dtype, device=in_1.device)

    is_bf16 = (in_1.dtype == torch.bfloat16)
    is_fp16 = (in_1.dtype == torch.float16)

    _fused_add_softmax_625_625[(NUM_HEADS, SEQ_LEN)](
        in_0, in_1, out,
        SEQ_LEN, WIDTH,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_W=BLOCK_W,
    )

    # Return two *views* of the same underlying buffer — identical semantics
    # to the original graph's view → view → dropout(p=0) chain.
    tmp_5 = out.view(8, 625, 625)
    tmp_3 = out.view(1, 8, 625, 625)
    return (tmp_5, tmp_3)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_softmax_625_625