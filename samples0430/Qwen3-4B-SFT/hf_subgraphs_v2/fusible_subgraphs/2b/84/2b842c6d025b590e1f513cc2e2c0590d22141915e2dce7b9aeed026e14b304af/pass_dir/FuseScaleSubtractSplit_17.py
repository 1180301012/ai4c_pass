import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the exact computation in model.py
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused Triton kernel
#   in_0  : [B, L, 1]  int64   (logits mask)
#   in_1  : [B, L, 2]  float16/bfloat16
#   out0  : [B, L]     float16/bfloat16  (tmp_7 = first half squeezed)
#   out1  : [B, L]     float16/bfloat16  (tmp_9 = second half squeezed)
#
# Each program handles one row (b, l).
# Tensor layout (contiguous):
#   in_0  flat index: b*L + l
#   in_1  flat index: b*L + l   (hidden_states[b,l,0])
#                     b*L + l + L   (hidden_states[b,l,1])
#   out   flat index: b*L + l
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": 32}),
        triton.Config({"BLOCK_L": 64}),
    ],
    key=["L"],
)
@triton.jit
def fused_scale_subtract_split_kernel(
    a_ptr,   # in_0  [L]      (int64, viewed as flat)
    b_ptr,   # in_1  [L, 2]   (float16/bfloat16, viewed flat)
    out0_ptr,  # output [L]
    out1_ptr,  # output [L]
    L,        # number of tokens
    BLOCK_L: tl.constexpr,
):
    pid = tl.program_id(0)
    l_offsets = pid * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    # Load mask (int64) for this block
    a = tl.load(a_ptr + l_offsets, mask=mask, other=0)

    # Load both hidden-state elements (same mask applies to both channels)
    b0_offsets = l_offsets * 2   # element 0 of each pair
    b1_offsets = b0_offsets + L  # element 1 of each pair

    b0 = tl.load(b_ptr + b0_offsets, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + b1_offsets, mask=mask, other=0.0)

    # Scale mask and subtract (all arithmetic in the native dtype of in_1)
    # NOTE: promotion from int64 to the element dtype is handled transparently
    result0 = b0 - a * 1000000.0
    result1 = b1 - a * 1000000.0

    tl.store(out0_ptr + l_offsets, result0, mask=mask)
    tl.store(out1_ptr + l_offsets, result1, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_scale_subtract_split(hidden_states, mask):
    B, L, two = hidden_states.shape   # e.g. [1, 17, 2]

    out0 = torch.empty((B, L), dtype=hidden_states.dtype, device=hidden_states.device)
    out1 = torch.empty((B, L), dtype=hidden_states.dtype, device=hidden_states.device)

    num_rows = B * L          # 17 in the benchmark shapes
    BLOCK_L = 32              # covers 17 in one block

    fused_scale_subtract_split_kernel[(num_rows,)](
        mask,
        hidden_states,
        out0,
        out1,
        L,
        BLOCK_L=BLOCK_L,
    )

    return out0, out1


# ---------------------------------------------------------------------------
# Required by the evaluation framework
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_scale_subtract_split