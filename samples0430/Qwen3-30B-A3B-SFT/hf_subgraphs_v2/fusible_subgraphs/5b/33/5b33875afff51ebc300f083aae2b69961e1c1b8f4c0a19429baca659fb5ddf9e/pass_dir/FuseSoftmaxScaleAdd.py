import torch
import triton
import triton.language as tl


# ─── Pattern: div + add (both use Python-level operators) ────────────────────
# These map to operator.truediv and operator.add in the FX graph.
# The softmax node remains in the graph after replacement.
def pattern(in_0, in_2):
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    return tmp_1


def replacement_args(in_0, in_2):
    return (in_0, in_2)


# ─── Triton kernel: fused (in_0 / scale) + broadcast mask ───────────────────
@triton.jit
def _scale_add_kernel(
    in0_ptr,   # [B, H, S, S]
    in2_ptr,   # [1, 1, 1, S]
    out_ptr,   # [B, H, S, S]
    H, S,
    SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid % S
    bh    = pid // S

    in_row  = in0_ptr + bh * S * S + q_idx * S
    in2_row = in2_ptr + q_idx   # in2 is [S]
    out_row = out_ptr + bh * S * S + q_idx * S

    cols = tl.arange(0, BLOCK_S)
    mask = cols < S

    x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = x / SCALE

    bias = tl.load(in2_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + bias

    tl.store(out_row + cols, x, mask=mask)


@torch.fx.wrap
def fused_scale_add(in_0, in_2):
    B, H, S, _ = in_0.shape
    out = torch.empty_like(in_0)
    total_rows = B * H * S
    BLOCK_S = 64  # covers S <= 45
    _scale_add_kernel[(total_rows,)](
        in_0, in_2, out,
        H, S,
        SCALE=8.0,
        BLOCK_S=BLOCK_S,
    )
    return out


def replacement_func():
    return fused_scale_add