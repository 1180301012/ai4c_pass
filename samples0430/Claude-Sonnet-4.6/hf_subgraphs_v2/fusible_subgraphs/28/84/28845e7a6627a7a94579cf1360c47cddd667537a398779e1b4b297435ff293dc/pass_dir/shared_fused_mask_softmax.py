"""Shared Triton kernel + dispatch wrapper used by all FuseMaskSoftmax passes.

Pattern: add + clamp → view(H,S,S)    [call_method anchor → avoids arg-normalization issue]
The downstream softmax stays in the graph and operates on the float32 output.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}),
        triton.Config({'BLOCK_SIZE': 32}),
    ],
    key=['S'],
)
@triton.jit
def _fused_add_clamp_kernel(
    in0_ptr, in1_ptr, out_ptr,
    H, S,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per output row (head * seq_pos).
    Computes: out[row] = clamp(in1[row] + in0[broadcast_row], NEG_INF)
    Output dtype: float32  (matches type-promotion from max(float16, float32_scalar))
    """
    row_idx  = tl.program_id(0)
    head_idx = row_idx // S
    seq_idx  = row_idx %  S

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < S

    in1 = tl.load(in1_ptr + head_idx * S * S + seq_idx * S + cols,
                  mask=mask, other=0.0).to(tl.float32)
    in0 = tl.load(in0_ptr + seq_idx * S + cols,
                  mask=mask, other=0.0).to(tl.float32)

    x = in1 + in0
    NEG_INF = -3.4028234663852886e+38
    x = tl.maximum(x, NEG_INF)

    tl.store(out_ptr + row_idx * S + cols, x, mask=mask)


@torch.fx.wrap
def fused_mask_softmax_dispatch(in0, in1, route):
    """Shared dispatch for all FuseMaskSoftmax passes.
    Returns the add+clamp result reshaped to (H, S, S) in float32.
    The downstream softmax in the graph operates on this tensor.
    """
    if route == "16_13_13":
        H, S = 16, 13
    elif route == "12_9_9":
        H, S = 12, 9
    else:                    # "16_9_9"
        H, S = 16, 9

    BLOCK_SIZE = 16
    out = torch.empty((H, S, S), dtype=torch.float32, device=in0.device)
    _fused_add_clamp_kernel[(H * S,)](
        in0, in1, out,
        H, S,
    )
    return out