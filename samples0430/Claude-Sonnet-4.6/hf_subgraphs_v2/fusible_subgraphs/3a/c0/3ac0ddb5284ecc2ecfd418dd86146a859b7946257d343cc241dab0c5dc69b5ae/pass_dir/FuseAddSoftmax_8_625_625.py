import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _fused_add_softmax_625_625_kernel(
    in0_ptr,   # [1, 1, 625, 625]  — attention mask, broadcast over heads
    in1_ptr,   # [1, 8, 625, 625]  — attention logits
    out_ptr,   # [8, 625, 625]     — output
    BLOCK_SIZE: tl.constexpr,
):
    # Constants baked into this specialised kernel
    S:  tl.constexpr = 625
    S2: tl.constexpr = 625

    row_idx = tl.program_id(0)          # 0 .. B*S-1
    b = row_idx // S                    # head index
    s = row_idx  - b * S                # sequence index

    # in0 strides: [S*S2, S*S2, S2, 1] → element [0,0,s,j] = s*S2 + j
    in0_base = s * S2
    # in1 strides: [B*S*S2, S*S2, S2, 1] → element [0,b,s,j] = b*S*S2 + s*S2 + j
    in1_base = b * S * S2 + s * S2
    out_base = b * S * S2 + s * S2

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < S2

    # Load both inputs; masked lanes get -inf so they don't affect softmax
    x0 = tl.load(in0_ptr + in0_base + offsets, mask=mask, other=float('-inf'))
    x1 = tl.load(in1_ptr + in1_base + offsets, mask=mask, other=float('-inf'))

    # Promote to float32 for numerically-stable softmax
    # (for float32 inputs this is a no-op cast, harmless)
    x = x0.to(tl.float32) + x1.to(tl.float32)

    # Numerically-stable softmax
    x_max     = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp     = tl.exp(x_shifted)
    x_sum     = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum

    # Cast back to the original dtype before storing
    tl.store(out_ptr + out_base + offsets, x_softmax.to(x0.dtype), mask=mask)


@torch.fx.wrap
def fused_add_softmax_625_625(in_0, in_1):
    """
    Fused broadcast-add + softmax for shapes [1,1,625,625] + [1,8,625,625].
    Returns (out3d [8,625,625], out4d [1,8,625,625]) matching the original graph.
    """
    B, S, S2 = 8, 625, 625
    out3d = torch.empty((B, S, S2), dtype=in_1.dtype, device=in_1.device)

    _fused_add_softmax_625_625_kernel[(B * S,)](
        in_0, in_1, out3d,
    )

    out4d = out3d.view(1, B, S, S2)
    return (out3d, out4d)


# ---------------------------------------------------------------------------
# Pattern / replacement hooks used by the AI4C pass framework
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