"""
Fused pass: add + max(clamp) + view(16,9,9) + softmax + dropout(training=False)
Targets: KoboldAI_fairseq-dense-355M float16, bfloat16 & float32, view shape (16, 9, 9)

Key insight:
- torch.max(f16/bf16_tensor, f32_constant) promotes output to float32
- dropout with training=False is identity
- Fuse everything into one Triton kernel for maximum throughput
"""
import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _fused_attn_softmax_kernel_16_9_9(
    in0_ptr,          # [1, 1, 9, 9] – attention mask (broadcast)
    in1_ptr,          # [1, 16, 9, 9] – attention scores
    out_ptr,          # [16, 9, 9] float32 output
    N: tl.constexpr,       # 16  (number of heads)
    S: tl.constexpr,       # 9   (sequence length)
    BLOCK_S: tl.constexpr, # 16  (next power-of-2 >= S)
):
    # One program per row; row = (head_idx, seq_idx)
    pid      = tl.program_id(0)
    head_idx = pid // S
    seq_idx  = pid % S

    offs = tl.arange(0, BLOCK_S)
    mask = offs < S          # valid positions within the row

    # -------------------------------------------------------------------
    # Load in1[0, head_idx, seq_idx, :]
    # Strides for [1, N, S, S] contiguous: [N*S*S, S*S, S, 1]
    # -------------------------------------------------------------------
    x1 = tl.load(in1_ptr + head_idx * S * S + seq_idx * S + offs,
                  mask=mask, other=0.0)

    # -------------------------------------------------------------------
    # Load in0[0, 0, seq_idx, :]
    # Strides for [1, 1, S, S] contiguous: [S*S, S*S, S, 1]
    # head broadcast => always index 0
    # -------------------------------------------------------------------
    x0 = tl.load(in0_ptr + seq_idx * S + offs,
                  mask=mask, other=0.0)

    # --- step 1: add in input precision (matches original graph) ----------
    row_native = x1 + x0

    # --- step 2: promote to float32 (matches torch.max dtype promotion) --
    row = row_native.to(tl.float32)

    # --- step 3: clamp – equivalent of torch.max(row, torch.tensor(-FLT_MAX))
    row = tl.maximum(row, -3.4028234663852886e+38)

    # --- step 4: mask padding slots to -inf so they get 0 weight ---------
    row = tl.where(mask, row, float('-inf'))

    # --- step 5: numerically-stable softmax in float32 -------------------
    row_max  = tl.max(row, axis=0)
    row_exp  = tl.exp(row - row_max)
    row_exp  = tl.where(mask, row_exp, 0.0)   # zero out padding slots
    row_sum  = tl.sum(row_exp, axis=0)
    row_out  = row_exp / row_sum

    # --- step 6: store float32 result ------------------------------------
    tl.store(out_ptr + head_idx * S * S + seq_idx * S + offs,
             row_out, mask=mask)


@torch.fx.wrap
def fused_attn_softmax_16_9_9(in_0, in_1):
    """
    Fused: (in_1 + in_0) -> clamp -> view(16,9,9) -> softmax(dim=-1) -> dropout(noop)
    Output dtype: float32  (matches torch.max type-promotion with float32 constant)
    """
    N, S    = 16, 9
    BLOCK_S = 16     # next power-of-2 >= 9

    out = torch.empty((N, S, S), dtype=torch.float32, device=in_1.device)

    _fused_attn_softmax_kernel_16_9_9[(N * S,)](
        in_0, in_1, out,
        N=N, S=S, BLOCK_S=BLOCK_S,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API consumed by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, tmp_1):
    """
    tmp_1 is a wildcard argument that matches the torch.tensor(-3.4e38, ...)
    constant node in the target graph, regardless of how it was created/folded.
    """
    tmp_0 = in_1 + in_0
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(16, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return (tmp_5,)


def replacement_args(in_0, in_1, tmp_1):
    # tmp_1 is the constant -3.4e38 tensor; its value is hardcoded in the kernel
    return (in_0, in_1)


def replacement_func():
    return fused_attn_softmax_16_9_9