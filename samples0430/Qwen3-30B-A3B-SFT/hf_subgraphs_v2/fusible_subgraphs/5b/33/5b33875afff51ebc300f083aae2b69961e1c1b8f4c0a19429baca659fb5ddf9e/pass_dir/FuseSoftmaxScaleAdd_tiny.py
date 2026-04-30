import torch
import triton
import triton.language as tl


# ─── Pattern: matches scale + add + softmax for tiny model (scale = 2.828...) ─
def pattern(in_0, in_2):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    return tmp_2


def replacement_args(in_0, in_2):
    return (in_0, in_2)


# ─── Triton kernel: scale + add broadcast + numerically-stable softmax ──────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 16}, num_warps=2),
        triton.Config({"BLOCK_S": 32}, num_warps=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128}, num_warps=4),
    ],
    key=["S"],
)
@triton.jit
def _fused_scale_add_softmax_tiny_kernel(
    in0_ptr,   # [B, H, S, S]  attention logits
    in2_ptr,   # [1, 1, 1, S]  attention mask
    out_ptr,   # [B, H, S, S]
    H, S,
    scale,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid % S
    bh    = pid // S

    in0_row = in0_ptr + bh * S * S + q_idx * S
    out_row = out_ptr  + bh * S * S + q_idx * S

    cols = tl.arange(0, BLOCK_S)
    mask = cols < S

    x = tl.load(in0_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x * scale

    bias = tl.load(in2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + bias

    x_max  = tl.max(x, axis=0)
    x_exp  = tl.exp(x - x_max)
    x_sum  = tl.sum(x_exp, axis=0)
    x_soft = x_exp / x_sum

    tl.store(out_row + cols, x_soft, mask=mask)


@torch.fx.wrap
def fused_scale_add_softmax_tiny(in_0, in_2):
    B, H, S, _ = in_0.shape
    out = torch.empty_like(in_0)
    SCALE = 2.8284271247461903

    total_rows = B * H * S
    _fused_scale_add_softmax_tiny_kernel[(total_rows,)](
        in_0, in_2, out,
        H, S,
        SCALE,
    )
    return out


def replacement_func():
    return fused_scale_add_softmax_tiny