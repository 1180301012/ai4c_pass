import torch
import triton
import triton.language as tl


# Fused kernel: attention_scores + pos_bias + 2*mask → softmax
# Shapes: in2[64,12,64,64], pos_bias[12,64,64], mask[64,64,64]
# Each program handles one row (b, n, row) with 64 elements
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=1),
        triton.Config({'BLOCK_S': 64}, num_warps=2),
        triton.Config({'BLOCK_S': 64}, num_warps=4),
        triton.Config({'BLOCK_S': 64}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _fused_add_softmax_b64_n12_kernel(
    in2_ptr,        # [64, 12, 64, 64]
    pos_bias_ptr,   # [12, 64, 64]
    mask_ptr,       # [64, 64, 64]
    out_ptr,        # [64, 12, 64, 64]
    BLOCK_S: tl.constexpr,  # = 64
):
    # B=64, N=12, S=64
    # Grid dim: B*N*S = 64*12*64 = 49152
    pid = tl.program_id(0)

    # Decompose pid → (b, n, row)
    row = pid % 64
    tmp = pid // 64
    n = tmp % 12
    b = tmp // 12

    cols = tl.arange(0, BLOCK_S)

    # Strides (contiguous row-major):
    # in2:     stride = [12*64*64, 64*64, 64, 1] = [49152, 4096, 64, 1]
    # pos_bias: stride = [64*64, 64, 1] = [4096, 64, 1]
    # mask:    stride = [64*64, 64, 1] = [4096, 64, 1]

    in2_base = b * 49152 + n * 4096 + row * 64
    pb_base  = n * 4096 + row * 64
    m_base   = b * 4096 + row * 64

    # Load and accumulate additions
    x = tl.load(in2_ptr + in2_base + cols)
    x = x + tl.load(pos_bias_ptr + pb_base + cols)
    m = tl.load(mask_ptr + m_base + cols)
    x = x + m + m  # equivalent to 2 * mask

    # Numerically stable softmax computed in float32
    x_f32 = x.to(tl.float32)
    x_max = tl.max(x_f32, axis=0)
    x_exp = tl.exp(x_f32 - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_out = (x_exp / x_sum).to(x.dtype)

    # Store result
    tl.store(out_ptr + in2_base + cols, x_out)


@torch.fx.wrap
def fused_add_softmax_b64_n12(in2, pos_bias, mask):
    # in2:      [64, 12, 64, 64]
    # pos_bias: [12, 64, 64]
    # mask:     [64, 64, 64]
    B, N, S = 64, 12, 64
    out = torch.empty_like(in2)
    grid = (B * N * S,)  # 49152
    _fused_add_softmax_b64_n12_kernel[grid](
        in2, pos_bias, mask, out,
    )
    return out


# ── Pattern to match ──────────────────────────────────────────────────────────
# Matches the subgraph: pos_bias.unsqueeze(0) + in_2 + 2*mask → softmax
# in_2 shape:    [64, 12, 64, 64]
# tmp_10 shape:  [12, 64, 64]  (scaled sigmoid position bias)
# in_3 shape:    [64, 64, 64]  (attention mask)
def pattern(in_2, tmp_10, in_3):
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 12, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22


def replacement_args(in_2, tmp_10, in_3):
    return (in_2, tmp_10, in_3)


def replacement_func():
    return fused_add_softmax_b64_n12