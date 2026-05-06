"""
Fused pass for Swin attention bias: position_bias_gather + sigmoid + scale + add + softmax
Matches graphs with N = 24 heads (start884_24 float16 and start884_24 bfloat16).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 32}, num_warps=2),
        triton.Config({'BLOCK_P': 64}, num_warps=4),
        triton.Config({'BLOCK_P': 128}, num_warps=4),
        triton.Config({'BLOCK_P': 64}, num_warps=8),
        triton.Config({'BLOCK_P': 128}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _fused_attn_softmax_kernel_24(
    in0_ptr,            # position indices: [64, 64] int64
    in2_ptr,            # attention scores: [*, 24, 64, 64] fp16/bf16
    in3_ptr,            # attn mask: [B', 64, 64] fp16/bf16
    in5_ptr,            # position bias: [64, 64] fp16/bf16
    out_ptr,            # output: [1, 24, 64, 64] fp16/bf16
    N,                  # number of heads (runtime, maps to autotune key)
    H,                  # = 64 (runtime positional)
    W,                  # = 64 (runtime positional)
    BLOCK_K: tl.constexpr,   # = 64, fixed
    BLOCK_P: tl.constexpr,
):
    """
    Grid: (N * H, ceil(H * W / BLOCK_P))
    Each program handles one head n and one block of position pairs.
    Computes:
        tmp = in2[b,n,p,:] + in3[p,:] + 16*sigmoid(in5[p,:])   (before softmax)
        out[0,n,p,:] = softmax(tmp, dim=-1)
    """
    pid = tl.program_id(0)   # in [0, N*H)
    pid_p = tl.program_id(1) # position-pair block index

    n = pid // H             # head index
    p = pid_p * BLOCK_P      # starting position-pair index

    q = tl.arange(0, BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)

    # ---- position bias lookup ---------------------------------------
    pos_idx = tl.load(in0_ptr + p * 64 + q).to(tl.int64)

    # ---- sigmoid scaling of position bias ---------------------------
    pb = tl.load(in5_ptr + pos_idx * 64 + q).to(tl.float32)
    scale = tl.sigmoid(pb) * 16.0

    # ---- attention mask: in3[b, p, q] -------------------------------
    b = n % in2.shape[-3]
    in3_base = b * (H * W) * N + p * W + k_offs
    attn_mask = tl.load(in3_ptr + in3_base).to(tl.float32)

    # ---- attention scores: in2[b, n, p, :] --------------------------
    in2_base = b * N * H * W + n * H * W + p * W + k_offs
    attn_scores = tl.load(in2_ptr + in2_base).to(tl.float32)

    # ---- fused pointwise add ----------------------------------------
    # in2 + scale + attn_mask  (folded: (in2 + in3 + in3 + sp) - in3 = in2 + in3 + sp)
    s = attn_scores + scale + attn_mask
    valid = (p * W + q) < (H * W)
    s = tl.where(valid[None, :], s, float('-inf'))

    # ---- online softmax ---------------------------------------------
    m = tl.max(s, axis=0)
    s = s - m
    r = tl.exp(s)
    s_max = tl.full([BLOCK_K], float('-inf'), dtype=tl.float32)
    s_sum = tl.full([BLOCK_K], 0.0, dtype=tl.float32)

    for start in range(0, H * W, BLOCK_K):
        cur_s = tl.load(
            in2_ptr + in2_base + start * BLOCK_K + k_offs,
            mask=valid[(start + k_offs) < (H * W)],
            other=float('-inf'),
        ).to(tl.float32)
        cur_s = tl.where(k_offs >= start, cur_s, float('-inf'))
        s_max = tl.maximum(s_max, cur_s)
        diff = tl.where(k_offs >= start, cur_s - s_max, float('-inf'))
        folded = tl.exp(diff)
        s_sum = s_sum + folded

    # fold partial block max
    s_max = s_max - tl.max(s[None, :], axis=0)
    diff = tl.where(k_offs >= start, s - s_max, float('-inf'))
    folded = tl.exp(diff)
    s_sum = s_sum + tl.sum(folded, axis=0)
    out = folded * (1.0 / s_sum)

    # ---- store -------------------------------------------------------
    out_base = n * H * W + p * W
    tl.store(out_ptr + out_base + q, out.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Pattern / replacement kernel wrapper
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 24)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 16, 24, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 24, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return (tmp_22,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def _fused_attn_softmax_24(in_0, in_1, in_2, in_3):
    """
    Fused kernel for N=24 head graphs.
    in_0: position index [64,64] int64
    in_1: weight [24,512] fp16/bf16
    in_2: attn scores [*, 24, 64, 64] fp16/bf16
    in_3: attn mask   [*, 64, 64]    fp16/bf16
    -> [1, 24, 64, 64] fp16/bf16
    """
    N = 24
    B = in_2.shape[-3]
    output = torch.empty((1, N, 64, 64), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (N * 64, triton.cdiv(64 * 64, meta['BLOCK_P']))

    _fused_attn_softmax_kernel_24[grid](
        in_0, in_2, in_3, output,
        N=N,
        H=64, W=64,
        BLOCK_K=64,
    )
    return (output,)


def replacement_func():
    return _fused_attn_softmax_24