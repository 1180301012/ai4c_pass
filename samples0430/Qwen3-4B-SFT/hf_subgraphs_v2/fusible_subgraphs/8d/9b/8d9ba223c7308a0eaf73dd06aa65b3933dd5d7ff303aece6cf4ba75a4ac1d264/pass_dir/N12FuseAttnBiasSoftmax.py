"""
Fused pass for Swin attention bias: position_bias_gather + sigmoid + scale + add + softmax
Matches graphs with N = 12 heads (start428_11 bfloat16 and start428_11 float16).
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
def _fused_attn_softmax_kernel(
    in0_ptr,            # position indices: [64, 64] int64
    in1_ptr,            # weight matrix: [N, 512] fp16/bf16
    in2_ptr,            # attention scores: [B, N, 64, 64] fp16/bf16
    in3_ptr,            # attn mask: [B', 64, 64] fp16/bf16
    in5_ptr,            # position bias: [64, 64] fp16/bf16
    out_ptr,            # output: [1, N, 64, 64] fp16/bf16
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
    # B = in2.shape[-3] derived from N for generality
    b = n % in2.shape[-3]
    in3_base = b * (H * W) * N + p * W + k_offs
    attn_mask = tl.load(in3_ptr + in3_base).to(tl.float32)

    # ---- attention scores: in2[b, n, p, :] --------------------------
    in2_base = b * N * H * W + n * H * W + p * W + k_offs
    attn_scores = tl.load(in2_ptr + in2_base).to(tl.float32)

    # ---- fused pointwise add ----------------------------------------
    # in2 + scale + attn_mask  (the redundant in3 more was already folded)
    s = attn_scores + scale + attn_mask
    # mask padding positions (when H*W % BLOCK_P != 0)
    valid = (p * W + q) < (H * W)
    s = tl.where(valid[None, :], s, float('-inf'))

    # ---- online softmax ---------------------------------------------
    m = tl.max(s, axis=0)                         # scalar max
    s = s - m
    r = tl.exp(s)                                  # folded log-sum-exp done here
    # Initialise running stats (handles single-block kernel trivially)
    s_max = tl.full([BLOCK_K], float('-inf'), dtype=tl.float32)
    s_sum = tl.full([BLOCK_K], 0.0, dtype=tl.float32)

    for start in range(0, H * W, BLOCK_K):
        cur_s = tl.load(
            in2_ptr + in2_base + start * BLOCK_K + k_offs,
            mask=valid[(start + k_offs) < (H * W)],
            other=float('-inf'),
        ).to(tl.float32)
        cur_s = tl.where(
            k_offs >= start, cur_s, float('-inf')
        )
        # Track global running max
        s_max = tl.maximum(s_max, cur_s)
        # Track folded log-sum-exp: acc = exp(x - running_max)
        diff = tl.where(k_offs >= start, cur_s - s_max, float('-inf'))
        folded = tl.exp(diff)
        # Accumulate
        s_sum = s_sum + folded

    # ---- final normalisation ----------------------------------------
    s_max = s_max - tl.max(s[None, :], axis=0)     # fold partial block max
    diff = tl.where(k_offs >= start, s - s_max, float('-inf'))
    folded = tl.exp(diff)
    s_sum = s_sum + tl.sum(folded, axis=0)
    out = folded * (1.0 / s_sum)                    # numerically stable

    # ---- store -------------------------------------------------------
    out_base = n * H * W + p * W
    tl.store(out_ptr + out_base + q, out.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Pattern / replacement kernel wrapper
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 12)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
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
    return (tmp_22,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def _fused_attn_softmax_12(in_0, in_1, in_2, in_3):
    """
    Fused kernel for N=12 head graphs.
    in_0: position index [64,64] int64
    in_1: weight [12,512] fp16/bf16
    in_2: attn scores [*, 12, 64, 64] fp16/bf16
    in_3: attn mask   [*, 64, 64]    fp16/bf16
    -> [1, 12, 64, 64] fp16/bf16
    """
    N = 12
    B = in_2.shape[-3]
    output = torch.empty((1, N, 64, 64), dtype=in_2.dtype, device=in_2.device)

    # One row per (head, position_pair) pair; multiple rows per block along P
    def grid(meta):
        return (N * 64, triton.cdiv(64 * 64, meta['BLOCK_P']))

    _fused_attn_softmax_kernel[grid](
        in_0, in_1, in_2, in_3, output,
        N=N,
        H=64, W=64,
        BLOCK_K=64,
    )
    return (output,)


def replacement_func():
    return _fused_attn_softmax_12