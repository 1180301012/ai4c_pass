import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: match the full relative-position-bias + softmax + attn@V + transpose
#   for 8-head variant with 64×64 attention (KV_HEADS=8, SEQ=8, N=64)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 8, 15)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 7], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 9, 15)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))]
    tmp_7 = tmp_6.reshape(4, 8, 1, 8, 8)
    tmp_8 = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 64, 64)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fuse relative-pos bias + add_abs + softmax + attn@V + transpose
#
#   in1  : [BATCH, H, S, HEAD_DIM]   (q embeddings, H=KV_HEADS)
#   in3  : [HEAD_DIM, 2*S-1]          (rel logits linear weights, S=8 → 15 cols)
#   in2  : [BATCH, H, S, H, S]        (rel logits, S=8)
#   in0  : [BATCH, N, N]              (abs logits, N=64)
#   in4  : [BATCH, N, D]              (values, N=64, D=128)
#   out  : [BATCH, D, N]              (transposed result, [4,128,64])
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'HEADS', 'SEQ', 'D_MODEL', 'KV_HEADS'],
)
@triton.jit
def _fused_kernel_8h_fp16(
    in1_ptr,   # [BATCH, H, S, HEAD_DIM]
    in3_ptr,   # [HEAD_DIM, 2*S-1]
    in2_ptr,   # [BATCH, H, S, H, S]
    in0_ptr,   # [BATCH, N, N]
    in4_ptr,   # [BATCH, N, D_MODEL]
    out_ptr,   # [BATCH, D_MODEL, N]
    BATCH,
    HEADS,
    SEQ,
    D_MODEL,
    KV_HEADS,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (BATCH*KV_HEADS, ceil(N/BLOCK_M), ceil(D_MODEL/BLOCK_N))
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    pid_b = pid_bh // HEADS
    pid_h = pid_bh  % HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query positions 0..N-1
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output features 0..D_MODEL-1

    # ── 1. Relative-position logits [BLOCK_M, BLOCK_K] ────────────────────
    #   k = h64 + j64*(SEQ-1),  h64 in [0,SEQ-2],  j64 in [0,SEQ-1]
    #   k_per_head = SEQ-1 = 7 for SEQ=8
    offs_k = tl.arange(0, BLOCK_K)   # 0..BLOCK_K-1; mask k >= (SEQ-1)*SEQ = 15

    # Load in1[pid_b, pid_h, offs_m, :] → [BLOCK_M, BLOCK_K]
    in1_base = pid_b * (HEADS * SEQ * D_MODEL) + pid_h * (SEQ * D_MODEL)
    in1_offs = in1_base + offs_m[:, None] * D_MODEL + offs_k[None, :]
    in1_mask = (offs_m[:, None] < N) & (offs_k[None, :] < SEQ * (SEQ - 1))
    in1_vals = tl.load(in1_ptr + in1_offs, mask=in1_mask, other=0.0)

    # Load in3: in3[offs_k, :] → [BLOCK_K, 2*S-1]
    in3_cols = tl.arange(0, 15)[None, :]          # 2*S-1 = 15
    in3_offs = offs_k[:, None] * 15 + in3_cols
    in3_mask = (offs_k[:, None] < SEQ * (SEQ - 1)) & (in3_cols < 15)
    in3_vals = tl.load(in3_ptr + in3_offs, mask=in3_mask, other=0.0)

    # [BLOCK_M, BLOCK_K] @ [BLOCK_K, 15] → [BLOCK_M, 15]
    logits = tl.dot(in1_vals.to(tl.float32), in3_vals.to(tl.float32))
    logits = logits[:, :15]

    # k → (h64, j64): h64 = k % (SEQ-1),  j64 = k // (SEQ-1)
    k_per_head = SEQ - 1                   # = 7
    h64  = offs_k % k_per_head             # head index
    j64  = offs_k // k_per_head            # rel-pos index
    # relative_pos[i, j64] = j64 - i + (SEQ-1)
    rel_pos = j64[None, :] - offs_m[:, None] + (SEQ - 1)

    # in2[pid_b, pid_h, offs_m, h64, j64]
    # linear offset: b*(H*S*H*S) + h*(S*H*S) + i*(H*S) + h64*S + j64
    in2_base = pid_b * (HEADS * SEQ * HEADS * SEQ) + pid_h * (SEQ * HEADS * SEQ)
    in2_offs = in2_base \
               + offs_m[:, None] * (HEADS * SEQ) \
               + h64[None, :] * SEQ \
               + j64[None, :]
    in2_mask = (offs_m[:, None] < N) & (offs_k[None, :] < SEQ * (SEQ - 1))
    rel_logits = tl.load(in2_ptr + in2_offs, mask=in2_mask, other=0.0).to(tl.float32)

    scores = logits + rel_logits

    # ── 2. Add absolute position bias ──────────────────────────────────────
    in0_offs = pid_b * N * N + offs_m[:, None] * N + offs_k[None, :]
    in0_mask = (offs_m[:, None] < N) & (offs_k[None, :] < SEQ * (SEQ - 1))
    abs_logits = tl.load(in0_ptr + in0_offs, mask=in0_mask, other=0.0).to(tl.float32)

    scores = scores + abs_logits

    # Mask k >= N (pad to N with -inf)
    scores = tl.where(offs_k[None, :] < N, scores, float('-inf'))

    # ── 3. Softmax over k (row-wise) ───────────────────────────────────────
    scores_max = tl.max(scores, axis=1)[:, None]
    scores = tl.where(offs_k[None, :] < N, scores - scores_max, float('-inf'))
    scores = tl.exp(scores)
    scores_sum = tl.sum(scores, axis=1)[:, None]
    scores = scores / scores_sum            # [BLOCK_M, BLOCK_K]

    # ── 4. Load values in4[pid_b, offs_k, offs_n] → [BLOCK_K, BLOCK_N] ────
    in4_offs = pid_b * N * D_MODEL \
               + offs_k[:, None] * D_MODEL \
               + offs_n[None, :]
    in4_mask = (offs_k[:, None] < N) & (offs_n[None, :] < D_MODEL)
    vals = tl.load(in4_ptr + in4_offs, mask=in4_mask, other=0.0)

    acc = tl.dot(scores.to(tl.float16), vals.to(tl.float16))  # [BLOCK_M, BLOCK_N]

    # ── 5. Store to out[pid_b, offs_n, offs_m] (transposed) ───────────────
    out_offs = pid_b * (D_MODEL * N) \
               + offs_n[None, :] * N \
               + offs_m[:, None]
    out_mask = (offs_n[None, :] < D_MODEL) & (offs_m[:, None] < N)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'HEADS', 'SEQ', 'D_MODEL', 'KV_HEADS'],
)
@triton.jit
def _fused_kernel_8h_bf16(
    in1_ptr,
    in3_ptr,
    in2_ptr,
    in0_ptr,
    in4_ptr,
    out_ptr,
    BATCH,
    HEADS,
    SEQ,
    D_MODEL,
    KV_HEADS,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    pid_b = pid_bh // HEADS
    pid_h = pid_bh  % HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)

    in1_base = pid_b * (HEADS * SEQ * D_MODEL) + pid_h * (SEQ * D_MODEL)
    in1_offs = in1_base + offs_m[:, None] * D_MODEL + offs_k[None, :]
    in1_mask = (offs_m[:, None] < N) & (offs_k[None, :] < SEQ * (SEQ - 1))
    in1_vals = tl.load(in1_ptr + in1_offs, mask=in1_mask, other=0.0)

    in3_cols = tl.arange(0, 15)[None, :]
    in3_offs = offs_k[:, None] * 15 + in3_cols
    in3_mask = (offs_k[:, None] < SEQ * (SEQ - 1)) & (in3_cols < 15)
    in3_vals = tl.load(in3_ptr + in3_offs, mask=in3_mask, other=0.0)

    logits = tl.dot(in1_vals.to(tl.float32), in3_vals.to(tl.float32))
    logits = logits[:, :15]

    k_per_head = SEQ - 1
    h64  = offs_k % k_per_head
    j64  = offs_k // k_per_head
    rel_pos = j64[None, :] - offs_m[:, None] + (SEQ - 1)

    in2_base = pid_b * (HEADS * SEQ * HEADS * SEQ) + pid_h * (SEQ * HEADS * SEQ)
    in2_offs = in2_base \
               + offs_m[:, None] * (HEADS * SEQ) \
               + h64[None, :] * SEQ \
               + j64[None, :]
    in2_mask = (offs_m[:, None] < N) & (offs_k[None, :] < SEQ * (SEQ - 1))
    rel_logits = tl.load(in2_ptr + in2_offs, mask=in2_mask, other=0.0).to(tl.float32)

    scores = logits + rel_logits

    in0_offs = pid_b * N * N + offs_m[:, None] * N + offs_k[None, :]
    in0_mask = (offs_m[:, None] < N) & (offs_k[None, :] < SEQ * (SEQ - 1))
    abs_logits = tl.load(in0_ptr + in0_offs, mask=in0_mask, other=0.0).to(tl.float32)

    scores = scores + abs_logits
    scores = tl.where(offs_k[None, :] < N, scores, float('-inf'))

    scores_max = tl.max(scores, axis=1)[:, None]
    scores = tl.where(offs_k[None, :] < N, scores - scores_max, float('-inf'))
    scores = tl.exp(scores)
    scores_sum = tl.sum(scores, axis=1)[:, None]
    scores = scores / scores_sum

    in4_offs = pid_b * N * D_MODEL \
               + offs_k[:, None] * D_MODEL \
               + offs_n[None, :]
    in4_mask = (offs_k[:, None] < N) & (offs_n[None, :] < D_MODEL)
    vals = tl.load(in4_ptr + in4_offs, mask=in4_mask, other=0.0)

    acc = tl.dot(scores.to(tl.bfloat16), vals.to(tl.bfloat16))

    out_offs = pid_b * (D_MODEL * N) \
               + offs_n[None, :] * N \
               + offs_m[:, None]
    out_mask = (offs_n[None, :] < D_MODEL) & (offs_m[:, None] < N)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Python wrappers
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _run_fused_relpos_8head_fp16(in_0, in_1, in_2, in_3, in_4):
    BATCH, HEADS, SEQ, D_MODEL = in_1.shape   # 4, 8, 8, 128
    KV_HEADS = HEADS
    N = HEADS * SEQ                            # 64

    out = torch.empty((BATCH, D_MODEL, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        BATCH * KV_HEADS,
        triton.cdiv(N,       meta['BLOCK_M']),
        triton.cdiv(D_MODEL, meta['BLOCK_N']),
    )

    _fused_kernel_8h_fp16[grid](
        in_1, in_3, in_2, in_0, in_4, out,
        BATCH, HEADS, SEQ, D_MODEL, KV_HEADS, N,
    )
    return out


@torch.fx.wrap
def _run_fused_relpos_8head_bf16(in_0, in_1, in_2, in_3, in_4):
    BATCH, HEADS, SEQ, D_MODEL = in_1.shape
    KV_HEADS = HEADS
    N = HEADS * SEQ

    out = torch.empty((BATCH, D_MODEL, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        BATCH * KV_HEADS,
        triton.cdiv(N,       meta['BLOCK_M']),
        triton.cdiv(D_MODEL, meta['BLOCK_N']),
    )

    _fused_kernel_8h_bf16[grid](
        in_1, in_3, in_2, in_0, in_4, out,
        BATCH, HEADS, SEQ, D_MODEL, KV_HEADS, N,
    )
    return out


def replacement_func():
    return _run_fused_relpos_8head_fp16