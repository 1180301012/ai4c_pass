import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: match the full relative-position-bias + softmax + attn@V + transpose
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 256, 256)
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
#   in1  : [BATCH, H, S, HEAD_DIM]   (q embeddings)
#   in3  : [HEAD_DIM, 31]             (rel logits linear weights)
#   in2  : [BATCH, H, S, H, S]        (rel logits, S=16)
#   in0  : [BATCH, N, N]              (abs logits, N=256)
#   in4  : [BATCH, N, D]              (values, N=256, D=128)
#   out  : [BATCH, D, N]              (transposed result)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'HEADS', 'SEQ', 'D_MODEL', 'KV_HEADS'],
)
@triton.jit
def _fused_kernel_fp16(
    in1_ptr,   # [BATCH, H, S, HEAD_DIM]
    in3_ptr,   # [HEAD_DIM, 31]
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
    # Grid: (BATCH*HEADS, ceil(N/BLOCK_M), ceil(D_MODEL/BLOCK_N))
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    pid_b = pid_bh // HEADS
    pid_h = pid_bh  % HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query positions (0..N-1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output features  (0..D_MODEL-1)

    # ── 1. Compute relative-position logits tensor [BLOCK_M, BLOCK_K] ──────
    #   rel_logit[b, h, i, j] = in1[b,h,i,:] @ in3[:, (j-i+SEQ-1) % (SEQ*2-1)]
    #   k = (j - i + SEQ - 1) % (2*SEQ-1)  encodes the relative position
    #   We tile over k (k = 0 .. 30, i.e. 31 values)
    #
    # Grid covers all k values in one tile (BLOCK_K >= 31)
    offs_k = tl.arange(0, BLOCK_K)   # 0..BLOCK_K-1, mask k>=31

    # Load in1 slice: in1[pid_b, pid_h, offs_m, :]
    #   linearised: pid_b*(H*S*D) + pid_h*(S*D) + offs_m*D + d
    in1_base = pid_b * (HEADS * SEQ * D_MODEL) + pid_h * (SEQ * D_MODEL)
    in1_offs = in1_base + offs_m[:, None] * D_MODEL + offs_k[None, :]  # [BLOCK_M, BLOCK_K]
    in1_mask = (offs_m[:, None] < N) & (offs_k[None, :] < 31)
    in1_vals = tl.load(in1_ptr + in1_offs, mask=in1_mask, other=0.0)   # float16

    # Load in3: in3[offs_k, :] for k in [0, 31)
    in3_offs = offs_k[:, None] * 31 + tl.arange(0, 31)[None, :]       # [BLOCK_K, 31]
    in3_mask = offs_k[:, None] < 31
    in3_vals = tl.load(in3_ptr + in3_offs, mask=in3_mask, other=0.0)   # float16

    # [BLOCK_M, BLOCK_K] @ [BLOCK_K, 31] → [BLOCK_M, 31]
    logits = tl.dot(in1_vals.to(tl.float32), in3_vals.to(tl.float32))  # fp32 acc
    logits = logits[:, :31]   # mask already zeroed k>=31

    # Map k -> head-index h64 and offset j64:
    #   k = h64 + j64*(SEQ-1)  →  h64 = k % (SEQ-1), j64 = k // (SEQ-1)
    #   Then: relative position = j64 - i + (SEQ-1) = k//(SEQ-1) - i + SEQ-1
    #   And head = h64 = k % (SEQ-1)
    k_per_head = SEQ - 1                        # = 15 for SEQ=16
    h64  = offs_k % k_per_head                  # head index 0..SEQ-2
    j64  = offs_k // k_per_head                 # relative pos 0..SEQ-1
    # relative_pos[i, j64] = j64 - i + (SEQ-1)
    #   = (offs_m row) and (offs_k col → j64)
    rel_pos = j64[None, :] - offs_m[:, None] + (SEQ - 1)   # [BLOCK_M, BLOCK_K]
    # in2 index: batch=pid_b, head=h64, query=i, k_head=j64, k_rel=i_rel
    #   NOTE: in2[b,h,i,h64,j64] = in2[b,h,i,(k%15),(k//15)]
    #   linear offset: b*(H*S*H*S) + h*(S*H*S) + i*(H*S) + h64*(S) + j64
    in2_base = pid_b * (HEADS * SEQ * HEADS * SEQ) + pid_h * (SEQ * HEADS * SEQ)
    in2_offs = in2_base \
               + offs_m[:, None] * (HEADS * SEQ) \
               + h64[None, :] * SEQ \
               + j64[None, :]                             # [BLOCK_M, BLOCK_K]
    in2_mask = (offs_m[:, None] < N) & (offs_k[None, :] < 31)
    rel_logits = tl.load(in2_ptr + in2_offs, mask=in2_mask, other=0.0).to(tl.float32)

    # Combined scores [BLOCK_M, BLOCK_K] = rel_logits + logits
    scores = logits + rel_logits

    # ── 2. Add absolute position bias ──────────────────────────────────────
    #   in0[pid_b, offs_m, k]  for k in [0, N)
    in0_offs = pid_b * N * N + offs_m[:, None] * N + offs_k[None, :]  # [BLOCK_M, BLOCK_K]
    in0_mask = (offs_m[:, None] < N) & (offs_k[None, :] < 31)
    abs_logits = tl.load(in0_ptr + in0_offs, mask=in0_mask, other=0.0).to(tl.float32)

    scores = scores + abs_logits

    # ── 3. Mask out k >= N (padding) ───────────────────────────────────────
    #   The computation above produced 31 columns (k=0..30). The attn
    #   matrix is square [N x N] so we need to pad k=31..N-1 with -inf.
    #   N=256 for 16-head case; we compute a full [BLOCK_M, BLOCK_N] tile
    #   of the output and handle the k dimension below.
    scores = tl.where(offs_k[None, :] < N, scores, float('-inf'))

    # ── 4. Softmax over k (row-wise) ───────────────────────────────────────
    scores_max = tl.max(scores, axis=1)[:, None]
    scores = tl.where(offs_k[None, :] < N, scores - scores_max, float('-inf'))
    scores = tl.exp(scores)
    scores_sum = tl.sum(scores, axis=1)[:, None]
    scores = scores / scores_sum                        # [BLOCK_M, BLOCK_K]

    # ── 5. Load values in4[pid_b, k, offs_n] → [BLOCK_K, BLOCK_N] ─────────
    #   Then acc += scores @ values  →  [BLOCK_M, BLOCK_N] accumulates D_MODEL dim
    in4_offs = pid_b * N * D_MODEL \
               + offs_k[:, None] * D_MODEL \
               + offs_n[None, :]                             # [BLOCK_K, BLOCK_N]
    in4_mask = (offs_k[:, None] < N) & (offs_n[None, :] < D_MODEL)
    vals = tl.load(in4_ptr + in4_offs, mask=in4_mask, other=0.0)

    acc = tl.dot(scores.to(tl.float16), vals.to(tl.float16))  # [BLOCK_M, BLOCK_N]

    # ── 6. Store to out[pid_b, offs_n, offs_m]  (transposed output) ────────
    out_offs = pid_b * (D_MODEL * N) \
               + offs_n[None, :] * N \
               + offs_m[:, None]                                # [BLOCK_M, BLOCK_N]
    out_mask = (offs_n[None, :] < D_MODEL) & (offs_m[:, None] < N)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'HEADS', 'SEQ', 'D_MODEL', 'KV_HEADS'],
)
@triton.jit
def _fused_kernel_bf16(
    in1_ptr,   # [BATCH, H, S, HEAD_DIM]
    in3_ptr,   # [HEAD_DIM, 31]
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
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    pid_b = pid_bh // HEADS
    pid_h = pid_bh  % HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)   # tiles the 31 rel-pos values

    in1_base = pid_b * (HEADS * SEQ * D_MODEL) + pid_h * (SEQ * D_MODEL)
    in1_offs = in1_base + offs_m[:, None] * D_MODEL + offs_k[None, :]
    in1_mask = (offs_m[:, None] < N) & (offs_k[None, :] < 31)
    in1_vals = tl.load(in1_ptr + in1_offs, mask=in1_mask, other=0.0)

    in3_offs = offs_k[:, None] * 31 + tl.arange(0, 31)[None, :]
    in3_mask = offs_k[:, None] < 31
    in3_vals = tl.load(in3_ptr + in3_offs, mask=in3_mask, other=0.0)

    logits = tl.dot(in1_vals.to(tl.float32), in3_vals.to(tl.float32))
    logits = logits[:, :31]

    k_per_head = SEQ - 1
    h64  = offs_k % k_per_head
    j64  = offs_k // k_per_head
    rel_pos = j64[None, :] - offs_m[:, None] + (SEQ - 1)

    in2_base = pid_b * (HEADS * SEQ * HEADS * SEQ) + pid_h * (SEQ * HEADS * SEQ)
    in2_offs = in2_base \
               + offs_m[:, None] * (HEADS * SEQ) \
               + h64[None, :] * SEQ \
               + j64[None, :]
    in2_mask = (offs_m[:, None] < N) & (offs_k[None, :] < 31)
    rel_logits = tl.load(in2_ptr + in2_offs, mask=in2_mask, other=0.0).to(tl.float32)

    scores = logits + rel_logits

    in0_offs = pid_b * N * N + offs_m[:, None] * N + offs_k[None, :]
    in0_mask = (offs_m[:, None] < N) & (offs_k[None, :] < 31)
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
# Python wrapper (float16)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _run_fused_relpos_16head_fp16(in_0, in_1, in_2, in_3, in_4):
    BATCH, HEADS, SEQ, D_MODEL = in_1.shape   # 4, 16, 16, 128
    KV_HEADS = HEADS
    N = HEADS * SEQ                            # 256

    out = torch.empty((BATCH, D_MODEL, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        BATCH * HEADS,
        triton.cdiv(N,    meta['BLOCK_M']),
        triton.cdiv(D_MODEL, meta['BLOCK_N']),
    )

    _fused_kernel_fp16[grid](
        in_1, in_3, in_2, in_0, in_4, out,
        BATCH, HEADS, SEQ, D_MODEL, KV_HEADS, N,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper (bfloat16)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _run_fused_relpos_16head_bf16(in_0, in_1, in_2, in_3, in_4):
    BATCH, HEADS, SEQ, D_MODEL = in_1.shape
    KV_HEADS = HEADS
    N = HEADS * SEQ

    out = torch.empty((BATCH, D_MODEL, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        BATCH * HEADS,
        triton.cdiv(N,    meta['BLOCK_M']),
        triton.cdiv(D_MODEL, meta['BLOCK_N']),
    )

    _fused_kernel_bf16[grid](
        in_1, in_3, in_2, in_0, in_4, out,
        BATCH, HEADS, SEQ, D_MODEL, KV_HEADS, N,
    )
    return out


def replacement_func():
    return _run_fused_relpos_16head_fp16