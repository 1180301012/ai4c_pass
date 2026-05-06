import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fuses the full embedding-similarity computation into one pass.
#
#   out10[0, n, c, d] = in4[0, n, d] - in0[c, d]   (view/expand/diff)
#   out9 [0, n, c]    = softmax(scale * sum_{c_dim}(in1[n,c_dim,d]-in2[c_dim,d])^2, dim=2)
#
# Grid: (N // BLOCK_N, C, D // BLOCK_D)
#   pid0 → batch of rows  (n rows)
#   pid1 → channel index  (c channel)
#   pid2 → block of D     (d blocks)
#
# Each CTA (block) handles one (n, c) pair for all BLOCK_D D-elements.
# The softmax reduction over C=32 is done with a Python loop (small C).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 2, 'BLOCK_D': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4, 'BLOCK_D': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 2, 'BLOCK_D': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4, 'BLOCK_D': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 1, 'BLOCK_D': 512}, num_warps=16, num_stages=3),
    ],
    key=['N', 'C', 'D'],
)
@triton.jit
def _fused_emb_sim_kernel(
    in1_ptr,    # [1, N, C, D]  – large feature map
    in2_ptr,    # [1, 1, C, D]  – codebook
    in3_ptr,    # [1, 1, C]     – per-channel scale
    in4_ptr,    # [1, N, D]     – attention context
    in0_ptr,    # [C, D]        – codebook vectors (flat)
    out10_ptr,  # [1, N, C, D]
    out9_ptr,   # [1, N, C]
    N, C, D,
    BLOCK_C: tl.constexpr,   # = 32  (all channels processed in loop)
    BLOCK_N: tl.constexpr,   # rows per CTA
    BLOCK_D: tl.constexpr,   # embedding dim elements per CTA
):
    pid_n = tl.program_id(0)   # which row block
    pid_c = tl.program_id(1)   # which channel
    pid_d = tl.program_id(2)   # which D-block

    n_start = pid_n * BLOCK_N
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # ── Load scale for this channel (scalar) ──────────────────────────────
    scale = tl.load(in3_ptr + pid_c).to(tl.float32)

    # ── Softmax numerator: sum_{c_dim}(in1 - in2)^2 ───────────────────────
    # We need per-row (c_dim) values, summed along C.
    # Loop over all C channels (unrolled since BLOCK_C = C = 32 constexpr).
    total_sq = tl.zeros([BLOCK_D], dtype=tl.float32)

    c_base = pid_c * D  # base offset into in0/in4 for this (n=0, c) pair

    for c in range(0, C, BLOCK_C):
        # Scalar load of in3[c]
        in3_val = tl.load(in3_ptr + c).to(tl.float32)

        # in0[c, d] : flat offset = c*D + d
        in0_base = c * D
        in0_val  = tl.load(in0_ptr + in0_base + d_offsets,
                           mask=d_mask, other=0.0).to(tl.float32)

        # in1[0, n_start//BLOCK_N, c, d] for the one row we're handling
        #   offset = (n_start + nn) * (C*D) + c*D + d
        # We handle BLOCK_N rows per CTA: nn in [0, BLOCK_N)
        nn   = n_start + tl.arange(0, BLOCK_N)          # [BLOCK_N]
        base = nn * (C * D) + c * D + d_offsets          # [BLOCK_N, BLOCK_D]
        in1_val = tl.load(in1_ptr + base,
                          mask=(nn < N)[:, None] & d_mask[None, :],
                          other=0.0).to(tl.float32)       # [BLOCK_N, BLOCK_D]

        # in2[0, 0, c, d] = in2_ptr[c*D + d]
        in2_val = tl.load(in2_ptr + c*D + d_offsets,
                          mask=d_mask, other=0.0).to(tl.float32)

        diff = in1_val - in2_val                          # [BLOCK_N, BLOCK_D]
        sq   = tl.sum(diff * diff, axis=0)               # [BLOCK_D]
        total_sq = total_sq + sq                          # accumulate

    sum_sq = tl.sum(total_sq)      # scalar: over BLOCK_D
    trans  = sum_sq * scale        # scale * cost

    # ── Numerically-stable softmax ────────────────────────────────────────
    # Promote to [BLOCK_N, 1] and broadcast for element-wise -trans in row
    num_rows  = tl.where(nn < N, 1.0, 1.0)   # [BLOCK_N] – always valid here
    exp_vals  = tl.exp(trans - num_rows * tl.log(sum_sq + 1.0e-8))
    norm_val  = sum_sq + 1.0e-8
    for c2 in range(0, C, BLOCK_C):           # re-sum for denominator
        base2   = (n_start // BLOCK_N) * (C * D) + c2 * D + d_offsets
        in1v_2d = tl.load(in1_ptr + base2,
                          mask=d_mask, other=0.0).to(tl.float32)
        in2v_2d = tl.load(in2_ptr + c2 * D + d_offsets,
                          mask=d_mask, other=0.0).to(tl.float32)
        diff2   = in1v_2d - in2v_2d
        sq2     = tl.sum(diff2 * diff2, axis=0)
        norm_val = sq2 + norm_val

    norm_val = tl.sum(norm_val)
    out9_val = tl.exp(trans - tl.log(norm_val + 1.0e-8)) / (norm_val + 1.0e-8)
    out9_val = out9_val.to(in3_ptr.dtype.element_ty)

    # ── Per-row offsets ────────────────────────────────────────────────────
    out10_base = (n_start // BLOCK_N) * (C * D)         # batch=0, scalar n

    # ── out10 = in4[n, d] - in0[c, d] ────────────────────────────────────
    # in4[0, n, d] offset = n*(C*D) + d  (strided by C*D for different n)
    # in0[c, d] offset    = c*D + d
    out_vals_10 = (in4_ptr + out10_base * (C * D) + d_offsets
                   - in0_ptr + c_base + d_offsets)

    # ── out9[0, n, pid_c] ─────────────────────────────────────────────────
    out_vals_9  = in3_ptr + (n_start // BLOCK_N) * C + pid_c

    # ── Store ─────────────────────────────────────────────────────────────
    tl.store(out10_ptr + out10_base * (C * D) + pid_c * D + d_offsets,
             tl.load(out_vals_10, mask=d_mask).to(tl.float16))

    # Write out9 with divided scale factor
    tl.store(out9_ptr + out_vals_9,
             trans * scale, mask=nn < N)


# ---------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_embedding_similarity(in_0, in_1, in_2, in_3, in_4):
    """
    Fused replacement for
        tmp_9 = softmax(in_3 * sum((in_1 - in_2)^2, dim=3), dim=2).unsqueeze(3)
        tmp_10 = (in_4.unsqueeze(2).expand(1,4096,32,512) - in_0.view(1,1,32,512))

    in_0 : [C, D]            – codewords
    in_1 : [1, N, C, D]      – expanded features
    in_2 : [1, 1, C, D]      – reshaped codewords
    in_3 : [1, 1, C]         – channel weights
    in_4 : [1, N, D]         – attention context
    """
    N = in_1.shape[1]   # 4096
    C = in_1.shape[2]   # 32
    D = in_1.shape[3]   # 512

    # in_0 is [C, D], treat as flat
    in4_v   = in_4.view(N, D)       # [N, D]
    in1_v   = in_1.reshape(N, C, D) # [N, C, D]
    in2_v   = in_2.reshape(1, 1, C, D)  # [1,1,C,D]
    in3_v   = in_3                       # [1,1,C]
    in0_v   = in_0                       # [C,D]

    out10   = torch.empty((1, N, C, D), dtype=in_1.dtype, device=in_1.device)
    out9    = torch.empty((1, N, C),    dtype=in_1.dtype, device=in_1.device)

    # Grid is a lambda so autotune adjusts BLOCK_N / BLOCK_D
    def grid(meta):
        return (
            (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
            C,
            (D + meta['BLOCK_D'] - 1) // meta['BLOCK_D'],
        )

    _fused_emb_sim_kernel[grid](
        in1_v, in2_v, in3_v, in4_v, in0_v,
        out10, out9,
        N, C, D,
        BLOCK_C=32,
    )

    return out10, out9


# ---------------------------------------------------------------------------
# Simple expand-subtract kernel for diagnostic matching
# Computes: out[0,n,c,d] = in_4[0,n,d] - in_0[c,d]
# ---------------------------------------------------------------------------
@triton.jit
def _expand_subtract_kernel(
    in4_ptr,    # [N, D]
    in0_ptr,    # [C, D]
    out_ptr,    # [N, C, D]
    N, C, D,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    n = pid_n
    c = pid_c
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D
    in4_val  = tl.load(in4_ptr + n * D + d_offsets, mask=d_mask, other=0.0)
    in0_val  = tl.load(in0_ptr + c * D + d_offsets, mask=d_mask, other=0.0)
    tl.store(out_ptr + n * C * D + c * D + d_offsets,
             in4_val - in0_val, mask=d_mask)


@torch.fx.wrap
def _expand_subtract_wrapper(in_4, in_0):
    N = in_4.shape[1]   # 4096
    C = in_0.shape[0]   # 32
    D = in_4.shape[2]   # 512
    out = torch.empty((1, N, C, D), dtype=in_4.dtype, device=in_4.device)
    BLOCK_D = 512
    # Pass tensors directly: since they're contiguous, pointer arithmetic in
    # the kernel still indexes correctly (skipping the batch=1 dimension for in_4).
    _expand_subtract_kernel[((N + BLOCK_D - 1) // BLOCK_D, C, 1)](
        in_4, in_0, out,
        N, C, D, BLOCK_D=BLOCK_D,
    )
    return out


def pattern(in_4, in_0):
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_4, in_0):
    return (in_4, in_0)


def replacement_func():
    return _expand_subtract_wrapper