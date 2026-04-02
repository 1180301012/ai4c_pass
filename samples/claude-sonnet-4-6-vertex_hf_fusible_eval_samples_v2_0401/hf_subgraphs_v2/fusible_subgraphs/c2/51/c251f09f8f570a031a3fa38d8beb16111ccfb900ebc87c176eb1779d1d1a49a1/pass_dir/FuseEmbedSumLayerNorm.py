import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 8 sequential additions of embedding tensors + LayerNorm + Dropout
# ---------------------------------------------------------------------------
# In LayoutLM the embedding output is the sum of 9 embedding vectors:
#   e0 (word)  + e1 (position) + e2 (x0) + e3 (y0) + e4 (x2)
#   + e5 (y3) + e6 (h) + e7 (w) + e8 (token_type)
# followed by LayerNorm and a no-op Dropout (training=False).
#
# e0, e8  shape: [B, S, 768]
# e1..e7  shape: [1, S, 768]  (broadcast over batch)
# ---------------------------------------------------------------------------

def pattern(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    t1 = e0 + e1
    t2 = t1 + e2
    t3 = t2 + e3
    t4 = t3 + e4
    t5 = t4 + e5
    t6 = t5 + e6
    t7 = t6 + e7
    t8 = t7 + e8
    t9 = torch.nn.functional.layer_norm(t8, (768,), ln_weight, ln_bias, 1e-12)
    out = torch.nn.functional.dropout(t9, 0.1, False, False)
    return out


def replacement_args(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    return (e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias)


# ===========================================================================
# Triton Kernel A: 9-tensor fused sum + LayerNorm
# ===========================================================================
# Used for large float32 / bfloat16 workloads.
# Each CTA processes ONE token (row of H=768 elements).
# Reads all 9 embeddings and performs LayerNorm in a single pass.
#
# e0, e8 : [B*S, H]  full-batch tensors   (row = pid)
# e1..e7 : [S,   H]  broadcast tensors    (row = pid % seq_len)
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=3),
    ],
    key=['n_tokens', 'hidden_size'],
)
@triton.jit
def _fused_9emb_sum_ln_kernel(
    e0_ptr, e1_ptr, e2_ptr, e3_ptr, e4_ptr,
    e5_ptr, e6_ptr, e7_ptr, e8_ptr,
    w_ptr, b_ptr, out_ptr,
    n_tokens, seq_len, hidden_size, eps,
    IS_BF16: tl.constexpr, IS_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    s   = pid % seq_len

    row_full = pid * hidden_size
    row_seq  = s   * hidden_size
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Load and accumulate all 9 embeddings in float32
    x  = tl.load(e0_ptr + row_full + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e1_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e2_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e3_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e4_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e5_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e6_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e7_ptr + row_seq  + cols, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e8_ptr + row_full + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=0) / hidden_size
    xc   = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    xn   = xc * rstd

    # Affine transform
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out_f32 = xn * w + b

    # Store (dropout training=False is identity → skip)
    if IS_BF16:
        tl.store(out_ptr + row_full + cols, out_f32.to(tl.bfloat16), mask=mask)
    elif IS_FP16:
        tl.store(out_ptr + row_full + cols, out_f32.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + row_full + cols, out_f32, mask=mask)


# ===========================================================================
# Triton Kernel B: standalone LayerNorm on a pre-summed tensor
# ===========================================================================
# Used for the optimized-fallback path (float16 / small n_tokens with B>1)
# and for the standard fallback (B=1).
# Input x is already the full sum: e0 + e1 + ... + e8.
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=3),
    ],
    key=['n_tokens', 'hidden_size'],
)
@triton.jit
def _triton_ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n_tokens, hidden_size, eps,
    IS_BF16: tl.constexpr, IS_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * hidden_size
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x    = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    xc   = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    xn   = xc * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out_f32 = xn * w + b

    if IS_BF16:
        tl.store(out_ptr + row_start + cols, out_f32.to(tl.bfloat16), mask=mask)
    elif IS_FP16:
        tl.store(out_ptr + row_start + cols, out_f32.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + row_start + cols, out_f32, mask=mask)


# ---------------------------------------------------------------------------
# Helper: launch the standalone Triton LayerNorm on a [B,S,H] tensor x.
# Not @torch.fx.wrap because it is only called from within the wrapper below.
# ---------------------------------------------------------------------------
def _apply_triton_ln(x, ln_weight, ln_bias, n_tokens, H, dtype, device):
    x_2d = x.reshape(n_tokens, H)
    out  = torch.empty(n_tokens, H, dtype=dtype, device=device)
    IS_BF16 = (dtype == torch.bfloat16)
    IS_FP16  = (dtype == torch.float16)
    _triton_ln_kernel[(n_tokens,)](
        x_2d, ln_weight, ln_bias, out,
        n_tokens, H, 1e-12,
        IS_BF16=IS_BF16, IS_FP16=IS_FP16,
    )
    return out.reshape(x.shape)


# ===========================================================================
# Top-level replacement wrapper
# ===========================================================================
@torch.fx.wrap
def fused_embed_sum_layernorm_dropout(
    e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias
):
    """
    Fused replacement: sum(9 embeddings) + LayerNorm + Dropout(training=False).

    Path A — 9-tensor Triton fused kernel  (large float32 / bfloat16):
      Reads all embeddings once; computes sum + LN in a single GPU kernel.
      Avoids 8 intermediate [B,S,H] tensors and 9 separate kernel launches.
      Broadcast tensors [1,S,H] are accessed via modulo and benefit from L2 reuse.

    Path B — optimized broadcast fallback  (float16, or small n_tokens, B>1):
      Pre-sums the 7 broadcast tensors [1,S,H] (small, fits in L2) with tensor '+',
      then performs 2 large [B,S,H] additions, then applies Triton LayerNorm.
      Reduces large-tensor HBM traffic from 8 passes → 2 passes (≈ 3× reduction).

    Path C — standard fallback  (B=1, all tensors same size):
      Sums all tensors with tensor '+', then applies Triton LayerNorm.
    """
    B      = e0.shape[0]
    S      = e0.shape[1]
    H      = e0.shape[2]   # 768
    n_tokens = B * S
    dtype    = e0.dtype
    device   = e0.device

    # ------------------------------------------------------------------
    # Path A: 9-tensor Triton fused kernel
    #   Best for large float32 / bfloat16 – a single GPU kernel reads
    #   every embedding exactly once and immediately computes LayerNorm,
    #   saving all 7 intermediate [B,S,H] materializations.
    # ------------------------------------------------------------------
    use_triton_fused = (
        (dtype == torch.float32  and n_tokens >= 256) or
        (dtype == torch.bfloat16 and n_tokens >= 512)
    )

    if use_triton_fused:
        e0_2d   = e0.reshape(n_tokens, H)
        e8_2d   = e8.reshape(n_tokens, H)
        seq_len = e1.shape[1]          # S  (e1 shape is [1, S, H])
        e1_2d   = e1.reshape(seq_len, H)
        e2_2d   = e2.reshape(seq_len, H)
        e3_2d   = e3.reshape(seq_len, H)
        e4_2d   = e4.reshape(seq_len, H)
        e5_2d   = e5.reshape(seq_len, H)
        e6_2d   = e6.reshape(seq_len, H)
        e7_2d   = e7.reshape(seq_len, H)
        out     = torch.empty(n_tokens, H, dtype=dtype, device=device)
        IS_BF16 = (dtype == torch.bfloat16)
        IS_FP16  = (dtype == torch.float16)
        _fused_9emb_sum_ln_kernel[(n_tokens,)](
            e0_2d, e1_2d, e2_2d, e3_2d, e4_2d,
            e5_2d, e6_2d, e7_2d, e8_2d,
            ln_weight, ln_bias, out,
            n_tokens, seq_len, H, 1e-12,
            IS_BF16=IS_BF16, IS_FP16=IS_FP16,
        )
        return out.reshape(e0.shape)

    # ------------------------------------------------------------------
    # Path B: broadcast optimized fallback  (B > 1)
    #   e1..e7 are [1,S,H] — sum them first (small, cache-resident),
    #   then do only 2 large [B,S,H] additions instead of 8.
    #   Triton LayerNorm is used instead of torch.nn.functional.layer_norm.
    # ------------------------------------------------------------------
    is_broadcast = (e1.shape[0] < e0.shape[0])   # True when B > 1

    if is_broadcast:
        sum_bc  = e1 + e2           # [1, S, H] — tiny, fits in L2
        sum_bc  = sum_bc + e3
        sum_bc  = sum_bc + e4
        sum_bc  = sum_bc + e5
        sum_bc  = sum_bc + e6
        sum_bc  = sum_bc + e7       # [1, S, H] — all 7 broadcast embeddings
        x = e0 + sum_bc             # [B, S, H] — broadcasts sum_bc over batch
        x = x + e8                  # [B, S, H] — second large addition
        return _apply_triton_ln(x, ln_weight, ln_bias, n_tokens, H, dtype, device)

    # ------------------------------------------------------------------
    # Path C: standard fallback  (B=1, no broadcast savings)
    #   All tensors are the same shape; just sum and apply Triton LN.
    # ------------------------------------------------------------------
    x = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8   # [B, S, H]
    return _apply_triton_ln(x, ln_weight, ln_bias, n_tokens, H, dtype, device)


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_embed_sum_layernorm_dropout