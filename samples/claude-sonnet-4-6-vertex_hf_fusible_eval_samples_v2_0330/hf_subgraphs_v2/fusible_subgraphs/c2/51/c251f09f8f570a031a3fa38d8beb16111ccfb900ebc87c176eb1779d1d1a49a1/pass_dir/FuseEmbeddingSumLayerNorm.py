import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 9-embedding sum  →  LayerNorm  →  Dropout(training=False)
# ---------------------------------------------------------------------------

def pattern(e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias):
    s = e1 + e2
    s = s + e3
    s = s + e4
    s = s + e5
    s = s + e6
    s = s + e7
    s = s + e8
    s = s + e9
    s = torch.nn.functional.layer_norm(s, (768,), ln_weight, ln_bias, 1e-12)
    s = torch.nn.functional.dropout(s, 0.1, False, False)
    return s


def replacement_args(e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias):
    return (e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias)


# ---------------------------------------------------------------------------
# Kernel A – "row kernel" (one CUDA program per token position, B*S total)
#
# Best for large B (e.g. B=128) where we need many programs to saturate the
# GPU SMs.  Uses clamped safe-offsets to avoid out-of-bounds memory faults
# when BLOCK_D=1024 > D=768.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_emb_sum_ln_row_kernel(
    e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr,
    e6_ptr, e7_ptr, e8_ptr, e9_ptr,
    w_ptr, b_ptr, out_ptr,
    N,                       # B * S
    S,                       # sequence length (broadcast modulo)
    D:          tl.constexpr,
    BLOCK_D:    tl.constexpr,
    IS_FP16:    tl.constexpr,
    IS_BF16:    tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    safe = tl.where(mask, offs, 0)

    full_base  = row * D
    bcast_base = (row % S) * D

    x  = tl.load(e1_ptr + full_base  + safe).to(tl.float32)
    x += tl.load(e2_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e3_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e4_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e5_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e6_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e7_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e8_ptr + bcast_base + safe).to(tl.float32)
    x += tl.load(e9_ptr + full_base  + safe).to(tl.float32)
    x  = tl.where(mask, x, 0.0)

    mean  = tl.sum(x, axis=0) / D
    diff  = tl.where(mask, x - mean, 0.0)
    var   = tl.sum(diff * diff, axis=0) / D
    rstd  = 1.0 / tl.sqrt(var + 1e-12)
    w     = tl.load(w_ptr + safe).to(tl.float32)
    b_v   = tl.load(b_ptr + safe).to(tl.float32)
    out_f = tl.where(mask, (x - mean) * rstd * w + b_v, 0.0)

    if IS_FP16:
        tl.store(out_ptr + full_base + safe, out_f.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + full_base + safe, out_f.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + full_base + safe, out_f, mask=mask)


# ---------------------------------------------------------------------------
# Kernel B – "batched kernel" (one program per sequence position, loop over B)
#
# Best for moderate B (≤ 32).  The 7 broadcast embeddings (shape [1,S,768])
# are loaded ONCE into registers per program and reused across all B batch
# items, reducing HBM traffic by up to ~3.5x vs Kernel A.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_emb_sum_ln_batched_kernel(
    e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr,
    e6_ptr, e7_ptr, e8_ptr, e9_ptr,
    w_ptr, b_ptr, out_ptr,
    B,                       # batch size (loop trip count)
    S,                       # sequence length
    D:          tl.constexpr,
    BLOCK_D:    tl.constexpr,
    IS_FP16:    tl.constexpr,
    IS_BF16:    tl.constexpr,
):
    seq  = tl.program_id(0)  # one program per sequence position
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    safe = tl.where(mask, offs, 0)

    bc_base = seq * D   # broadcast embeddings are identical for all batches

    # Load 7 broadcast embeddings ONCE; sum into a single register vector
    bc  = tl.load(e2_ptr + bc_base + safe).to(tl.float32)
    bc += tl.load(e3_ptr + bc_base + safe).to(tl.float32)
    bc += tl.load(e4_ptr + bc_base + safe).to(tl.float32)
    bc += tl.load(e5_ptr + bc_base + safe).to(tl.float32)
    bc += tl.load(e6_ptr + bc_base + safe).to(tl.float32)
    bc += tl.load(e7_ptr + bc_base + safe).to(tl.float32)
    bc += tl.load(e8_ptr + bc_base + safe).to(tl.float32)
    bc  = tl.where(mask, bc, 0.0)

    # LN weight/bias: also shared across batches
    w   = tl.load(w_ptr + safe).to(tl.float32)
    b_v = tl.load(b_ptr + safe).to(tl.float32)

    # Loop over batch items – only full-batch embeddings change
    for bi in range(0, B):
        full_base = (bi * S + seq) * D

        x  = tl.load(e1_ptr + full_base + safe).to(tl.float32)   # word emb
        x += tl.load(e9_ptr + full_base + safe).to(tl.float32)   # token-type emb
        x += bc                       # add precomputed broadcast sum
        x  = tl.where(mask, x, 0.0)  # zero out padded lanes

        mean  = tl.sum(x, axis=0) / D
        diff  = tl.where(mask, x - mean, 0.0)
        var   = tl.sum(diff * diff, axis=0) / D
        rstd  = 1.0 / tl.sqrt(var + 1e-12)
        out_f = tl.where(mask, (x - mean) * rstd * w + b_v, 0.0)

        if IS_FP16:
            tl.store(out_ptr + full_base + safe, out_f.to(tl.float16), mask=mask)
        elif IS_BF16:
            tl.store(out_ptr + full_base + safe, out_f.to(tl.bfloat16), mask=mask)
        else:
            tl.store(out_ptr + full_base + safe, out_f, mask=mask)


# ---------------------------------------------------------------------------
# Python-level wrapper
# ---------------------------------------------------------------------------

# Use the batched kernel only when:
#   1. B > 1   – there are actual broadcast savings to exploit
#   2. S >= 200 – enough programs to saturate the GPU SMs (A30 has 56 SMs;
#                 S=256 → 256/56 ≈ 4.6 programs/SM, good latency hiding;
#                 S=128 → 2.3 programs/SM, bandwidth utilisation suffers)
_BATCHED_S_THRESHOLD = 200

@torch.fx.wrap
def fused_emb_sum_layer_norm(
    e1, e2, e3, e4, e5, e6, e7, e8, e9,
    ln_weight, ln_bias,
):
    dtype   = e1.dtype
    is_fp16 = dtype == torch.float16
    is_bf16 = dtype == torch.bfloat16

    # e1 shape: [B, S, 768];  e2..e8 shape: [1, S, 768]
    B = e1.shape[0]
    S = e2.shape[-2]
    N = e1.numel() // 768   # = B * S

    out = torch.empty_like(e1)

    if B > 1 and S >= _BATCHED_S_THRESHOLD:
        # Kernel B: one program per seq position, loop over B
        # Amortises the 7 broadcast embedding loads across all B batches.
        # Only used when S is large enough to saturate all SMs.
        _fused_emb_sum_ln_batched_kernel[(S,)](
            e1, e2, e3, e4, e5, e6, e7, e8, e9,
            ln_weight, ln_bias, out,
            B=B, S=S,
            D=768, BLOCK_D=1024,
            IS_FP16=is_fp16, IS_BF16=is_bf16,
            num_warps=8,
        )
    else:
        # Kernel A: one program per row – maintains high SM occupancy.
        _fused_emb_sum_ln_row_kernel[(N,)](
            e1, e2, e3, e4, e5, e6, e7, e8, e9,
            ln_weight, ln_bias, out,
            N=N, S=S,
            D=768, BLOCK_D=1024,
            IS_FP16=is_fp16, IS_BF16=is_bf16,
            num_warps=8,
        )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-arg function that returns the callable wrapper
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_emb_sum_layer_norm