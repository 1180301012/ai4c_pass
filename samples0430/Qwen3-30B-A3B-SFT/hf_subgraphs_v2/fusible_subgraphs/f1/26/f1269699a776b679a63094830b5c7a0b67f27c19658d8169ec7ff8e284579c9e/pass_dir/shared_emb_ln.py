"""
Shared Triton kernels for fused embedding + layer-norm.
Both pass files import fused_emb_ln_dispatch so they share one
replacement_func, avoiding the replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ─── 768-dim kernel (eps = 1e-5) ─────────────────────────────────────────────
# One CTA per token.  D=768 < BLOCK_D=1024 so we use a mask.
# Single-pass Welford accumulates mean and variance simultaneously.

@triton.jit
def _fused_emb_ln_768_kernel(
    word_ids_ptr,
    pos_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    word_id = tl.load(word_ids_ptr + pid)
    pos_id  = tl.load(pos_ids_ptr  + pid)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Load embeddings; upcast to fp32 for numerical stability
    word_emb = tl.load(word_emb_ptr + word_id * D + offs, mask=mask, other=0.0).to(tl.float32)
    pos_emb  = tl.load(pos_emb_ptr  + pos_id  * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = word_emb + pos_emb

    # Single-pass Welford: two independent reductions, no sequential dependency
    n_inv  = 1.0 / D
    sum_x  = tl.sum(x,     axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean   = sum_x  * n_inv
    var    = sum_x2 * n_inv - mean * mean
    rstd   = 1.0 / tl.sqrt(var + eps)

    ln_w = tl.load(ln_w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    result = (x - mean) * rstd * ln_w + ln_b

    tl.store(out_ptr + pid * D + offs, result.to(OUT_DTYPE), mask=mask)


# ─── 64-dim kernel (eps = 1e-12) ──────────────────────────────────────────────
# D=64 fits in BLOCK_D=64 exactly — no masking overhead.

@triton.jit
def _fused_emb_ln_64_kernel(
    word_ids_ptr,
    pos_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    word_id = tl.load(word_ids_ptr + pid)
    pos_id  = tl.load(pos_ids_ptr  + pid)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    word_emb = tl.load(word_emb_ptr + word_id * D + offs, mask=mask, other=0.0).to(tl.float32)
    pos_emb  = tl.load(pos_emb_ptr  + pos_id  * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = word_emb + pos_emb

    # Single-pass Welford
    n_inv  = 1.0 / D
    sum_x  = tl.sum(x,     axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean   = sum_x  * n_inv
    var    = sum_x2 * n_inv - mean * mean
    rstd   = 1.0 / tl.sqrt(var + eps)

    ln_w = tl.load(ln_w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    result = (x - mean) * rstd * ln_w + ln_b

    tl.store(out_ptr + pid * D + offs, result.to(OUT_DTYPE), mask=mask)


# ─── Shared dispatch wrapper ───────────────────────────────────────────────────

# Map from torch dtype to triton dtype constant
_TRITON_DTYPE_MAP_768 = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@torch.fx.wrap
def fused_emb_ln_dispatch(in_0, in_4, in_5, in_3, in_2, in_1, route):
    """
    in_0 : input_ids    [B, S]           int64
    in_4 : word weight  [vocab, D]
    in_5 : position_ids [B, S]           int64
    in_3 : pos weight   [num_pos, D]
    in_2 : ln weight    [D]
    in_1 : ln bias      [D]
    route: "route_768" or "route_64"
    """
    B     = in_0.shape[0]
    S     = in_0.shape[1]
    n_seq = B * S

    if route == "route_768":
        D = 768
        BLOCK_D = 1024
        out = torch.empty((B, S, D), dtype=in_4.dtype, device=in_4.device)
        out_dtype = _TRITON_DTYPE_MAP_768[in_4.dtype]
        _fused_emb_ln_768_kernel[(n_seq,)](
            in_0, in_5,
            in_4, in_3,
            in_2, in_1,
            out,
            D=D,
            eps=1e-5,
            BLOCK_D=BLOCK_D,
            OUT_DTYPE=out_dtype,
            num_warps=4,
        )
        return out
    elif route == "route_64":
        D = 64
        BLOCK_D = 64
        out = torch.empty((B, S, D), dtype=in_4.dtype, device=in_4.device)
        out_dtype = _TRITON_DTYPE_MAP_768[in_4.dtype]
        _fused_emb_ln_64_kernel[(n_seq,)](
            in_0, in_5,
            in_4, in_3,
            in_2, in_1,
            out,
            D=D,
            eps=1e-12,
            BLOCK_D=BLOCK_D,
            OUT_DTYPE=out_dtype,
            num_warps=4,
        )
        return out