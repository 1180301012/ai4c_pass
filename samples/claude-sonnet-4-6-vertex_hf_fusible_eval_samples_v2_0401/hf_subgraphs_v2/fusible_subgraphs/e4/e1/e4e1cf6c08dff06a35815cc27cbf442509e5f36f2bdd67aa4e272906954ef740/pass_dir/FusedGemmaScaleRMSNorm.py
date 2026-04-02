import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: RMSNorm from tmp_2/in_1 → tmp_13
#
# tmp_2 is already computed externally (in_0 * in_2) and is also returned
# by the model, so we start the pattern from tmp_2 to avoid a multi-output
# returning-node conflict.
#
#   tmp_4  = tmp_2.float()
#   tmp_5  = tmp_4.pow(2)
#   tmp_6  = tmp_5.mean(-1, keepdim=True)
#   tmp_7  = tmp_6 + 1e-06
#   tmp_8  = torch.rsqrt(tmp_7)
#   tmp_9  = tmp_4 * tmp_8
#   tmp_10 = in_1.float()
#   tmp_11 = 1.0 + tmp_10
#   tmp_12 = tmp_9 * tmp_11
#   tmp_13 = tmp_12.type_as(tmp_2)   →  bfloat16
# ---------------------------------------------------------------------------
def pattern(tmp_2, in_1):
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13


def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)


# ---------------------------------------------------------------------------
# Triton kernel – one program per row, no autotune overhead.
# BLOCK_SIZE == n_cols == 2048 is a compile-time constant; the compiler
# bakes in 1/2048 and turns row_base multiply into a left-shift.
# num_warps=4 (128 threads, 16 elements/thread) is optimal for a 2048-elem
# single-block reduction on Ampere.
# ---------------------------------------------------------------------------
@triton.jit
def gemma_rmsnorm_kernel(
    x_ptr,           # [n_rows, BLOCK_SIZE]  bfloat16
    w_ptr,           # [BLOCK_SIZE]           bfloat16
    out_ptr,         # [n_rows, BLOCK_SIZE]  bfloat16
    BLOCK_SIZE: tl.constexpr,
):
    row_idx  = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    row_base = row_idx * BLOCK_SIZE   # power-of-2 shift

    # ---- load (no mask needed: BLOCK_SIZE == n_cols) ----
    x_bf16 = tl.load(x_ptr + row_base + col_offs)
    w_bf16 = tl.load(w_ptr + col_offs)

    # ---- upcast to float32 ----
    tmp_4  = x_bf16.to(tl.float32)
    tmp_11 = 1.0 + w_bf16.to(tl.float32)   # Gemma: 1 + weight

    # ---- RMSNorm  (1/BLOCK_SIZE is a compile-time literal → no division) ----
    rms_sq  = tl.sum(tmp_4 * tmp_4, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(rms_sq + 1e-06)

    # ---- normalize, scale, cast ----
    tmp_13 = (tmp_4 * inv_rms * tmp_11).to(tl.bfloat16)

    tl.store(out_ptr + row_base + col_offs, tmp_13)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX does not trace inside it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def gemma_rmsnorm(tmp_2, in_1):
    n_cols = tmp_2.shape[-1]
    n_rows = tmp_2.numel() // n_cols

    out = torch.empty_like(tmp_2)

    gemma_rmsnorm_kernel[(n_rows,)](
        tmp_2,
        in_1,
        out,
        BLOCK_SIZE=n_cols,
        num_warps=4,
    )

    return out


# ---------------------------------------------------------------------------
# Required entry-point – returns the callable, does NOT call it
# ---------------------------------------------------------------------------
def replacement_func():
    return gemma_rmsnorm