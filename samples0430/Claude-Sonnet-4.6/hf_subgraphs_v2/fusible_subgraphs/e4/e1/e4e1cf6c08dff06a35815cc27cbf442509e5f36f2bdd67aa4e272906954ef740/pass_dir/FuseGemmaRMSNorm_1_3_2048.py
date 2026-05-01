import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuses the Gemma RMSNorm body (NOT including tmp_2 = in_0 * in_2,
# which is returned by the model and must remain in the graph).
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
#   tmp_13 = tmp_12.type_as(tmp_2)
#   return tmp_13          ← single output (ONE returning node)
#
# tmp_2 = in_0 * in_2 stays in the original graph (it is directly returned
# by the model so it must not be erased by the pattern replacement).
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
# Triton kernel
#   - One CTA per row; BLOCK_SIZE = 2048 covers the full row
#   - Fixed config: num_warps=8 (optimal for 2048-element bf16 row on A30)
#   - Both loads issued early to maximise memory-level parallelism
# ---------------------------------------------------------------------------
@triton.jit
def _gemma_rmsnorm_kernel(
    in0_ptr,   # [n_rows, n_cols] bfloat16  (= tmp_2)
    in1_ptr,   # [n_cols]         bfloat16  (= in_1, RMSNorm weight)
    out_ptr,   # [n_rows, n_cols] bfloat16  (= tmp_13)
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_base = row * n_cols

    # Issue both loads early to maximise memory-level parallelism.
    # The GPU can overlap the two cache-line fetches.
    x = tl.load(in0_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(in1_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS: mean of squares → rsqrt (computation hides w-load latency)
    sq_sum  = tl.sum(x * x, axis=0)
    mean_sq = sq_sum / n_cols
    rstd    = 1.0 / tl.sqrt(mean_sq + eps)

    # Normalized output → bfloat16
    tl.store(out_ptr + row_base + cols,
             (x * rstd * (1.0 + w)).to(tl.bfloat16),
             mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper  (@torch.fx.wrap → FX treats it as a leaf, single output)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_gemma_rmsnorm(tmp_2, in_1):
    """
    tmp_2 : [1, 3, 2048]  bfloat16  CUDA  (scaled input, already computed)
    in_1  : [2048]         bfloat16  CUDA  (RMSNorm weight)
    returns: tmp_13  [1, 3, 2048] bfloat16 CUDA
    """
    n_cols = tmp_2.shape[-1]
    # Pure Python arithmetic on torch.Size — no aten dispatch, no overhead
    n_rows = 1
    for d in tmp_2.shape[:-1]:
        n_rows = n_rows * d

    out = torch.empty_like(tmp_2)   # will hold tmp_13

    _gemma_rmsnorm_kernel[(n_rows,)](
        tmp_2, in_1,
        out,
        n_cols,
        1e-6,
        BLOCK_SIZE=2048,
        num_warps=8,
    )

    return out


def replacement_func():
    return fused_gemma_rmsnorm