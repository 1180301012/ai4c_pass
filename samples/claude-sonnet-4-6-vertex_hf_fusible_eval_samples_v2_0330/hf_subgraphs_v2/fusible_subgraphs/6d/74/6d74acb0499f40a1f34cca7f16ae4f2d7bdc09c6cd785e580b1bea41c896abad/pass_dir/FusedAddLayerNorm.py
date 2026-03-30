import torch
import triton
import triton.language as tl


# ────────────────────────────────────────────────────────────────────────────
# Pattern: matches the full Add+LayerNorm subgraph present in all target graphs
# ────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ────────────────────────────────────────────────────────────────────────────
# Triton kernel – one program per row; BLOCK_SIZE must cover N_COLS (768).
#
# Key optimisations vs naïve implementation:
#   1. inv_N_COLS passed as fp32 scalar → multiplications, no per-row divisions
#   2. Variance via E[x²] – E[x]² – eliminates one tl.where and one tl.sum
#      on centred data (masked positions are 0.0 in x, contribute nothing)
#   3. tl.math.rsqrt → single hardware MUFU rsqrt instruction
#   4. Weight/bias pre-fetched before main data → latency hiding in L2
#   5. autotune over num_warps (BLOCK_SIZE=1024 minimal waste; =2048 extra ILP)
# ────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # num_warps=4 (128 threads, 8 elems/thread) → maximum ILP per thread,
        # 16 concurrent CTAs/SM (2048/128), best latency hiding.
        # num_warps=8 gives 8 CTAs/SM; both have 64 active warps/SM, but
        # num_warps=4 has 2× the per-thread ILP for load pipelining.
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ],
    key=["N_COLS"],
)
@triton.jit
def _fused_add_layernorm_kernel(
    in3_ptr,      # [N_ROWS, N_COLS]  (in_3)
    in2_ptr,      # [N_ROWS, N_COLS]  (in_2)
    weight_ptr,   # [N_COLS]          (in_1 / weight)
    bias_ptr,     # [N_COLS]          (in_0 / bias)
    out_ptr,      # [N_ROWS, N_COLS]  float32 output
    N_ROWS,
    N_COLS,
    inv_N_COLS,   # precomputed 1.0 / N_COLS
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    base = row * N_COLS
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # ── pre-fetch weight & bias (L2-cached after first row) ──────────────────
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)

    # ── load & add (masked positions → 0.0, contribute nothing to sums) ──────
    x3 = tl.load(in3_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in2_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    x  = x3 + x2

    # ── single-pass statistics: E[x] and E[x²] ───────────────────────────────
    x_sq  = x * x
    sum_x  = tl.sum(x,    axis=0)
    sum_x2 = tl.sum(x_sq, axis=0)

    mean = sum_x  * inv_N_COLS
    var  = sum_x2 * inv_N_COLS - mean * mean   # var = E[x²] – E[x]²

    # ── normalise + affine ────────────────────────────────────────────────────
    rstd = tl.math.rsqrt(var + eps)
    xn   = (x - mean) * rstd                  # masked positions masked at store
    out  = w * xn + b

    tl.store(out_ptr + base + cols, out, mask=mask)


# ────────────────────────────────────────────────────────────────────────────
# Wrapper (must be decorated with @torch.fx.wrap)
# ────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [N_COLS]          (any fp dtype)
    in_1 : weight [N_COLS]          (any fp dtype)
    in_2 : input  [*, N_COLS]       (any fp dtype)
    in_3 : input  [*, N_COLS]       (any fp dtype)
    Returns a float32 tensor with the same shape as in_2 / in_3.
    """
    shape  = in_2.shape
    N_COLS = shape[-1]
    N_ROWS = in_2.numel() // N_COLS

    in2_flat = in_2.reshape(N_ROWS, N_COLS)
    in3_flat = in_3.reshape(N_ROWS, N_COLS)

    out = torch.empty(N_ROWS, N_COLS, dtype=torch.float32, device=in_2.device)

    _fused_add_layernorm_kernel[(N_ROWS,)](
        in3_flat, in2_flat,
        in_1, in_0,
        out,
        N_ROWS, N_COLS,
        1.0 / N_COLS,   # inv_N_COLS
        1e-7,           # eps
    )

    return out.reshape(shape)


def replacement_func():
    return fused_add_layernorm