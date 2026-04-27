import torch
import triton
import triton.language as tl


@triton.jit
def _sum_and_sum2(a, a2, b, b2):
    """Combine two partial (sum, sum-of-squares) pairs for tl.reduce."""
    return a + b, a2 + b2


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['N_ROWS', 'H'],
)
@triton.jit
def fused_residual_layernorm_kernel(
    in0_ptr,   # bias  [H]
    in1_ptr,   # weight [H]
    in2_ptr,   # x1   [N_ROWS, H]
    in3_ptr,   # x2   [N_ROWS, H]
    out_ptr,   # output [N_ROWS, H] float32
    N_ROWS,
    H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    row_off = row * H

    # Load inputs (streaming: each row accessed once → evict_first to free L2)
    x1 = tl.load(in2_ptr + row_off + cols, mask=mask, other=0.0,
                  eviction_policy="evict_first").to(tl.float32)
    x2 = tl.load(in3_ptr + row_off + cols, mask=mask, other=0.0,
                  eviction_policy="evict_first").to(tl.float32)
    x = x1 + x2

    # Compute mean and variance via E[X²]−E[X]².
    # Padding positions are 0.0 so they contribute nothing to either sum.
    sum_x, sum_x2 = tl.reduce((x, x * x), axis=0, combine_fn=_sum_and_sum2)
    mean = sum_x / H
    var  = sum_x2 / H - mean * mean

    # Normalise, scale, shift
    x_c    = tl.where(mask, x - mean, 0.0)
    x_norm = x_c * tl.math.rsqrt(var + 1e-7)

    # Load weight/bias AFTER the reductions to minimise peak register use
    # during the reduction phase.  evict_last keeps them in L2 across rows.
    w = tl.load(in1_ptr + cols, mask=mask, other=0.0,
                 eviction_policy="evict_last").to(tl.float32)
    b = tl.load(in0_ptr + cols, mask=mask, other=0.0,
                 eviction_policy="evict_last").to(tl.float32)
    out = w * x_norm + b

    # Store: write-once streaming → evict_first to avoid polluting L2
    tl.store(out_ptr + row_off + cols, out, mask=mask,
             eviction_policy="evict_first")


@torch.fx.wrap
def fused_residual_layernorm(in_0, in_1, in_2, in_3):
    """
    Fused: residual add + layer-norm + affine transform.
      in_0 : bias   [H]                 (any float dtype)
      in_1 : weight [H]                 (any float dtype)
      in_2 : x1     [*, H]              (any float dtype)
      in_3 : x2     [*, H]              (any float dtype)
    Returns: weight * layernorm(x1 + x2) + bias  in float32, shape [*, H]
    Tensors are contiguous, so we can use pointer arithmetic directly without reshape.
    """
    H = in_2.shape[-1]
    N_ROWS = in_2.numel() // H

    # Must use torch.empty with explicit dtype=float32 — torch.empty_like inherits
    # the source dtype (bfloat16/float16) and ignores the override in this context.
    out = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)

    # BLOCK_H must be a power-of-2 >= H.  All graphs have H=768, so 1024 is correct.
    # Hardcoding avoids a Python function call on every forward pass.
    BLOCK_H = 1024 if H <= 1024 else triton.next_power_of_2(H)

    fused_residual_layernorm_kernel[(N_ROWS,)](
        in_0, in_1, in_2, in_3, out,
        N_ROWS, H,
        BLOCK_H=BLOCK_H,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

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


def replacement_func():
    return fused_residual_layernorm