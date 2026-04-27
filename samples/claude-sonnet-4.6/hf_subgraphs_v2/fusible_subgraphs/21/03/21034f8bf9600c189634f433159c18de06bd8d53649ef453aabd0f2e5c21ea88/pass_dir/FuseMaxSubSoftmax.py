import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the 5-node chain that computes
#   max(in_0, -1, keepdim=True)[0].expand_as(in_0) - in_0  →  softmax(dim=-1)
# This is mathematically equivalent to softmax(-in_0, dim=-1).
# ---------------------------------------------------------------------------
def pattern(in_0):
    max_result = torch.max(in_0, -1, keepdim=True)
    max_vals = max_result[0]
    expanded = max_vals.expand_as(in_0)
    shifted = expanded - in_0
    out = torch.nn.functional.softmax(shifted, dim=-1)
    return out


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: row-wise fused (max-subtract-softmax).
# Each program handles exactly one row of N elements.
# Computation is performed in float32 for numerical stability regardless of
# the storage dtype (float32 / float16 / bfloat16).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=32, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def _fused_max_sub_softmax_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load row; for float16/bfloat16 storage tl.load returns those types.
    x = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)

    # Upcast to float32 for stable arithmetic.
    x_f32 = x.to(tl.float32)

    # Compute row-wise max, then shifted = max_val - x  (values ≤ 0).
    max_val = tl.max(x_f32, axis=0)
    shifted = max_val - x_f32

    # Softmax numerator / denominator.
    exp_vals = tl.exp(shifted)
    # Mask out padding entries (if BLOCK_SIZE > N) from the sum.
    exp_masked = tl.where(mask, exp_vals, 0.0)
    sum_exp = tl.sum(exp_masked, axis=0)
    out_f32 = exp_masked / sum_exp

    # Cast back to the original storage dtype and store.
    tl.store(out_ptr + row_start + offsets, out_f32.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap so FX does not trace into it).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_max_sub_softmax(in_0):
    N = in_0.shape[-1]
    n_rows = in_0.numel() // N
    out = torch.empty_like(in_0)
    _fused_max_sub_softmax_kernel[(n_rows,)](
        in_0,
        out,
        N=N,
    )
    return out


def replacement_func():
    return fused_max_sub_softmax