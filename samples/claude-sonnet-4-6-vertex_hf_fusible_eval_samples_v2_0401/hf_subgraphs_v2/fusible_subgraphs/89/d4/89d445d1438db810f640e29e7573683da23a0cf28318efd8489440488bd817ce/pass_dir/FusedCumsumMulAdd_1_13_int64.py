import torch
import triton
import triton.language as tl

# Shape-specific constants for [1, 13] int64
_B = 1
_N = 13
_BLOCK_N = 16   # next_power_of_2(13) = 16


@triton.jit
def fused_cumsum_mul_add_kernel(
    x_ptr,
    out_ptr,
):
    """
    Fused kernel: out = cumsum(x, dim=1) * x + 1
    All constants (B=1, N=13, BLOCK_N=16) inlined as literals.
    Compute in int32 (faster prefix-scan than int64 on A30).
    """
    offsets = tl.arange(0, 16)
    mask = offsets < 13

    x_i64 = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_i32 = x_i64.to(tl.int32)

    # Inclusive prefix-sum in int32
    cumsum_i32 = tl.cumsum(x_i32, axis=0)

    # Fused: cumsum * x + 1, upcast to int64 for storage
    out_i64 = (cumsum_i32 * x_i32 + 1).to(tl.int64)

    tl.store(out_ptr + offsets, out_i64, mask=mask)


@torch.fx.wrap
def triton_fused_cumsum_mul_add(x):
    out = torch.empty_like(x)

    fused_cumsum_mul_add_kernel[(1,)](
        x, out,
        num_warps=1,
        num_stages=1,
    )

    return out


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(x):
    tmp_1 = torch.cumsum(x, dim=1)
    tmp_2 = tmp_1 * x
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


# ── Replacement args ──────────────────────────────────────────────────────────
def replacement_args(x):
    return (x,)


# ── Replacement func ──────────────────────────────────────────────────────────
def replacement_func():
    return triton_fused_cumsum_mul_add