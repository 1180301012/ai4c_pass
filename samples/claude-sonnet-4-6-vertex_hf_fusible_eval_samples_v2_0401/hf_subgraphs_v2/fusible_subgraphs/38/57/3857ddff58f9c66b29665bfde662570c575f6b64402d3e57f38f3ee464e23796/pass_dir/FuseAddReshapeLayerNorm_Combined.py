"""
Combined fused pass: add + reshape + layer_norm → single Triton kernel.
Handles any N dynamically by dispatching to an appropriately-sized kernel.
The pattern uses in_0.shape[0] so it can match both N=768 and N=16 graphs.
"""
import torch
import triton
import triton.language as tl


# ── small-N kernel (BLOCK_SIZE=32, suitable for N≤32) ────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['N', 'num_rows'],
)
@triton.jit
def fused_add_ln_small_kernel(
    in2_ptr, in3_ptr,
    weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    num_rows,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x2_raw = tl.load(in2_ptr + row_start + offsets, mask=mask, other=0.0)
    x3_raw = tl.load(in3_ptr + row_start + offsets, mask=mask, other=0.0)
    input_dtype = x2_raw.dtype

    x = x2_raw.to(tl.float32) + x3_raw.to(tl.float32)

    # Round to input dtype before storing (and before LN stats)
    x_rounded = x.to(input_dtype)
    tl.store(out_add_ptr + row_start + offsets, x_rounded, mask=mask)

    # LN on the rounded values — matches what PyTorch layer_norm reads
    x_ln = x_rounded.to(tl.float32)
    mean = tl.sum(x_ln, axis=0) / N
    x_c = x_ln - mean
    x_sq = tl.where(mask, x_c * x_c, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    out_ln = x_c * rstd * w + b
    tl.store(out_ln_ptr + row_start + offsets, out_ln.to(input_dtype), mask=mask)


# ── large-N kernel (BLOCK_SIZE=1024, suitable for N=768) ─────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N', 'num_rows'],
)
@triton.jit
def fused_add_ln_large_kernel(
    in2_ptr, in3_ptr,
    weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    num_rows,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x2_raw = tl.load(in2_ptr + row_start + offsets, mask=mask, other=0.0)
    x3_raw = tl.load(in3_ptr + row_start + offsets, mask=mask, other=0.0)
    input_dtype = x2_raw.dtype

    x = x2_raw.to(tl.float32) + x3_raw.to(tl.float32)

    # Round to input dtype before storing (and before LN stats)
    x_rounded = x.to(input_dtype)
    tl.store(out_add_ptr + row_start + offsets, x_rounded, mask=mask)

    # LN on the rounded values — matches what PyTorch layer_norm reads
    x_ln = x_rounded.to(tl.float32)
    mean = tl.sum(x_ln, axis=0) / N
    x_c = x_ln - mean
    x_sq = tl.where(mask, x_c * x_c, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    out_ln = x_c * rstd * w + b
    tl.store(out_ln_ptr + row_start + offsets, out_ln.to(input_dtype), mask=mask)


# ── opaque wrapper — one FX node ─────────────────────────────────────────────

@torch.fx.wrap
def _fused_add_layernorm_impl(in_0, in_1, in_2, in_3):
    """
    Dispatch to the right kernel based on runtime N = in_0.shape[0].
    Returns (out_add [rows, N], out_ln [rows, N]).
    """
    N = in_0.shape[0]
    num_rows = in_2.numel() // N

    out_add = torch.empty(num_rows, N, dtype=in_2.dtype, device=in_2.device)
    out_ln  = torch.empty(num_rows, N, dtype=in_2.dtype, device=in_2.device)

    if N <= 32:
        fused_add_ln_small_kernel[(num_rows,)](
            in_2, in_3, in_1, in_0, out_add, out_ln,
            num_rows=num_rows, N=N, eps=1e-5,
        )
    else:
        fused_add_ln_large_kernel[(num_rows,)](
            in_2, in_3, in_1, in_0, out_add, out_ln,
            num_rows=num_rows, N=N, eps=1e-5,
        )

    return (out_add, out_ln)


# ── outer replacement (NOT wrapped): traced by FX → 2 getitem nodes ──────────

def fused_add_layernorm_combined(in_0, in_1, in_2, in_3):
    result = _fused_add_layernorm_impl(in_0, in_1, in_2, in_3)
    return result[0], result[1]


# ── pattern / replacement interface ──────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    """
    Dynamic pattern: reshape dimension = in_0.shape[0] (N).
    This allows a single pattern to match any hidden-state size.
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, in_0.shape[0])
    tmp_4 = torch.nn.functional.layer_norm(
        tmp_3, in_0.shape, in_1, in_0, 1e-05
    )
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_layernorm_combined