import torch
import triton
import triton.language as tl

# Fused Triton kernel: add + layer_norm in single pass per row
# Each program handles one row, computes sum in registers,
# then computes mean/var/normalize without re-reading from memory
@triton.jit
def fused_add_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    sum_out_ptr, norm_out_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs (upcast to float32 for numerical stability)
    in_2 = tl.load(in_2_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute sum
    sum_val = in_2 + in_3

    # Store sum to output buffer (for tmp_2)
    tl.store(sum_out_ptr + row_start + col_offsets, sum_val, mask=mask)

    # Compute mean
    mean = tl.sum(sum_val, axis=0) / n_cols

    # Compute variance using E[X^2] - (E[X])^2
    var = tl.sum(sum_val * sum_val, axis=0) / n_cols - mean * mean

    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    norm_val = (sum_val - mean) * rstd

    # Load weight and bias (shared across all rows)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    result = norm_val * weight + bias

    # Store result (for tmp_4)
    tl.store(norm_out_ptr + row_start + col_offsets, result, mask=mask)


# Autotuned single-pass kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_add_layernorm_kernel_autotune(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    sum_out_ptr, norm_out_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load inputs (upcast to float32 for numerical stability)
    in_2 = tl.load(in_2_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute sum
    sum_val = in_2 + in_3

    # Store sum to output buffer (for tmp_2)
    tl.store(sum_out_ptr + row_start + col_offsets, sum_val, mask=mask)

    # Compute mean
    mean = tl.sum(sum_val, axis=0) / n_cols

    # Compute variance using E[X^2] - (E[X])^2
    var = tl.sum(sum_val * sum_val, axis=0) / n_cols - mean * mean

    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    norm_val = (sum_val - mean) * rstd

    # Load weight and bias (shared across all rows)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    result = norm_val * weight + bias

    # Store result (for tmp_4)
    tl.store(norm_out_ptr + row_start + col_offsets, result, mask=mask)


# Core implementation using autotuned kernels
@torch.fx.wrap
def _fused_add_layernorm_impl(in_0, in_1, in_2, in_3):
    n_cols = in_2.shape[-1]
    n_rows = in_2.numel() // n_cols

    sum_out = torch.empty_like(in_2)
    norm_out = torch.empty_like(in_2)

    grid = (n_rows,)

    # Use single-pass autotuned kernel (BLOCK_SIZE >= n_cols via autotune configs)
    fused_add_layernorm_kernel_autotune[grid](
        in_2, in_3, in_1, in_0,
        sum_out, norm_out,
        n_rows, n_cols,
        eps=1e-05,
    )

    return sum_out, norm_out


# Dispatch wrapper (shared across all pass files for replacement_func_limit)
@torch.fx.wrap
def fused_add_layernorm_dispatch(in_0, in_1, in_2, in_3, route):
    sum_out, norm_out = _fused_add_layernorm_impl(in_0, in_1, in_2, in_3)
    if route == "sum_first":
        return (sum_out, norm_out)
    elif route == "norm_first":
        return (norm_out, sum_out)
    else:
        raise ValueError(f"Unknown route: {route}")


# Shared replacement_func - must be identical across all pass files
def replacement_func():
    return fused_add_layernorm_dispatch