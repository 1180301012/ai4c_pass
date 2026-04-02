import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────
# Generic pattern: normalized_shape as a wildcard placeholder.
# ──────────────────────────────────────────────────────────────
def pattern(x, normalized_shape, weight, bias):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, 1e-12)


def replacement_args(x, normalized_shape, weight, bias):
    return (x, weight, bias)


# ──────────────────────────────────────────────────────────────
# Kernel 1 – small N (≤128): 2D parallel multi-row tile
#   Rows processed simultaneously by different warps.
#   Grid = (ceil(M / ROWS_PER_BLOCK),)
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32,  "ROWS_PER_BLOCK": 4},  num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 32,  "ROWS_PER_BLOCK": 8},  num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 64,  "ROWS_PER_BLOCK": 4},  num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 128, "ROWS_PER_BLOCK": 4},  num_warps=8),
    ],
    key=["N", "M"],
)
@triton.jit
def _ln_kernel_2d_small(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, stride_x, stride_y, eps,
    BLOCK_SIZE_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    base_row  = block_id * ROWS_PER_BLOCK

    # 2D index arrays (broadcast-friendly shapes)
    rows = tl.arange(0, ROWS_PER_BLOCK)[:, None]   # [R, 1]
    cols = tl.arange(0, BLOCK_SIZE_N)[None, :]      # [1, N]

    row_mask  = (base_row + rows) < M               # [R, 1]
    col_mask  = cols < N                            # [1, N]
    full_mask = row_mask & col_mask                 # [R, N]

    # 2D pointer arrays
    X_ptrs = X_ptr + (base_row + rows) * stride_x + cols   # [R, N]
    Y_ptrs = Y_ptr + (base_row + rows) * stride_y + cols   # [R, N]

    # Load full tile [R, N] in one operation
    x = tl.load(X_ptrs, mask=full_mask, other=0.0).to(tl.float32)

    # Weight / bias [1, N]
    w = tl.load(W_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    # Per-row mean [R, 1]  (masked zeros do not bias the sum)
    mean = tl.sum(x, axis=1, keep_dims=True) / N

    # Per-row variance [R, 1]
    x_c  = tl.where(full_mask, x - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=1, keep_dims=True) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Scale + shift [R, N]
    y = x_c * rstd * w + b

    # Store (invalid rows / columns masked out)
    tl.store(Y_ptrs, y, mask=full_mask)


# ──────────────────────────────────────────────────────────────
# Kernel 2 – medium N (≤512, e.g. N=384): one row per block
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _ln_kernel_medium(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, stride_x, stride_y, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row   = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Y_row = Y_ptr + row * stride_y

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    x = tl.load(X_row + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offsets,  mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets,  mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_c  = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y    = x_c * rstd * w + b

    tl.store(Y_row + offsets, y, mask=mask)


# ──────────────────────────────────────────────────────────────
# Kernel 3 – large N (>512, e.g. N=768): one row per block
# Only use BLOCK_SIZE=1024 (2048 wastes 63% bandwidth for N=768)
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _ln_kernel_large(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, stride_x, stride_y, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row   = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Y_row = Y_ptr + row * stride_y

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    x = tl.load(X_row + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offsets,  mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets,  mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_c  = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y    = x_c * rstd * w + b

    tl.store(Y_row + offsets, y, mask=mask)


# ──────────────────────────────────────────────────────────────
# Python wrapper – dispatches to the right kernel based on N
# ──────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_layer_norm_generic(x, weight, bias):
    N = weight.numel()
    M = x.numel() // N

    orig_shape = x.shape
    x_flat = x.contiguous().view(M, N)
    y_flat = torch.empty_like(x_flat)

    sx = x_flat.stride(0)
    sy = y_flat.stride(0)

    if N <= 128:
        # 2D parallel multi-row kernel: rows processed simultaneously by warps
        grid = lambda meta: (triton.cdiv(M, meta['ROWS_PER_BLOCK']),)
        _ln_kernel_2d_small[grid](
            X_ptr=x_flat, W_ptr=weight, B_ptr=bias, Y_ptr=y_flat,
            M=M, N=N, stride_x=sx, stride_y=sy, eps=1e-12,
        )
    elif N <= 512:
        _ln_kernel_medium[(M,)](
            X_ptr=x_flat, W_ptr=weight, B_ptr=bias, Y_ptr=y_flat,
            M=M, N=N, stride_x=sx, stride_y=sy, eps=1e-12,
        )
    else:
        _ln_kernel_large[(M,)](
            X_ptr=x_flat, W_ptr=weight, B_ptr=bias, Y_ptr=y_flat,
            M=M, N=N, stride_x=sx, stride_y=sy, eps=1e-12,
        )

    return y_flat.view(orig_shape)


# ──────────────────────────────────────────────────────────────
# Required entry-point
# ──────────────────────────────────────────────────────────────
def replacement_func():
    return triton_layer_norm_generic