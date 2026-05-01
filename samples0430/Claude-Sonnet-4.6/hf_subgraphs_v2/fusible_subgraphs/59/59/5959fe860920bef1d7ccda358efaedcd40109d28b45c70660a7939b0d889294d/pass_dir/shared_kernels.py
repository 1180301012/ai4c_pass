import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────
# Flat cos kernel – bfloat16 output
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _cos_cast_bf16_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.cos(x).to(tl.bfloat16), mask=mask)


# ──────────────────────────────────────────────────────────────
# Flat sin kernel – bfloat16 output
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _sin_cast_bf16_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.sin(x).to(tl.bfloat16), mask=mask)


# ──────────────────────────────────────────────────────────────
# Flat cos kernel – float32 output
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _cos_cast_fp32_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.cos(x), mask=mask)


# ──────────────────────────────────────────────────────────────
# Flat sin kernel – float32 output
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _sin_cast_fp32_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.sin(x), mask=mask)


# ──────────────────────────────────────────────────────────────
# RMSNorm kernel – bfloat16 output, eps = 1e-6
# Single-pass: load x and w once, compute RMS, normalize, store
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def _rmsnorm_bf16_1e6_kernel(x_ptr, w_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Single load of x and w (keeps values in registers)
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    # Compute inv_rms using cached x
    inv_rms = tl.rsqrt(tl.sum(x * x) / n_cols + 1e-6)

    # Normalize, scale and store as bfloat16
    tl.store(out_row + cols, (x * inv_rms * w).to(tl.bfloat16), mask=mask)


# ──────────────────────────────────────────────────────────────
# RMSNorm kernel – float32 output, eps = 1e-5
# Single-pass: load x and w once, compute RMS, normalize, store
# ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def _rmsnorm_fp32_1e5_kernel(x_ptr, w_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Single load of x and w
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    # Compute inv_rms using cached x
    inv_rms = tl.rsqrt(tl.sum(x * x) / n_cols + 1e-5)

    # Normalize, scale and store as float32
    tl.store(out_row + cols, x * inv_rms * w, mask=mask)


# ──────────────────────────────────────────────────────────────
# Private implementation helpers
# ──────────────────────────────────────────────────────────────
def _impl_rope_cos_bf16(catted):
    n = catted.numel()
    out = torch.empty(catted.shape, dtype=torch.bfloat16, device=catted.device)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _cos_cast_bf16_kernel[grid](catted, out, n)
    return out


def _impl_rope_sin_bf16(catted):
    n = catted.numel()
    out = torch.empty(catted.shape, dtype=torch.bfloat16, device=catted.device)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sin_cast_bf16_kernel[grid](catted, out, n)
    return out


def _impl_rope_cos_fp32(catted):
    n = catted.numel()
    out = torch.empty(catted.shape, dtype=torch.float32, device=catted.device)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _cos_cast_fp32_kernel[grid](catted, out, n)
    return out


def _impl_rope_sin_fp32(catted):
    n = catted.numel()
    out = torch.empty(catted.shape, dtype=torch.float32, device=catted.device)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sin_cast_fp32_kernel[grid](catted, out, n)
    return out


def _impl_rmsnorm_bf16_1e6(weight, x):
    H = x.shape[-1]
    n_rows = x.numel() // H
    out = torch.empty_like(x)
    _rmsnorm_bf16_1e6_kernel[(n_rows,)](x, weight, out, H)
    return out


def _impl_rmsnorm_fp32_1e5(weight, x):
    H = x.shape[-1]
    n_rows = x.numel() // H
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _rmsnorm_fp32_1e5_kernel[(n_rows,)](x, weight, out, H)
    return out


# ──────────────────────────────────────────────────────────────
# Shared dispatch wrapper (returned by ALL pass files)
# Each pass appends a route string as the last arg.
# ──────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_dispatch(a, b, route):
    if route == "rope_cos_bf16":
        return _impl_rope_cos_bf16(a)
    elif route == "rope_sin_bf16":
        return _impl_rope_sin_bf16(a)
    elif route == "rope_cos_fp32":
        return _impl_rope_cos_fp32(a)
    elif route == "rope_sin_fp32":
        return _impl_rope_sin_fp32(a)
    elif route == "rmsnorm_bf16_1e6":
        return _impl_rmsnorm_bf16_1e6(a, b)
    elif route == "rmsnorm_fp32_1e5":
        return _impl_rmsnorm_fp32_1e5(a, b)
    # Fallback (never reached)
    return _impl_rmsnorm_bf16_1e6(a, b)