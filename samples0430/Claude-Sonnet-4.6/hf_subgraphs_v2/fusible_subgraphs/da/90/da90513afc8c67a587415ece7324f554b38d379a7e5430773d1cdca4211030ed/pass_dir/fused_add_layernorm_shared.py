import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Kernel for N=768  (BLOCK_N=1024, masking for elements 768..1023)
# N and eps are compile-time constants → fewer runtime args → less Python overhead
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_add_layernorm_768_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    M,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    N    = 768   # compile-time constant
    eps  = 1e-5  # compile-time constant

    row_idx  = tl.program_id(0)
    offsets  = tl.arange(0, BLOCK_N)
    mask     = offsets < N
    row_start = row_idx * N

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    sum_z  = tl.sum(z,     axis=0)
    sum_z2 = tl.sum(z * z, axis=0)
    mean   = sum_z / N
    var    = sum_z2 / N - mean * mean
    var    = tl.maximum(var, 0.0)
    rstd   = 1.0 / tl.sqrt(var + eps)
    z_n    = (z - mean) * rstd

    w   = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    out = z_n * w + b

    if IS_FP16:
        tl.store(out_ptr + row_start + offsets, out.to(tl.float16),  mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row_start + offsets, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row_start + offsets, out, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel for N=1024  (BLOCK_N=1024, no masking)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_add_layernorm_1024_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    M,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    N   = 1024
    eps = 1e-5

    row_idx   = tl.program_id(0)
    offsets   = tl.arange(0, BLOCK_N)
    row_start = row_idx * N

    x = tl.load(x_ptr + row_start + offsets).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets).to(tl.float32)
    z = x + y

    sum_z  = tl.sum(z,     axis=0)
    sum_z2 = tl.sum(z * z, axis=0)
    mean   = sum_z / N
    var    = sum_z2 / N - mean * mean
    var    = tl.maximum(var, 0.0)
    rstd   = 1.0 / tl.sqrt(var + eps)
    z_n    = (z - mean) * rstd

    w   = tl.load(weight_ptr + offsets).to(tl.float32)
    b   = tl.load(bias_ptr   + offsets).to(tl.float32)
    out = z_n * w + b

    if IS_FP16:
        tl.store(out_ptr + row_start + offsets, out.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + row_start + offsets, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row_start + offsets, out)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel for N=16  (BLOCK_N=16, no masking)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_add_layernorm_16_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    M,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    N   = 16
    eps = 1e-5

    row_idx   = tl.program_id(0)
    offsets   = tl.arange(0, BLOCK_N)
    row_start = row_idx * N

    x = tl.load(x_ptr + row_start + offsets).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets).to(tl.float32)
    z = x + y

    sum_z  = tl.sum(z,     axis=0)
    sum_z2 = tl.sum(z * z, axis=0)
    mean   = sum_z / N
    var    = sum_z2 / N - mean * mean
    var    = tl.maximum(var, 0.0)
    rstd   = 1.0 / tl.sqrt(var + eps)
    z_n    = (z - mean) * rstd

    w   = tl.load(weight_ptr + offsets).to(tl.float32)
    b   = tl.load(bias_ptr   + offsets).to(tl.float32)
    out = z_n * w + b

    if IS_FP16:
        tl.store(out_ptr + row_start + offsets, out.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + row_start + offsets, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row_start + offsets, out)


# ──────────────────────────────────────────────────────────────────────────────
# Single shared dispatch wrapper – returned by ALL pass files' replacement_func()
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def dispatch_fused_add_layernorm(bias, weight, x, y, route):
    if route == "route_768":
        N = 768
        M = x.numel() // N
        out = torch.empty_like(x)
        IS_FP16 = x.dtype == torch.float16
        IS_BF16 = x.dtype == torch.bfloat16
        _fused_add_layernorm_768_kernel[(M,)](
            x, y, weight, bias, out,
            M,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            BLOCK_N=1024,
            num_warps=16,
            num_stages=1,
        )
        return out
    elif route == "route_1024":
        N = 1024
        M = x.numel() // N
        out = torch.empty_like(x)
        IS_FP16 = x.dtype == torch.float16
        IS_BF16 = x.dtype == torch.bfloat16
        _fused_add_layernorm_1024_kernel[(M,)](
            x, y, weight, bias, out,
            M,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            BLOCK_N=1024,
            num_warps=16,
            num_stages=1,
        )
        return out
    elif route == "route_16":
        N = 16
        M = x.numel() // N
        out = torch.empty_like(x)
        IS_FP16 = x.dtype == torch.float16
        IS_BF16 = x.dtype == torch.bfloat16
        _fused_add_layernorm_16_kernel[(M,)](
            x, y, weight, bias, out,
            M,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
            BLOCK_N=16,
            num_warps=1,
            num_stages=1,
        )
        return out
    # Fallback (should never be reached)
    return x + y