"""Shared Triton kernels and single dispatch wrapper for all passes."""
import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm kernel  (eps=1e-6, bfloat16 output)  ── SmolLM3
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 2048}, num_warps=16),
        triton.Config({"BLOCK_D": 2048}, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_warps=4),
        triton.Config({"BLOCK_D": 2048}, num_warps=32),
    ],
    key=["D"],
    warmup=10,
    rep=10,
)
@triton.jit
def _rmsnorm_1e6_bf16_kernel(
    x_ptr, w_ptr, out_ptr,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    x_start = row * D
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    x   = tl.load(x_ptr + x_start + offsets, mask=mask, other=0.0).to(tl.float32)
    sq  = tl.sum(x * x, axis=0) / D
    rrms = tl.rsqrt(sq + 1e-06)
    xn  = x * rrms
    w   = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + x_start + offsets, (xn * w).to(tl.bfloat16), mask=mask)


def _do_rmsnorm_1e6_bf16(w, x):
    D = x.shape[-1]
    N = x.numel() // D
    out = torch.empty_like(x, dtype=torch.bfloat16)
    _rmsnorm_1e6_bf16_kernel[(N,)](x, w, out, D=D)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm kernel  (eps=1e-5, float32 output)  ── TinyLlama
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 2048}, num_warps=16),
        triton.Config({"BLOCK_D": 2048}, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_warps=4),
        triton.Config({"BLOCK_D": 2048}, num_warps=32),
    ],
    key=["D"],
    warmup=10,
    rep=10,
)
@triton.jit
def _rmsnorm_1e5_f32_kernel(
    x_ptr, w_ptr, out_ptr,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    x_start = row * D
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    x   = tl.load(x_ptr + x_start + offsets, mask=mask, other=0.0).to(tl.float32)
    sq  = tl.sum(x * x, axis=0) / D
    rrms = tl.rsqrt(sq + 1e-05)
    xn  = x * rrms
    w   = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + x_start + offsets, xn * w, mask=mask)   # float32 output


def _do_rmsnorm_1e5_f32(w, x):
    D = x.shape[-1]
    N = x.numel() // D
    out = torch.empty_like(x, dtype=torch.float32)
    _rmsnorm_1e5_f32_kernel[(N,)](x, w, out, D=D)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE cos/sin elementwise kernels  (input = cat result, any float dtype)
# Fuses:  cos/sin  →  ×1.0 (no-op)  →  cast
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _cos_bf16_elem_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, tl.cos(x).to(tl.bfloat16), mask=mask)


@triton.jit
def _sin_bf16_elem_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, tl.sin(x).to(tl.bfloat16), mask=mask)


@triton.jit
def _cos_f32_elem_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, tl.cos(x), mask=mask)


@triton.jit
def _sin_f32_elem_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, tl.sin(x), mask=mask)


def _do_cos_bf16(x):
    N = x.numel()
    BLOCK = 1024
    out = torch.empty_like(x, dtype=torch.bfloat16)
    _cos_bf16_elem_kernel[((N + BLOCK - 1) // BLOCK,)](x, out, N, BLOCK=BLOCK)
    return out


def _do_sin_bf16(x):
    N = x.numel()
    BLOCK = 1024
    out = torch.empty_like(x, dtype=torch.bfloat16)
    _sin_bf16_elem_kernel[((N + BLOCK - 1) // BLOCK,)](x, out, N, BLOCK=BLOCK)
    return out


def _do_cos_f32(x):
    N = x.numel()
    BLOCK = 1024
    out = torch.empty_like(x, dtype=torch.float32)
    _cos_f32_elem_kernel[((N + BLOCK - 1) // BLOCK,)](x, out, N, BLOCK=BLOCK)
    return out


def _do_sin_f32(x):
    N = x.numel()
    BLOCK = 1024
    out = torch.empty_like(x, dtype=torch.float32)
    _sin_f32_elem_kernel[((N + BLOCK - 1) // BLOCK,)](x, out, N, BLOCK=BLOCK)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Single shared dispatch wrapper returned by ALL pass files
# ═══════════════════════════════════════════════════════════════════════════════
@torch.fx.wrap
def shared_dispatch(*args):
    """Route to the correct kernel based on the trailing route-string argument."""
    route = args[-1]
    if route == "rmsnorm_1e6_bf16":
        return _do_rmsnorm_1e6_bf16(args[0], args[1])
    elif route == "rmsnorm_1e5_f32":
        return _do_rmsnorm_1e5_f32(args[0], args[1])
    elif route == "cos_bf16":
        return _do_cos_bf16(args[0])
    elif route == "sin_bf16":
        return _do_sin_bf16(args[0])
    elif route == "cos_f32":
        return _do_cos_f32(args[0])
    elif route == "sin_f32":
        return _do_sin_f32(args[0])
    # Should never reach here
    return args[0]