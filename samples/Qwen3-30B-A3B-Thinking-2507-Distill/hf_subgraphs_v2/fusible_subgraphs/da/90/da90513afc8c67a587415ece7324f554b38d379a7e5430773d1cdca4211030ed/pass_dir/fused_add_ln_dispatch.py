"""
Shared fused Add + LayerNorm dispatch wrapper.
All three pass files (768 / 1024 / 16) import this single function
so the framework sees only ONE unique replacement_func and loads all passes.
"""
import torch
import triton
import triton.language as tl


# ── Kernel for N=768 (BLOCK_N=1024, with masking) ────────────────────────────
@triton.jit
def _fused_add_ln_768_kernel(
    in2_ptr, in3_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    N: tl.constexpr,        # 768
    eps,
    BLOCK_N: tl.constexpr,  # 1024
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x2 = tl.load(in2_ptr + row * N + cols, mask=mask, other=0.0)
    x3 = tl.load(in3_ptr + row * N + cols, mask=mask, other=0.0)
    x  = x2 + x3

    x_f32 = x.to(tl.float32)
    mean  = tl.sum(x_f32, axis=0) / N

    xc   = tl.where(mask, x_f32 - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)

    out_f32 = xc * rstd * w + b
    tl.store(out_ptr + row * N + cols, out_f32.to(x.dtype), mask=mask)


# ── Kernel for N=1024 (BLOCK_N=1024, exact fit) ───────────────────────────────
@triton.jit
def _fused_add_ln_1024_kernel(
    in2_ptr, in3_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    N: tl.constexpr,        # 1024
    eps,
    BLOCK_N: tl.constexpr,  # 1024
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    x2 = tl.load(in2_ptr + row * N + cols)
    x3 = tl.load(in3_ptr + row * N + cols)
    x  = x2 + x3

    x_f32 = x.to(tl.float32)
    mean  = tl.sum(x_f32, axis=0) / N

    xc   = x_f32 - mean
    var  = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols).to(tl.float32)
    b = tl.load(bias_ptr   + cols).to(tl.float32)

    out_f32 = xc * rstd * w + b
    tl.store(out_ptr + row * N + cols, out_f32.to(x.dtype))


# ── Kernel for N=16 (BLOCK_N=16, exact fit) ──────────────────────────────────
@triton.jit
def _fused_add_ln_16_kernel(
    in2_ptr, in3_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    N: tl.constexpr,        # 16
    eps,
    BLOCK_N: tl.constexpr,  # 16
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    x2 = tl.load(in2_ptr + row * N + cols)
    x3 = tl.load(in3_ptr + row * N + cols)
    x  = x2 + x3

    x_f32 = x.to(tl.float32)
    mean  = tl.sum(x_f32, axis=0) / N

    xc   = x_f32 - mean
    var  = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols).to(tl.float32)
    b = tl.load(bias_ptr   + cols).to(tl.float32)

    out_f32 = xc * rstd * w + b
    tl.store(out_ptr + row * N + cols, out_f32.to(x.dtype))


# ── Shared dispatch wrapper (single replacement_func for all 3 passes) ───────
@torch.fx.wrap
def fused_add_layernorm_dispatch(in_0, in_1, in_2, in_3, route):
    """
    in_0 : bias  [N]
    in_1 : weight [N]
    in_2 : tensor1 [*, N]
    in_3 : tensor2 [*, N]
    route: "768" | "1024" | "16"
    """
    out = torch.empty_like(in_2)

    if route == "768":
        N, BLOCK_N = 768, 1024
        M    = in_2.numel() // N
        eps  = 1e-5
        _fused_add_ln_768_kernel[(M,)](
            in_2, in_3, in_1, in_0, out,
            N=N, eps=eps, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
    elif route == "1024":
        N, BLOCK_N = 1024, 1024
        M    = in_2.numel() // N
        eps  = 1e-5
        _fused_add_ln_1024_kernel[(M,)](
            in_2, in_3, in_1, in_0, out,
            N=N, eps=eps, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
    elif route == "16":
        N, BLOCK_N = 16, 16
        M    = in_2.numel() // N
        eps  = 1e-5
        _fused_add_ln_16_kernel[(M,)](
            in_2, in_3, in_1, in_0, out,
            N=N, eps=eps, BLOCK_N=BLOCK_N,
            num_warps=1,
        )

    return out