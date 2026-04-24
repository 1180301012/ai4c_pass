"""
Shared kernels and dispatch wrapper for all passes.
All pass files import dispatch_wrapper from here so they share the
SAME function object, satisfying output_pass_replacement_func_limit=1.

dispatch_wrapper(a0, a1, a2, a3, a4, route):
  route="scale_add"  -> a0=conv_out[B,C,H,W], a1=gamma[C,1,1], a2=residual[B,C,H,W]
  route="batch_norm" -> a0=x[B,C,H,W], a1=mean[C], a2=var[C], a3=weight[C], a4=bias[C]
"""
import torch
import triton
import triton.language as tl


# ── Kernel 1: fused element-wise scale + add (with autotune) ─────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['NCHW'],
)
@triton.jit
def _scale_add_kernel(
    x_ptr, gamma_ptr, r_ptr, out_ptr,
    NCHW, HW, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NCHW

    c_idx = (offs // HW) % C

    gamma  = tl.load(gamma_ptr + c_idx, mask=mask, other=0.0)
    x      = tl.load(x_ptr     + offs,  mask=mask, other=0.0)
    r      = tl.load(r_ptr     + offs,  mask=mask, other=0.0)

    gamma_f32 = gamma.to(tl.float32)
    x_f32     = x.to(tl.float32)
    r_f32     = r.to(tl.float32)

    out = r_f32 + x_f32 * gamma_f32

    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


# ── Kernel 2: batch-norm inference (precompute scale/shift for efficiency) ───
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['NCHW'],
)
@triton.jit
def _bn_infer_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    NCHW, HW, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NCHW

    c_idx = (offs // HW) % C

    x      = tl.load(x_ptr      + offs,   mask=mask, other=0.0)
    mean   = tl.load(mean_ptr   + c_idx,  mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(var_ptr    + c_idx,  mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_idx,  mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + c_idx,  mask=mask, other=0.0).to(tl.float32)

    # Precompute scale = w/sqrt(v+e) and shift = b - mean*scale
    # to reduce per-element operations
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale   = weight * inv_std
    shift   = bias - mean * scale

    x_f32 = x.to(tl.float32)
    out   = x_f32 * scale + shift

    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


# ── Shared dispatch wrapper ───────────────────────────────────────────────────
@torch.fx.wrap
def dispatch_wrapper(a0, a1, a2, a3, a4, route):
    """
    All passes share this single replacement function.
    Route "scale_add":  a0=conv_out, a1=gamma, a2=residual
    Route "batch_norm": a0=x, a1=mean, a2=var, a3=weight, a4=bias
    """
    if route == "scale_add":
        conv_out, gamma, residual = a0, a1, a2
        N, C, H, W = conv_out.shape
        NCHW = N * C * H * W
        HW   = H * W
        out  = torch.empty_like(conv_out)
        grid = lambda meta: (triton.cdiv(NCHW, meta['BLOCK_SIZE']),)
        _scale_add_kernel[grid](
            conv_out, gamma, residual, out,
            NCHW, HW, C,
        )
        return out
    elif route == "batch_norm":
        x, mean, var, weight, bias = a0, a1, a2, a3, a4
        N, C, H, W = x.shape
        NCHW = N * C * H * W
        HW   = H * W
        out  = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(NCHW, meta['BLOCK_SIZE']),)
        _bn_infer_kernel[grid](
            x, mean, var, weight, bias, out,
            NCHW, HW, C,
        )
        return out
    # Fallback: should never reach here
    return a0