import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused scale * gamma + residual_add + inference BN
#
# x     : conv output or dropout output  [N, C, H, W]  (NCHW contiguous)
# gamma : layer-scale  [C, 1, 1]           (broadcast over N, H, W)
# resid : residual input               [N, C, H, W]
# bn_w  : BN weight               [C]
# bn_b  : BN bias                 [C]
# bn_m  : BN running_mean         [C]
# bn_v  : BN running_var          [C]
#
# Key observation for NCHW layout:
#   flat_index = n*C*HW + c*HW + hw
#   channel(c)  = (flat_index % HW) // HW
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_full', 'C', 'HW'],
)
@triton.jit
def fused_scale_add_bn_kernel(
    x_ptr, gamma_ptr, resid_ptr,
    bn_w_ptr, bn_b_ptr, bn_mean_ptr, bn_var_ptr,
    out_ptr,          # output [N,C,H,W] = inference BN result
    N_full, C, HW,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N_full * C * HW

    # channel index in NCHW layout
    c_idx = (offsets % HW) // HW

    # loads — promote to fp32 for numerical stability
    x       = tl.load(x_ptr     + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma   = tl.load(gamma_ptr + c_idx,   mask=mask, other=1.0).to(tl.float32)
    in7     = tl.load(resid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bn_w    = tl.load(bn_w_ptr  + c_idx,   mask=mask, other=1.0).to(tl.float32)
    bn_b    = tl.load(bn_b_ptr  + c_idx,   mask=mask, other=0.0).to(tl.float32)
    bn_mean = tl.load(bn_mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    bn_var  = tl.load(bn_var_ptr + c_idx,  mask=mask, other=1.0).to(tl.float32)

    # fused computation
    tmp9  = x * gamma
    tmp10 = in7 + tmp9
    inv_std = 1.0 / tl.sqrt(bn_var + 1e-5)
    # inference BN: (x - mean) / sqrt(var + eps) * weight + bias
    out   = (tmp10 - bn_mean) * inv_std * bn_w + bn_b

    if IS_FP16:
        tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Single-output Python wrapper (must return ONE value, not a tuple)
#
# Design note: dropout(conv, p=0, training=False) is a mathematical no-op.
# We accept `dropped` (the output after dropout) as the pattern's free
# variable `x`, so the pattern matches both:
#   1. with dropout:  tmp_8 * gamma + residual  → then BN
#   2. without dropout: tmp_7 * gamma + residual → then BN
# (because the SubgraphMatcher allows `x` to match ANY node feeding into
#  the first op, including the dropout-output or directly the conv-output)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_scale_add_bn(x, gamma, residual, bn_weight, bn_bias, bn_mean, bn_var):
    N  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W
    NC = N * C

    out      = torch.empty_like(x)
    is_fp16  = (x.dtype == torch.float16)
    is_bf16  = (x.dtype == torch.bfloat16)

    grid = lambda meta: (triton.cdiv(NC * HW, meta['BLOCK_SIZE']),)

    fused_scale_add_bn_kernel[grid](
        x, gamma, residual,
        bn_weight, bn_bias, bn_mean, bn_var,
        out,
        NC, C, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )
    return out

# ---------------------------------------------------------------------------
# Pattern / replacement interface
#
# NOTE: SINGLE-OUTPUT pattern — the framework wraps the replacement in exactly
#       one opaque node via with_dispatch_wrapper_run, so the number of
#       returning_nodes in the replacement must equal the number of nodes in
#       match.returning_nodes.
#
# Pattern: match only the batch_norm (that covers tmp_11 in the model return).
# `dropped` is a free variable that binds to the conv2d / dropout output node
# so the pattern matches regardless of whether dropout(x, 0.0, False, False)
# is present in the graph or not.
# ---------------------------------------------------------------------------

def pattern(dropped, bn_mean, bn_var, bn_weight, bn_bias):
    result = torch.nn.functional.batch_norm(
        dropped, bn_mean, bn_var, bn_weight, bn_bias,
        False, 0.1, 1e-05
    )
    return result


def replacement_args(dropped, bn_mean, bn_var, bn_weight, bn_bias):
    return (dropped, bn_mean, bn_var, bn_weight, bn_bias)


# ---------------------------------------------------------------------------
# Triton kernel: inference-only BN  (dropped = pre-computed sum)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def bn_inference_only_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N * C * HW

    # channel index via NCHW stride arithmetic
    # offsets = n*C*HW + c*HW + hw
    # c = (offsets - n*C*HW) // HW  where n = offsets//(C*HW)
    chan_block_start = (offsets // (C * HW)) * (C * HW)
    c_idx    = (offsets - chan_block_start) // HW

    x   = tl.load(x_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    mu  = tl.load(mean_ptr + c_idx,  mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr  + c_idx,  mask=mask, other=1.0).to(tl.float32)
    w   = tl.load(weight_ptr + c_idx,mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(bias_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    out     = (x - mu) * inv_std * w + b

    if IS_FP16:
        tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def bn_inference_only(x, mean, var, weight, bias):
    N  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W
    NC = N * C

    out      = torch.empty_like(x)
    is_fp16  = (x.dtype == torch.float16)
    is_bf16  = (x.dtype == torch.bfloat16)

    grid = lambda meta: (triton.cdiv(NC * HW, meta['BLOCK_SIZE']),)

    bn_inference_only_kernel[grid](
        x, mean, var, weight, bias, out,
        NC, C, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )
    return out


def replacement_func():
    return bn_inference_only