"""
Pass 2: BatchNorm (inference) + Residual Add fusion — residual-first add variant
Matches: residual + batch_norm(...)  (resnet10t graphs where in_6 += tmp_6)
Fuses BN normalization and residual add into a single Triton kernel.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total', 'C', 'HW'],
)
@triton.jit
def bn_add_fused_p2(
    x_ptr, res_ptr, mean_ptr, var_ptr, w_ptr, b_ptr, out_ptr,
    total, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    c    = (offs // HW) % C
    x    = tl.load(x_ptr   + offs, mask=mask, other=0.0).to(tl.float32)
    r    = tl.load(res_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c,   mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c,   mask=mask, other=1.0).to(tl.float32)
    bn_w = tl.load(w_ptr    + c,   mask=mask, other=1.0).to(tl.float32)
    bn_b = tl.load(b_ptr    + c,   mask=mask, other=0.0).to(tl.float32)
    out  = (bn_w * (x - mean) * tl.rsqrt(var + 1e-5) + bn_b).to(r.dtype) + r
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def bn_add_wrapper_p2(mean, var, weight, bias, x, residual):
    N, C, H, W = x.shape
    HW    = H * W
    total = N * C * HW
    out   = residual.new_empty(residual.shape)
    grid  = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    bn_add_fused_p2[grid](x, residual, mean, var, weight, bias, out, total, C, HW)
    return out


def pattern(mean, var, weight, bias, x, residual):
    """residual += bn(x,...)  — matches resnet10t graphs"""
    bn = torch.nn.functional.batch_norm(x, mean, var, weight, bias,
                                         False, 0.1, 1e-05)
    residual += bn
    return residual


def replacement_args(mean, var, weight, bias, x, residual):
    return (mean, var, weight, bias, x, residual)


def replacement_func():
    return bn_add_wrapper_p2