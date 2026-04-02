import torch
import triton
import triton.language as tl
import operator

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16},  num_warps=1),
        triton.Config({'BLOCK_C': 32},  num_warps=1),
    ],
    key=['N', 'C_val'],
)
@triton.jit
def _fused_ln_relu_kernel_19(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_val, eps,
    BLOCK_C: tl.constexpr,
):
    pid   = tl.program_id(0)
    c_off = tl.arange(0, BLOCK_C)
    mask  = c_off < C_val
    x    = tl.load(x_ptr + pid * C_val + c_off, mask=mask, other=0.0).to(tl.float32)
    x_m  = tl.where(mask, x, 0.0)
    mean = tl.sum(x_m) / C_val
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff) / C_val
    norm = diff * tl.rsqrt(var + eps)
    w    = tl.load(w_ptr + c_off, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + c_off, mask=mask, other=0.0).to(tl.float32)
    out  = tl.maximum(norm * w + b, 0.0)
    tl.store(out_ptr + pid * C_val + c_off, out, mask=mask)


def pattern(conv_out, ln_weight, ln_bias):
    ln_out   = torch.ops.aten.layer_norm.default(conv_out, [19, 1, 1], ln_weight, ln_bias, 1e-05, True)
    relu_out = torch.ops.aten.relu.default(ln_out)
    return relu_out


def replacement_args(conv_out, ln_weight, ln_bias):
    return (conv_out, ln_weight, ln_bias)


@torch.fx.wrap
def fused_ln_relu_19(conv_out, ln_weight, ln_bias):
    N   = conv_out.shape[0]
    C   = conv_out.shape[1]
    xf  = conv_out.contiguous().view(N, C)
    wf  = ln_weight.contiguous().view(C)
    bf  = ln_bias.contiguous().view(C)
    out = torch.empty((N, C), dtype=torch.float32, device=conv_out.device)
    _fused_ln_relu_kernel_19[(N,)](xf, wf, bf, out, N, C, 1e-5)
    return out.to(conv_out.dtype).view_as(conv_out)


def replacement_func():
    return fused_ln_relu_19