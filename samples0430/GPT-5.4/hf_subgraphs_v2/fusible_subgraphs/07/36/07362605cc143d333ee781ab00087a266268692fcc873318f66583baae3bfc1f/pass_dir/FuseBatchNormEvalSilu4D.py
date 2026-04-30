import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    tmp = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    out = torch.nn.functional.silu(tmp, inplace=True)
    return out


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.jit
def fused_bn_silu_eval_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid_m = tl.program_id(1)

    hw = H * W
    m = N * hw
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs_m < m

    n_idx = offs_m // hw
    rem = offs_m % hw
    h_idx = rem // W
    w_idx = rem % W

    x_ptrs = x_ptr + n_idx * stride_xn + c * stride_xc + h_idx * stride_xh + w_idx * stride_xw
    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    scale = weight / tl.sqrt(var + EPS)
    shift = bias - mean * scale
    y = x * scale + shift
    sig = 1.0 / (1.0 + tl.exp(-y))
    out = y * sig

    out_ptrs = out_ptr + n_idx * stride_on + c * stride_oc + h_idx * stride_oh + w_idx * stride_ow
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def fused_bn_silu_eval(x, running_mean, running_var, weight, bias):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    m = n * h * w

    if m <= 64:
        block_m = 64
        num_warps = 2
    elif m <= 128:
        block_m = 128
        num_warps = 4
    else:
        block_m = 256
        num_warps = 4

    out = torch.empty_like(x)
    grid = (c, triton.cdiv(m, block_m))
    fused_bn_silu_eval_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        n,
        h,
        w,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        EPS=1e-05,
        BLOCK_M=block_m,
        num_warps=num_warps,
    )
    return out


def replacement_func():
    return fused_bn_silu_eval