import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_7):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_7):
    return (in_0, in_1, in_2, in_3, in_7)


@triton.jit
def _batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    C,
    stride_xm,
    stride_xc,
    stride_om,
    stride_oc,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < C

    x = tl.load(x_ptr + pid_m * stride_xm + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + eps)
    y = y * weight + bias

    tl.store(out_ptr + pid_m * stride_om + cols * stride_oc, y, mask=mask)


@torch.fx.wrap
def triton_batch_norm_inference(mean, var, bias, weight, x):
    m = x.shape[0]
    c = x.shape[1]
    out = torch.empty((m, c), device=x.device, dtype=x.dtype)
    _batch_norm_inference_kernel[(m, triton.cdiv(c, 128))](
        x,
        mean,
        var,
        weight,
        bias,
        out,
        m,
        c,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        1e-5,
        BLOCK_C=128,
    )
    return out


def replacement_func():
    return triton_batch_norm_inference