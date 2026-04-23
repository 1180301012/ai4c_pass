import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_4 + in_5
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_5, tmp_6)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit

def _fused_add_gelu_bn_kernel(
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    x_ptr,
    y_ptr,
    gelu_out_ptr,
    bn_out_ptr,
    C,
    HW,
    total_nc,
    EPS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    if pid_nc >= total_nc:
        return

    c = pid_nc % C
    base = pid_nc * HW
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW
    offs = base + offs_hw

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    gelu = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865475))

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + EPS)
    bn = (gelu - mean) * inv_std * weight + bias

    tl.store(gelu_out_ptr + offs, gelu, mask=mask)
    tl.store(bn_out_ptr + offs, bn, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn(mean, var, bias, weight, x, y):
    out_gelu = torch.empty_like(x)
    out_bn = torch.empty_like(x)

    shape = x.shape
    C = shape[1]
    HW = shape[2] * shape[3]
    total_nc = shape[0] * C

    grid = lambda meta: (triton.cdiv(HW, meta['BLOCK_HW']), total_nc)
    _fused_add_gelu_bn_kernel[grid](
        mean,
        var,
        bias,
        weight,
        x,
        y,
        out_gelu,
        out_bn,
        C,
        HW,
        total_nc,
        EPS=1e-5,
    )
    return (out_gelu, out_bn)


def replacement_func():
    return fused_add_gelu_bn