import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_6, conv2d):
    tmp_7 = torch.cat([in_6, conv2d], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.prelu(tmp_8, in_0)
    return tmp_9



def replacement_args(in_0, in_1, in_2, in_3, in_4, in_6, conv2d):
    return (in_6, conv2d, in_1, in_2, in_4, in_3, in_0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=2),
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _fused_cat_bn_prelu_kernel(
    in6_ptr,
    conv_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    prelu_ptr,
    out_ptr,
    HW,
    EPS,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    c = pid_nc % 128
    n = pid_nc // 128

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    src_c = c
    use_in6 = c < 64
    src_c = tl.where(use_in6, src_c, src_c - 64)

    src_plane_base = (n * 64 + src_c) * HW
    out_plane_base = (n * 128 + c) * HW

    x_in6 = tl.load(in6_ptr + src_plane_base + offs_hw, mask=mask, other=0.0)
    x_conv = tl.load(conv_ptr + src_plane_base + offs_hw, mask=mask, other=0.0)
    x = tl.where(use_in6, x_in6, x_conv).to(tl.float32)

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)
    alpha = tl.load(prelu_ptr + c).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + EPS)
    y = y * gamma + beta
    y = tl.where(y >= 0, y, y * alpha)

    tl.store(out_ptr + out_plane_base + offs_hw, y, mask=mask)


@torch.fx.wrap
def fused_cat_bn_prelu(in6, conv2d, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    n = in6.shape[0]
    h = in6.shape[2]
    w = in6.shape[3]
    hw = h * w
    out = torch.empty((n, 128, h, w), device=in6.device, dtype=in6.dtype)

    grid = (triton.cdiv(hw, 256), n * 128)
    _fused_cat_bn_prelu_kernel[grid](
        in6,
        conv2d,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        prelu_weight,
        out,
        hw,
        0.001,
    )
    return out



def replacement_func():
    return fused_cat_bn_prelu