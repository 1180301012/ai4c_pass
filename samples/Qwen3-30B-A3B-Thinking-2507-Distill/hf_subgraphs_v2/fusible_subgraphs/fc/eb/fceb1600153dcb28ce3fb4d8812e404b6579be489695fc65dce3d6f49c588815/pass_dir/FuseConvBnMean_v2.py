import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    conv2d = torch.conv2d(in_6, in_5, in_4, (1, 1), (0, 0), (1, 1), 128)
    tmp_7 = in_7 + conv2d
    tmp_8 = tmp_7 + in_6
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    return tmp_9, tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}),
        triton.Config({'BLOCK_HW': 128}),
        triton.Config({'BLOCK_HW': 256}),
        triton.Config({'BLOCK_HW': 512}),
        triton.Config({'BLOCK_HW': 1024}),
        triton.Config({'BLOCK_HW': 2048}),
        triton.Config({'BLOCK_HW': 4096}),
        triton.Config({'BLOCK_HW': 8192}),
        triton.Config({'BLOCK_HW': 16384}),
    ],
    key=['HW'],
)
@triton.jit
def _fused_conv_bn_mean_kernel_v2(
    in6_ptr, in7_ptr,
    conv_w_ptr, conv_b_ptr,
    bn_rm_ptr, bn_rv_ptr, bn_w_ptr, bn_b_ptr,
    tmp9_ptr, mean_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    nc_idx = n * C + c

    # Load per-channel params, compute in fp32
    conv_w = tl.load(conv_w_ptr + c).to(tl.float32)
    conv_b = tl.load(conv_b_ptr + c).to(tl.float32)
    rm = tl.load(bn_rm_ptr + c).to(tl.float32)
    rv = tl.load(bn_rv_ptr + c).to(tl.float32)
    bn_w = tl.load(bn_w_ptr + c).to(tl.float32)
    bn_b = tl.load(bn_b_ptr + c).to(tl.float32)

    # Fused BN scale: bn_w / sqrt(rv + eps)
    bn_scale = bn_w / tl.sqrt(rv + 1e-5)
    # Fused BN bias: bn_b - rm * bn_scale
    bn_bias = bn_b - rm * bn_scale
    # Fused conv+BN scale: conv_w * bn_scale
    conv_bn_scale = conv_w * bn_scale
    # Fused conv+BN bias: conv_b * bn_scale + bn_bias
    conv_bn_bias = conv_b * bn_scale + bn_bias

    base = nc_idx * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    x2 = tl.load(in6_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in7_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused computation: bn(conv(x2) + x1 + x2)
    tmp = conv_bn_scale * x2 + conv_bn_bias + x1

    tl.store(tmp9_ptr + base + offsets, tmp.to(tmp9_ptr.dtype.element_ty), mask=mask)

    # Spatial mean (masked elements are 0 so they don't affect the sum)
    mean_val = tl.sum(tmp, axis=0) / HW
    tl.store(mean_ptr + nc_idx, mean_val.to(mean_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_conv_bn_mean_v2(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # in_0: bn_running_mean [C]
    # in_1: bn_running_var  [C]
    # in_2: bn_bias         [C]
    # in_3: bn_weight       [C]
    # in_4: conv_bias       [C]
    # in_5: conv_weight     [C, 1, 1, 1]
    # in_6: x1              [N, C, H, W]  (added first: in_7 + conv2d)
    # in_7: x2              [N, C, H, W]  (added second: conv2d + in_6)
    N, C, H, W = in_7.shape
    HW = H * W

    tmp9 = torch.empty_like(in_7)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_7.dtype, device=in_7.device)

    grid = (N, C)
    _fused_conv_bn_mean_kernel_v2[grid](
        in_6, in_7,
        in_5, in_4,
        in_0, in_1, in_3, in_2,
        tmp9, mean_out,
        C, HW,
    )

    return tmp9, mean_out


def replacement_func():
    return fused_conv_bn_mean_v2