import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    in_4 += in_5
    in_6 = in_4
    tmp_5 = torch.nn.functional.gelu(in_6, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit

def fused_add_gelu_bn_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_gelu_ptr,
    out_bn_ptr,
    NC,
    C,
    HW,
    W,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    y_s0,
    y_s1,
    y_s2,
    y_s3,
    og_s0,
    og_s1,
    og_s2,
    og_s3,
    ob_s0,
    ob_s1,
    ob_s2,
    ob_s3,
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    n = pid_nc // C
    c = pid_nc % C
    h = offs_hw // W
    w = offs_hw % W

    x_offs = n * x_s0 + c * x_s1 + h * x_s2 + w * x_s3
    y_offs = n * y_s0 + c * y_s1 + h * y_s2 + w * y_s3
    og_offs = n * og_s0 + c * og_s1 + h * og_s2 + w * og_s3
    ob_offs = n * ob_s0 + c * ob_s1 + h * ob_s2 + w * ob_s3

    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offs, mask=mask, other=0.0)

    # Match the original in-place add semantics by rounding in the native tensor dtype
    summed_native = x + y
    summed = summed_native.to(tl.float32)

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    gelu = 0.5 * summed * (1.0 + tl.erf(summed * 0.7071067811865476))

    # batch_norm consumes tmp_5, so use the rounded/stored dtype value semantically
    gelu_native = gelu.to(x.dtype)
    gelu_for_bn = gelu_native.to(tl.float32)

    scale = weight * tl.rsqrt(var + eps)
    shift = bias - mean * scale
    bn = gelu_for_bn * scale + shift

    tl.store(out_gelu_ptr + og_offs, gelu_native, mask=mask)
    tl.store(out_bn_ptr + ob_offs, bn, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    shape = in_4.shape
    c = shape[1]
    w = shape[3]
    hw = shape[2] * shape[3]
    nc = shape[0] * shape[1]

    out_gelu = torch.empty_like(in_4)
    out_bn = torch.empty_like(in_4)

    x_s0, x_s1, x_s2, x_s3 = in_4.stride()
    y_s0, y_s1, y_s2, y_s3 = in_5.stride()
    og_s0, og_s1, og_s2, og_s3 = out_gelu.stride()
    ob_s0, ob_s1, ob_s2, ob_s3 = out_bn.stride()

    grid = lambda meta: (nc, triton.cdiv(hw, meta["BLOCK_HW"]))

    fused_add_gelu_bn_kernel[grid](
        in_4,
        in_5,
        in_0,
        in_1,
        in_2,
        in_3,
        out_gelu,
        out_bn,
        nc,
        c,
        hw,
        w,
        x_s0,
        x_s1,
        x_s2,
        x_s3,
        y_s0,
        y_s1,
        y_s2,
        y_s3,
        og_s0,
        og_s1,
        og_s2,
        og_s3,
        ob_s0,
        ob_s1,
        ob_s2,
        ob_s3,
        1e-05,
    )

    return (out_gelu, out_bn)


def replacement_func():
    return fused_add_gelu_bn