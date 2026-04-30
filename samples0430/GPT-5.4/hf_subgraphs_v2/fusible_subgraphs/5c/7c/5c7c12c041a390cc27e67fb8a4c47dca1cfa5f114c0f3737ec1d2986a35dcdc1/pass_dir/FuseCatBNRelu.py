import torch
import triton
import triton.language as tl


def pattern(cat_other, upsampled, running_mean, running_var, bias, weight):
    tmp_0 = torch.cat([cat_other, upsampled], 1)
    tmp_1 = torch.nn.functional.batch_norm(tmp_0, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
    return (tmp_2,)


def replacement_args(cat_other, upsampled, running_mean, running_var, bias, weight):
    return (cat_other, upsampled, running_mean, running_var, bias, weight)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_cat_bn_relu_kernel(
    cat_other_ptr,
    upsampled_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    cat_c,
    total_c,
    out_h,
    out_w,
    upsampled_c,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    hw = out_h * out_w
    chw = total_c * hw
    cat_chw = cat_c * hw
    up_chw = upsampled_c * hw

    n = offs // chw
    rem = offs - n * chw
    c = rem // hw
    rem2 = rem - c * hw
    h = rem2 // out_w
    w = rem2 - h * out_w

    mean = tl.load(mean_ptr + c, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1).to(tl.float32)

    cat_mask = mask & (c < cat_c)
    up_mask = mask & (c >= cat_c)

    cat_offs = n * cat_chw + c * hw + h * out_w + w
    x_cat = tl.load(cat_other_ptr + cat_offs, mask=cat_mask, other=0).to(tl.float32)

    uc = tl.where(c >= cat_c, c - cat_c, 0)
    up_offs = n * up_chw + uc * hw + h * out_w + w
    x_up = tl.load(upsampled_ptr + up_offs, mask=up_mask, other=0).to(tl.float32)

    x = tl.where(c < cat_c, x_cat, x_up)
    y = (x - mean) * tl.math.rsqrt(var + eps) * weight + bias
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_cat_bn_relu(cat_other, upsampled, running_mean, running_var, bias, weight):
    n = cat_other.shape[0]
    cat_c = cat_other.shape[1]
    out_h = cat_other.shape[2]
    out_w = cat_other.shape[3]
    upsampled_c = upsampled.shape[1]
    total_c = running_mean.shape[0]
    out = torch.empty((n, total_c, out_h, out_w), device=cat_other.device, dtype=cat_other.dtype)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_cat_bn_relu_kernel[grid](
        cat_other,
        upsampled,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        n_elements,
        cat_c,
        total_c,
        out_h,
        out_w,
        upsampled_c,
        0.001,
    )
    return (out,)


def replacement_func():
    return fused_cat_bn_relu