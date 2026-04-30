import torch
import triton
import triton.language as tl


def pattern(conv_out, layer_scale, running_mean, running_var, bn_bias, bn_weight, residual):
    tmp_8 = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_9 = tmp_8 * layer_scale
    tmp_10 = residual + tmp_9
    tmp_11 = torch.nn.functional.batch_norm(
        tmp_10, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05
    )
    return (tmp_11, tmp_10)


def replacement_args(conv_out, layer_scale, running_mean, running_var, bn_bias, bn_weight, residual):
    return (conv_out, layer_scale, running_mean, running_var, bn_bias, bn_weight, residual)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit

def fused_dropout_mul_add_batchnorm_eval_kernel(
    conv_ptr,
    residual_ptr,
    layer_scale_ptr,
    bn_scale_ptr,
    bn_bias_ptr,
    out_bn_ptr,
    out_pre_bn_ptr,
    n_elements,
    hw,
    c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    channel_idx = (offsets // hw) % c

    conv_val = tl.load(conv_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual_val = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    layer_scale_val = tl.load(layer_scale_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    bn_scale_val = tl.load(bn_scale_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    bn_bias_val = tl.load(bn_bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)

    pre_bn = residual_val + conv_val * layer_scale_val
    bn_out = pre_bn * bn_scale_val + bn_bias_val

    tl.store(out_pre_bn_ptr + offsets, pre_bn, mask=mask)
    tl.store(out_bn_ptr + offsets, bn_out, mask=mask)


@torch.fx.wrap
def fused_dropout_mul_add_batchnorm_eval(
    conv_out,
    layer_scale,
    running_mean,
    running_var,
    bn_bias,
    bn_weight,
    residual,
):
    if conv_out.ndim != 4 or (not conv_out.is_contiguous()) or (not residual.is_contiguous()):
        layer_scale_4d = layer_scale.reshape(1, -1, 1, 1)
        bn_scale_4d = bn_weight.reshape(1, -1, 1, 1) * (running_var.reshape(1, -1, 1, 1) + 1e-05).rsqrt()
        bn_bias_4d = bn_bias.reshape(1, -1, 1, 1) - running_mean.reshape(1, -1, 1, 1) * bn_scale_4d
        out_pre_bn = residual + conv_out * layer_scale_4d
        out_bn = out_pre_bn * bn_scale_4d + bn_bias_4d
        return out_bn, out_pre_bn

    layer_scale_1d = layer_scale.reshape(-1)
    bn_scale_1d = bn_weight * (running_var + 1e-05).rsqrt()
    bn_bias_1d = bn_bias - running_mean * bn_scale_1d

    out_bn = torch.empty_like(conv_out)
    out_pre_bn = torch.empty_like(conv_out)

    n_elements = conv_out.numel()
    c = conv_out.shape[1]
    hw = conv_out.shape[2] * conv_out.shape[3]

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_dropout_mul_add_batchnorm_eval_kernel[grid](
        conv_out,
        residual,
        layer_scale_1d,
        bn_scale_1d,
        bn_bias_1d,
        out_bn,
        out_pre_bn,
        n_elements,
        hw,
        c,
    )
    return out_bn, out_pre_bn


def replacement_func():
    return fused_dropout_mul_add_batchnorm_eval