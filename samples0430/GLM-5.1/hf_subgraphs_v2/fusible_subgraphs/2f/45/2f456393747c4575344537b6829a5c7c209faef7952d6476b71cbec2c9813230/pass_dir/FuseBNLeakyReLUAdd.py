import torch
import triton
import triton.language as tl


def pattern(conv_output, running_mean, running_var, bn_weight, bn_bias, residual):
    bn_out = torch.nn.functional.batch_norm(conv_output, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    leaky_out = torch.nn.functional.leaky_relu(bn_out, 0.01, True)
    result = leaky_out + residual
    return result


def replacement_args(conv_output, running_mean, running_var, bn_weight, bn_bias, residual):
    return (conv_output, running_mean, running_var, bn_weight, bn_bias, residual)


@triton.jit
def fused_bn_leaky_relu_add_kernel(
    conv_ptr, residual_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    spatial_size,
    eps: tl.constexpr,
    neg_slope: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr,
):
    c_pid = tl.program_id(0)
    s_pid = tl.program_id(1)

    s_start = s_pid * SPATIAL_BLOCK
    s_offsets = s_start + tl.arange(0, SPATIAL_BLOCK)
    s_mask = s_offsets < spatial_size

    HW = H * W
    CHW = C * H * W

    n = s_offsets // HW
    hw_rem = s_offsets % HW
    h = hw_rem // W
    w = hw_rem % W

    flat_idx = n * CHW + c_pid * HW + h * W + w

    # Load and cast to fp32 for precision in BN computation
    conv_val = tl.load(conv_ptr + flat_idx, mask=s_mask, other=0.0).to(tl.float32)
    res_val = tl.load(residual_ptr + flat_idx, mask=s_mask, other=0.0).to(tl.float32)

    # Load BN params (one per channel, loaded once per program)
    mean_val = tl.load(mean_ptr + c_pid).to(tl.float32)
    var_val = tl.load(var_ptr + c_pid).to(tl.float32)
    w_val = tl.load(weight_ptr + c_pid).to(tl.float32)
    b_val = tl.load(bias_ptr + c_pid).to(tl.float32)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    bn_out = (conv_val - mean_val) * inv_std * w_val + b_val

    # LeakyReLU: x if x > 0, neg_slope * x if x <= 0
    leaky_out = tl.where(bn_out > 0.0, bn_out, bn_out * neg_slope)

    # Add residual
    add_out = leaky_out + res_val

    tl.store(out_ptr + flat_idx, add_out, mask=s_mask)


@torch.fx.wrap
def fused_bn_leaky_relu_add(conv_output, running_mean, running_var, bn_weight, bn_bias, residual):
    N, C, H, W = conv_output.shape
    spatial_size = N * H * W

    output = torch.empty_like(conv_output)

    SPATIAL_BLOCK = 1024
    num_spatial_programs = (spatial_size + SPATIAL_BLOCK - 1) // SPATIAL_BLOCK

    grid = (C, num_spatial_programs)

    fused_bn_leaky_relu_add_kernel[grid](
        conv_output, residual, running_mean, running_var, bn_weight, bn_bias, output,
        N, C, H, W,
        spatial_size,
        eps=1e-05,
        neg_slope=0.01,
        SPATIAL_BLOCK=SPATIAL_BLOCK,
    )

    return output


def replacement_func():
    return fused_bn_leaky_relu_add