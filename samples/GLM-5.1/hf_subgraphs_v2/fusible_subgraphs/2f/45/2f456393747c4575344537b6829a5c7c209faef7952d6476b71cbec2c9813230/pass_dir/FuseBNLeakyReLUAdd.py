import torch
import triton
import triton.language as tl


def pattern(conv_output, running_mean, running_var, weight, bias, residual):
    bn_out = torch.nn.functional.batch_norm(conv_output, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    lrelu_out = torch.nn.functional.leaky_relu(bn_out, 0.01, True)
    add_out = lrelu_out + residual
    return (add_out,)


def replacement_args(conv_output, running_mean, running_var, weight, bias, residual):
    return (conv_output, running_mean, running_var, weight, bias, residual)


@triton.jit
def fused_bn_lrelu_add_kernel(
    conv_ptr, res_ptr, out_ptr,
    mean_ptr, var_ptr, w_ptr, b_ptr,
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)

    c = pid_c
    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)

    total_hw = N * HW
    hw_mask = hw_offs < total_hw

    n_idx = hw_offs // HW
    spatial_idx = hw_offs % HW

    # Load BN params once per channel per program - scalar loads, no mask needed
    mean_val = tl.load(mean_ptr + c).to(tl.float32)
    var_val = tl.load(var_ptr + c).to(tl.float32)
    w_val = tl.load(w_ptr + c).to(tl.float32)
    b_val = tl.load(b_ptr + c).to(tl.float32)

    # BN scale and shift
    inv_std = 1.0 / tl.sqrt(var_val + 1e-05)
    scale = w_val * inv_std
    shift = b_val - w_val * mean_val * inv_std

    # Compute flat offsets in NCHW contiguous layout
    offs = n_idx * C * HW + c * HW + spatial_idx

    # Load inputs
    conv_val = tl.load(conv_ptr + offs, mask=hw_mask, other=0.0).to(tl.float32)
    res_val = tl.load(res_ptr + offs, mask=hw_mask, other=0.0).to(tl.float32)

    # Fused: BN -> LeakyReLU -> Add
    result = scale * conv_val + shift
    result = tl.where(result >= 0.0, result, result * 0.01)
    result = result + res_val

    output_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + offs, result.to(output_dtype), mask=hw_mask)


@torch.fx.wrap
def fused_bn_lrelu_add(conv_output, running_mean, running_var, weight, bias, residual):
    device = conv_output.device
    if running_mean.device != device:
        running_mean = running_mean.to(device)
    if running_var.device != device:
        running_var = running_var.to(device)
    if weight.device != device:
        weight = weight.to(device)
    if bias.device != device:
        bias = bias.to(device)
    if residual.device != device:
        residual = residual.to(device)

    N, C, H, W = conv_output.shape
    HW = H * W

    output = torch.empty((N, C, H, W), dtype=conv_output.dtype, device=device)

    BLOCK_HW = 512
    grid_c = C
    grid_hw = (N * HW + BLOCK_HW - 1) // BLOCK_HW

    grid = (grid_c, grid_hw)

    fused_bn_lrelu_add_kernel[grid](
        conv_output, residual, output,
        running_mean, running_var, weight, bias,
        N, C, HW,
        BLOCK_HW=BLOCK_HW,
    )

    return output


def replacement_func():
    return fused_bn_lrelu_add