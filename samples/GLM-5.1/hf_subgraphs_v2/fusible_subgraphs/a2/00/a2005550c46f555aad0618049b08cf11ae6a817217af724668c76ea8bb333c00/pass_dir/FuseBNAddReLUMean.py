import torch
import triton
import triton.language as tl


def pattern(running_mean, running_var, bias, weight, bn_input, add_input):
    tmp_0 = running_mean
    tmp_1 = running_var
    tmp_2 = bias
    tmp_3 = weight
    tmp_4 = torch.nn.functional.batch_norm(bn_input, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
    tmp_5 = add_input + tmp_4
    tmp_4 = None
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_5 = None
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(running_mean, running_var, bias, weight, bn_input, add_input):
    return (running_mean, running_var, bias, weight, bn_input, add_input)


@triton.jit
def fused_bn_add_relu_mean_kernel(
    bn_input_ptr, add_input_ptr, out_relu_ptr, out_mean_ptr,
    scale_ptr, shift_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
    N_elements: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    # Load scale and shift for this channel block
    scale = tl.load(scale_ptr + c_offsets, mask=c_mask, other=0.0)
    shift = tl.load(shift_ptr + c_offsets, mask=c_mask, other=0.0)

    # Initialize accumulators for mean
    # mean over spatial dimensions: sum over H*W then divide by H*W
    sum_vals = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Loop over spatial dimensions
    hw_start_base = 0
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW

    for hw_block in range(num_hw_blocks):
        hw_start = hw_block * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        # Compute full offsets into flattened [N, C, H*W] tensor
        # For element at (n, c, hw): offset = n * C * HW + c * HW + hw
        full_offsets = pid_n * C * HW + c_offsets[:, None] * HW + hw_offsets[None, :]
        full_mask = c_mask[:, None] & hw_mask[None, :]

        # Load bn_input and add_input
        bn_val = tl.load(bn_input_ptr + full_offsets, mask=full_mask, other=0.0).to(tl.float32)
        add_val = tl.load(add_input_ptr + full_offsets, mask=full_mask, other=0.0).to(tl.float32)

        # BN: scale * bn_input + shift
        bn_out = bn_val * scale[:, None] + shift[:, None]

        # Add
        add_out = add_val + bn_out

        # ReLU
        relu_out = tl.where(add_out > 0.0, add_out, 0.0)

        # Accumulate for mean
        # Only sum valid elements
        valid_relu = tl.where(full_mask, relu_out, 0.0)
        sum_vals += tl.sum(valid_relu, axis=1)

        # Store relu output
        out_dtype = out_relu_ptr.dtype.element_ty
        tl.store(out_relu_ptr + full_offsets, relu_out.to(out_dtype), mask=full_mask)

    # Compute mean for each channel
    mean_vals = sum_vals / HW

    # Store mean output: shape [N, C, 1, 1], offset = n * C * 1 * 1 + c * 1 * 1 + 0
    mean_offsets = pid_n * C + c_offsets
    out_mean_dtype = out_mean_ptr.dtype.element_ty
    tl.store(out_mean_ptr + mean_offsets, mean_vals.to(out_mean_dtype), mask=c_mask)


@torch.fx.wrap
def fused_bn_add_relu_mean(running_mean, running_var, bias, weight, bn_input, add_input):
    # Compute scale and shift from BN parameters (eval mode)
    eps = 1e-05
    # BN in eval mode: output = weight * (input - running_mean) / sqrt(running_var + eps) + bias
    # Simplify: output = input * (weight / sqrt(running_var + eps)) + (bias - running_mean * weight / sqrt(running_var + eps))
    inv_std = 1.0 / torch.sqrt(running_var.float() + eps)
    scale = weight.float() * inv_std
    shift = bias.float() - running_mean.float() * weight.float() * inv_std

    # Match input dtype
    scale = scale.to(bn_input.dtype)
    shift = shift.to(bn_input.dtype)

    N, C, H, W = bn_input.shape
    HW = H * W

    # Allocate outputs
    out_relu = torch.empty_like(bn_input)
    out_mean = torch.empty((N, C, 1, 1), dtype=bn_input.dtype, device=bn_input.device)

    BLOCK_C = C if C <= 32 else 32
    BLOCK_HW = 256

    grid = (triton.cdiv(C, BLOCK_C), N)

    fused_bn_add_relu_mean_kernel[grid](
        bn_input_ptr=bn_input,
        add_input_ptr=add_input,
        out_relu_ptr=out_relu,
        out_mean_ptr=out_mean,
        scale_ptr=scale,
        shift_ptr=shift,
        C=C, H=H, W=W, HW=HW,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
        N_elements=N * C * HW,
    )

    return (out_relu, out_mean)


def replacement_func():
    return fused_bn_add_relu_mean