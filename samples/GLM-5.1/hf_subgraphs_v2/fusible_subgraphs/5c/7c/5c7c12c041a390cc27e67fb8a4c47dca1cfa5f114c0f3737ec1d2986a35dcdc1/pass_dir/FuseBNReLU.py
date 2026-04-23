import torch
import triton
import triton.language as tl


def pattern(input, running_mean, running_var, weight, bias):
    bn_out = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    relu_out = torch.nn.functional.relu(bn_out, inplace=False)
    return relu_out


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)


@triton.jit
def fused_bn_relu_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    total_elements, hw, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load input - cast to float32 for precision
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute channel index for NCHW layout: channel = (flat_idx // HW) % C
    # Use safe offsets to ensure channel index is always in bounds
    safe_offsets = tl.where(mask, offsets, 0)
    chan_idx = (safe_offsets // hw) % C

    # Load BN parameters for corresponding channel - cast to float32
    mean_val = tl.load(mean_ptr + chan_idx, mask=mask, other=0.0).to(tl.float32)
    var_val = tl.load(var_ptr + chan_idx, mask=mask, other=1.0).to(tl.float32)
    w_val = tl.load(weight_ptr + chan_idx, mask=mask, other=1.0).to(tl.float32)
    b_val = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0).to(tl.float32)

    # Fused BN + ReLU: relu((x - mean) / sqrt(var + eps) * weight + bias)
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    normalized = (x - mean_val) * inv_std
    scaled = normalized * w_val + b_val
    result = tl.maximum(scaled, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_bn_relu(input, running_mean, running_var, weight, bias):
    eps = 0.001
    N, C, H, W = input.shape
    total_elements = input.numel()
    hw = H * W

    output = torch.empty_like(input)

    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_bn_relu_kernel[(num_programs,)](
        input_ptr=input,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        total_elements=total_elements,
        hw=hw,
        C=C,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_bn_relu