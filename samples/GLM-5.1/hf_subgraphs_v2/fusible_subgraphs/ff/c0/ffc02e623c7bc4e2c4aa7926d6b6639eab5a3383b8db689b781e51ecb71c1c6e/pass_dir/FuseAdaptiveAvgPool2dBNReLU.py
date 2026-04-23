import torch
import triton
import triton.language as tl


def pattern(in_5, in_1, in_2, in_4, in_3):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8


def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)


@triton.jit
def fused_pool_bn_relu_kernel(
    input_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    output_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    if n >= N or c >= C:
        return

    # Compute mean of spatial elements for (n, c)
    # Input is [N, C, H, W] contiguous
    spatial_base = n * C * H * W + c * H * W
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < H * W
    input_vals = tl.load(input_ptr + spatial_base + hw_offsets, mask=hw_mask, other=0.0).to(tl.float32)

    # Compute mean in float32 for numerical stability
    total = tl.sum(input_vals)
    mean_val = total / (H * W)

    # Load BN parameters for channel c, cast to float32 for computation
    rm = tl.load(bn_mean_ptr + c).to(tl.float32)
    rv = tl.load(bn_var_ptr + c).to(tl.float32)
    w = tl.load(bn_weight_ptr + c).to(tl.float32)
    b = tl.load(bn_bias_ptr + c).to(tl.float32)

    # BN in eval mode: weight * (input - running_mean) / sqrt(running_var + eps) + bias
    bn_out = w * (mean_val - rm) / tl.sqrt(rv + eps) + b

    # ReLU
    result = tl.maximum(bn_out, 0.0)

    # Store to output [N, C, 1, 1] - contiguous layout: offset = n * C + c
    tl.store(output_ptr + n * C + c, result)


@torch.fx.wrap
def fused_pool_bn_relu(input, running_mean, running_var, weight, bias):
    N, C, H, W = input.shape
    eps = 1e-05
    BLOCK_HW = triton.next_power_of_2(H * W)

    output = torch.empty(N, C, 1, 1, dtype=input.dtype, device=input.device)

    grid = (N, C)
    fused_pool_bn_relu_kernel[grid](
        input_ptr=input,
        bn_mean_ptr=running_mean,
        bn_var_ptr=running_var,
        bn_weight_ptr=weight,
        bn_bias_ptr=bias,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        eps=eps,
        BLOCK_HW=BLOCK_HW,
    )

    return output


def replacement_func():
    return fused_pool_bn_relu