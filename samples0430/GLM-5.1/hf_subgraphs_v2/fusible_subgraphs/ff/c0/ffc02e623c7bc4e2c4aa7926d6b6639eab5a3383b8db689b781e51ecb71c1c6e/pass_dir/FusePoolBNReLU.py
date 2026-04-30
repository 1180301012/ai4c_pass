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
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_channels,
    channels,
    HW: tl.constexpr,
    eps: tl.constexpr,
    is_float16: tl.constexpr,
    is_bfloat16: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    if bc_idx >= batch_channels:
        return

    batch_idx = bc_idx // channels
    channel_idx = bc_idx % channels

    # Load spatial data and convert to float32 for computation
    hw_offsets = tl.arange(0, HW)
    input_offset = batch_idx * channels * HW + channel_idx * HW
    spatial_data = tl.load(input_ptr + input_offset + hw_offsets).to(tl.float32)

    # Compute average over spatial dimensions (adaptive_avg_pool2d to (1,1))
    avg = tl.sum(spatial_data) / float(HW)

    # Load batch norm parameters and convert to float32
    ch_mean = tl.load(mean_ptr + channel_idx).to(tl.float32)
    ch_var = tl.load(var_ptr + channel_idx).to(tl.float32)
    ch_weight = tl.load(weight_ptr + channel_idx).to(tl.float32)
    ch_bias = tl.load(bias_ptr + channel_idx).to(tl.float32)

    # Batch norm: (avg - running_mean) / sqrt(running_var + eps) * weight + bias
    normalized = (avg - ch_mean) / tl.sqrt(ch_var + eps)
    bn_result = normalized * ch_weight + ch_bias

    # ReLU
    result_f32 = tl.max(bn_result, 0.0)

    # Convert back to original dtype for storage
    if is_float16:
        result = result_f32.to(tl.float16)
    elif is_bfloat16:
        result = result_f32.to(tl.bfloat16)
    else:
        result = result_f32

    # Store output at (batch, channel, 0, 0)
    output_offset = batch_idx * channels + channel_idx
    tl.store(output_ptr + output_offset, result)


@torch.fx.wrap
def fused_pool_bn_relu(input, running_mean, running_var, weight, bias):
    batch_size = input.shape[0]
    channels = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    HW = H * W

    # Create output tensor with same dtype and device as input
    output = torch.empty((batch_size, channels, 1, 1), dtype=input.dtype, device=input.device)

    batch_channels = batch_size * channels
    grid = (batch_channels,)

    is_float16 = input.dtype == torch.float16
    is_bfloat16 = input.dtype == torch.bfloat16

    fused_pool_bn_relu_kernel[grid](
        input_ptr=input,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_channels=batch_channels,
        channels=channels,
        HW=HW,
        eps=1e-05,
        is_float16=is_float16,
        is_bfloat16=is_bfloat16,
    )

    return output


def replacement_func():
    return fused_pool_bn_relu