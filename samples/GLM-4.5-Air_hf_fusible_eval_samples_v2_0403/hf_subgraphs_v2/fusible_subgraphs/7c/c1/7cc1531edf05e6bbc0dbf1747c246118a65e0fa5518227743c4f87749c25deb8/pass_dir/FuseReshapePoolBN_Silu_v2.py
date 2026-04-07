import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    h_in,
    w_in,
    eps: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    hw_in = h_in * w_in
    hw_in_block = tl.program_id(2)
    
    out_h = hw_in_block // w_in
    out_w = hw_in_block % w_in
    
    row_offset = batch_idx * channels * h_in * w_in + channel_idx * h_in * w_in + out_h * w_in + out_w
    row_offset_2x = batch_idx * channels * (h_in * 2) * (w_in * 2) + channel_idx * (h_in * 2) * (w_in * 2)
    
    h0 = 2 * out_h
    w0 = 2 * out_w
    
    sum_val = 0.0
    for dh in tl.static_range(2):
        for dw in tl.static_range(2):
            h = h0 + dh
            w = w0 + dw
            flat_idx = row_offset_2x + h * (w_in * 2) + w
            x = tl.load(input_ptr + flat_idx)
            sum_val = sum_val + x
    
    pooled = sum_val / 4.0
    
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    x = pooled - running_mean
    x = x / tl.sqrt(running_var + eps)
    x = x * weight + bias
    
    exp_neg_x = tl.exp(-x)
    out = x / (1.0 + exp_neg_x)
    
    tl.store(out_ptr + row_offset, out)


@torch.fx.wrap
def fused_kernel(
    input_tensor,
    running_mean,
    running_var,
    weight,
    bias,
    eps=1e-5,
):
    batch_size = input_tensor.shape[0]
    channels = input_tensor.shape[1]
    h_in_2 = input_tensor.shape[2]
    w_in_2 = input_tensor.shape[3]
    h_in = h_in_2 // 2
    w_in = w_in_2 // 2
    
    out_shape = (batch_size, channels, h_in, w_in)
    out = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    grid = (batch_size, channels, h_in * w_in)
    fused_kernel_kernel[grid](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        batch_size,
        channels,
        h_in,
        w_in,
        eps,
    )
    
    return out


def pattern(input_tensor, running_mean, running_var, weight, bias):
    reshaped = input_tensor.reshape(1, 512, 16, 16)
    pooled = torch.nn.functional.avg_pool2d(reshaped, 2, 2, 0, False, True, None)
    normalized = torch.nn.functional.batch_norm(
        pooled, running_mean, running_var, weight, bias, False, 0.1, 1e-5
    )
    activated = torch.nn.functional.silu(normalized, inplace=True)
    return activated


def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_kernel