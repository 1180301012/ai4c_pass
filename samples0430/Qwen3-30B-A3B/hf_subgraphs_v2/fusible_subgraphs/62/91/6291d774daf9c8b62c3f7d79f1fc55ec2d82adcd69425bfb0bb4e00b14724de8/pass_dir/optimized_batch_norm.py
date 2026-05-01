import torch
import triton
import triton.language as tl

def pattern(x, mean, var, weight, bias, training, momentum, eps):
    return torch.nn.functional.batch_norm(x, mean, var, weight, bias, training, momentum, eps)

def replacement_args(x, mean, var, weight, bias, training, momentum, eps):
    return (x, mean, var, weight, bias, training, momentum, eps)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    block_start = batch_id * BLOCK_SIZE
    batch_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    batch_mask = batch_offsets < batch_size

    mean_val = tl.load(mean_ptr + channel_id)
    var_val = tl.load(var_ptr + channel_id)
    weight_val = tl.load(weight_ptr + channel_id)
    bias_val = tl.load(bias_ptr + channel_id)
    var_sqrt = tl.sqrt(var_val + eps)

    for i in range(BLOCK_SIZE):
        if not batch_mask[i]:
            break
        input_idx = batch_offsets[i] * channels + channel_id
        output_idx = batch_offsets[i] * channels + channel_id
        
        input_val = tl.load(input_ptr + input_idx)
        normalized = (input_val - mean_val) / var_sqrt
        output_val = normalized * weight_val + bias_val
        tl.store(output_ptr + output_idx, output_val)

@torch.fx.wrap
def batch_norm_kernel_wrapper(x, mean, var, weight, bias, training, momentum, eps):
    batch_size = x.shape[0]
    channels = x.shape[1]
    
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    num_batches = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_channels = channels
    
    batch_norm_kernel[(num_batches, num_channels), 1](
        x,
        mean,
        var,
        weight,
        bias,
        output,
        batch_size,
        channels,
        eps,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return batch_norm_kernel_wrapper