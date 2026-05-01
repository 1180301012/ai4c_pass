import torch
import triton
import triton.language as tl

def pattern(in_7, in_0, in_1, in_3, in_2):
    return torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)

def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    channels,
    eps: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_channels = tl.program_id(1)
    
    sum = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    
    for start_batch in range(0, batch, BLOCK_BATCH):
        batch_indices = start_batch + tl.arange(0, BLOCK_BATCH)
        mask = batch_indices < batch
        x = tl.load(x_ptr + batch_indices[:, None] * channels + pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS), 
                   mask=mask[:, None], 
                   other=0.0)
        sum += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    
    mean = sum / batch
    var = sum_sq / batch - mean * mean
    var = tl.sqrt(var + eps)
    
    for start_batch in range(0, batch, BLOCK_BATCH):
        batch_indices = start_batch + tl.arange(0, BLOCK_BATCH)
        mask = batch_indices < batch
        x = tl.load(x_ptr + batch_indices[:, None] * channels + pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS), 
                   mask=mask[:, None], 
                   other=0.0)
        x_normalized = (x - mean[None, :]) / var[None, :]
        weight = tl.load(weight_ptr + pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS), 
                        mask=pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS) < channels)
        bias = tl.load(bias_ptr + pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS), 
                      mask=pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS) < channels)
        x_normalized = x_normalized * weight[None, :] + bias[None, :]
        tl.store(out_ptr + batch_indices[:, None] * channels + pid_channels * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS), 
                x_normalized, 
                mask=mask[:, None])

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    batch, channels = x.shape
    
    BLOCK_BATCH = 64
    BLOCK_CHANNELS = 64
    
    grid = (triton.cdiv(batch, BLOCK_BATCH), triton.cdiv(channels, BLOCK_CHANNELS))
    
    out = torch.empty_like(x)
    batch_norm_kernel[grid](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch=batch,
        channels=channels,
        eps=1e-05,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_CHANNELS=BLOCK_CHANNELS
    )
    
    return out

def replacement_func():
    return optimized_batch_norm