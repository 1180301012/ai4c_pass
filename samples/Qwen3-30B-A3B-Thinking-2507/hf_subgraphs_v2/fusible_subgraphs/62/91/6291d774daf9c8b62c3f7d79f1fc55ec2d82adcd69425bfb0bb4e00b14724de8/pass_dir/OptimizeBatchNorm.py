import torch
import triton
import triton.language as tl
def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)
@triton.jit
def batch_norm_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr, batch_size: tl.constexpr, num_channels: tl.constexpr, eps: tl.float32, BLOCK_SIZE: tl.constexpr):
    c = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    weight_val = tl.load(weight_ptr + c)
    bias_val = tl.load(bias_ptr + c)
    input_ptr += c
    output_ptr += c
    input_vals = tl.load(input_ptr + offsets * num_channels, mask=mask, other=0.0)
    normalized = (input_vals - mean) / tl.sqrt(var + eps)
    output_vals = normalized * weight_val + bias_val
    tl.store(output_ptr + offsets * num_channels, output_vals, mask=mask)
@torch.fx.wrap
def batch_norm_wrapper(x, running_mean, running_var, weight, bias):
    batch_size = x.shape[0]
    num_channels = x.shape[1]
    eps = 1e-5
    BLOCK_SIZE = 1024
    num_batches = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_channels, num_batches)
    output = torch.empty_like(x)
    batch_norm_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        batch_size,
        num_channels,
        eps,
        BLOCK_SIZE
    )
    return output
def replacement_func():
    return batch_norm_wrapper