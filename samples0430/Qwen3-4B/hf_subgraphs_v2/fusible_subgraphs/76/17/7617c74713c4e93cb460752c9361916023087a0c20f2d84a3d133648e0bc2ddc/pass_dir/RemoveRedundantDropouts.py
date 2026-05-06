import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    # Compute the reduction before dropouts
    sum_tensor = in_5 + in_4
    mean_tensor = sum_tensor.mean(dim=(2, 3), keepdim=False)
    # Batch normalization (dropouts are redundant with p=0.0)
    batch_norm = torch.nn.functional.batch_norm(
        mean_tensor,
        in_0,
        in_1,
        in_3,
        in_2,
        training=False,
        momentum=0.1,
        eps=1e-5
    )
    return (batch_norm, mean_tensor)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def batch_norm_triton_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
):
    pos = tl.program_id(0)
    if pos >= batch_size:
        return
    
    # Extract tensors
    input_tensor = tl.load(input_ptr + pos * channels)
    running_mean = tl.load(running_mean_ptr + pos * channels)
    running_var = tl.load(running_var_ptr + pos * channels)
    weight = tl.load(weight_ptr + pos * channels)
    bias = tl.load(bias_ptr + pos * channels)
    
    # Batch normalization computation
    normalized = (input_tensor - running_mean) / tl.sqrt(running_var + 1e-5)
    output = weight * normalized + bias
    
    tl.store(output_ptr + pos * channels, output)

@torch.fx.wrap
def batch_norm_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    batch_size, channels = in_0.shape[0], in_0.shape[1]
    output = torch.empty((batch_size, channels), dtype=in_0.dtype, device=in_0.device)
    grid = (batch_size,)
    batch_norm_triton_kernel[grid](
        input_ptr=in_0,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels
    )
    return (output, in_0)

def replacement_func():
    return batch_norm_wrapper