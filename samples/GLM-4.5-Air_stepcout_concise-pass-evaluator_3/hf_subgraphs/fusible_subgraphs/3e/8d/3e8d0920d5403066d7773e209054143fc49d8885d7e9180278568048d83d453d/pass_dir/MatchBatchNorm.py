import torch
import triton
import triton.language as tl

def pattern(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3):
    # Batch norm operation - this should be in all graphs
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    return tmp_8

def replacement_args(tmp_7, tmp_1, tmp_2, tmp_4, tmp_3):
    return (tmp_7, tmp_1, tmp_2, tmp_4, tmp_3)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_n,
    input_c,
    input_h,
    input_w,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output coordinates for single-element per thread
    output_idx = pid
    
    # Check bounds
    if output_idx >= input_n * input_c * input_h * input_w:
        return
    
    # Flatten index to get batch, channel, h, w
    batch = output_idx // (input_c * input_h * input_w)
    remaining = output_idx % (input_c * input_h * input_w)
    channel_idx = remaining // (input_h * input_w)
    remaining = remaining % (input_h * input_w)
    h = remaining // input_w
    w = remaining % input_w
    
    # Load input value
    input_idx = batch * input_c * input_h * input_w + channel_idx * input_h * input_w + h * input_w + w
    input_val = tl.load(input_ptr + input_idx)
    
    # Load batch norm parameters
    mean_val = tl.load(running_mean_ptr + channel_idx)
    var_val = tl.load(running_var_ptr + channel_idx)
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Batch norm formula: (x - running_mean) / sqrt(running_var + eps) * weight + bias
    denom = tl.sqrt(var_val + eps)
    normalized = (input_val - mean_val) / denom
    batch_norm_result = normalized * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + output_idx, batch_norm_result)

@torch.fx.wrap
def optimized_batch_norm(input, running_mean, running_var, weight, bias):
    input_n, input_c, input_h, input_w = input.shape
    
    output = torch.empty((input_n, input_c, input_h, input_w), dtype=input.dtype, device=input.device)
    
    total_elements = input_n * input_c * input_h * input_w
    grid_size = total_elements
    
    batch_norm_kernel[(grid_size,)](
        input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        input_n,
        input_c,
        input_h,
        input_w,
        0.001  # epsilon from the graphs
    )
    
    return output

def replacement_func():
    return optimized_batch_norm