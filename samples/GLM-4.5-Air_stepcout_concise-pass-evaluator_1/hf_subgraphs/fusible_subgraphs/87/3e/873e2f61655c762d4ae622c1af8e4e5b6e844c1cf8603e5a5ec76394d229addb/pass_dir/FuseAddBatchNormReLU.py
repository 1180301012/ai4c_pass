import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
    # tmp_8 = in_7 + tmp_7 (input_tensor here is the result of the previous upsample)
    added = input_tensor
    # tmp_9 = torch.nn.functional.batch_norm(tmp_8, running_mean, running_var, weight, bias, False, momentum, eps)
    batch_norm_out = torch.nn.functional.batch_norm(added, running_mean, running_var, weight, bias, False, momentum, eps)
    # tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
    relu_out = torch.nn.functional.relu(batch_norm_out, inplace=True)
    return relu_out

def replacement_args(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
    return (input_tensor, running_mean, running_var, weight, bias, momentum, eps)

@triton.jit
def fused_add_batch_norm_relu_kernel(
    input_ptr,           # [batch, channels, height, width]
    running_mean_ptr,    # [channels]
    running_var_ptr,     # [channels]
    weight_ptr,          # [channels]  
    bias_ptr,            # [channels]
    output_ptr,          # [batch, channels, height, width]
    batch_size,
    num_channels,
    height, width,
    momentum,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one spatial location for one channel
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    pid_y = tl.program_id(2)  # height
    pid_x = tl.program_id(3)  # width
    
    # Bounds checking
    if (pid_b >= batch_size or pid_c >= num_channels or 
        pid_y >= height or pid_x >= width):
        return
    
    # Load BN parameters for this channel
    mean = tl.load(running_mean_ptr + pid_c)
    var = tl.load(running_var_ptr + pid_c)
    weight_val = tl.load(weight_ptr + pid_c)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Load input value
    input_offset = (pid_b * num_channels + pid_c) * height * width + pid_y * width + pid_x
    input_val = tl.load(input_ptr + input_offset, mask=True, other=0.0)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    var_eps = var + eps
    inv_std = 1.0 / tl.sqrt(var_eps)
    normalized = (input_val - mean) * inv_std * weight_val + bias_val
    
    # Apply ReLU: max(0, normalized)
    relu_out = tl.maximum(normalized, 0.0)
    
    # Store result
    output_offset = (pid_b * num_channels + pid_c) * height * width + pid_y * width + pid_x
    tl.store(output_ptr + output_offset, relu_out)

@torch.fx.wrap
def fused_add_batch_norm_relu(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
    # Get tensor shapes
    batch_size, num_channels, height, width = input_tensor.shape
    
    # Create output tensor
    output = torch.empty((batch_size, num_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    grid = (batch_size, num_channels, height, width)
    
    fused_add_batch_norm_relu_kernel[grid](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        momentum=momentum,
        eps=eps,
        BLOCK_SIZE_C=32,
    )
    
    return output

def replacement_func():
    return fused_add_batch_norm_relu