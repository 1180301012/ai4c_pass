import torch
import triton
import triton.language as tl

# Pattern matching function to match BatchNorm + Add + ReLU
def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    # BatchNorm: input=in_4, running_mean=in_0, running_var=in_1, weight=in_3, bias=in_2
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Element-wise addition with in_5
    tmp_5 = in_5 + tmp_4
    
    # ReLU activation 
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    
    # Return the tuple that matches what the original function returns
    return (tmp_6, tmp_6.mean((2, 3), keepdim=True))

def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)

@triton.jit
def fused_batch_norm_add_relu_kernel(
    # Input tensor pointers
    x_ptr, in_5_ptr,
    # BatchNorm parameters
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output tensor pointers
    relu_out_ptr, mean_sum_ptr,
    # Shape information
    batch_size, num_channels, height, width,
    # BatchNorm parameters
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(batch_size * num_channels * height * width, BLOCK_SIZE)
    
    if pid >= num_pid:
        return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * num_channels * height * width)
    
    # Reshape offsets for 4D tensor access
    b = offsets // (num_channels * height * width)
    c = (offsets // (height * width)) % num_channels
    h = (offsets // width) % height  
    w = offsets % width
    
    idx_bchw = b * num_channels * height * width + c * height * width + h * width + w
    
    # Load input tensors
    x = tl.load(x_ptr + idx_bchw, mask=mask, other=0.0)
    in_5_val = tl.load(in_5_ptr + idx_bchw, mask=mask, other=0.0)
    
    # Load BatchNorm parameters for this channel - create per-channel mask
    channel_mask = (c < num_channels)
    
    # Load BatchNorm parameters with proper masking
    mean = tl.load(running_mean_ptr + c, mask=channel_mask, other=0.0)
    var = tl.load(running_var_ptr + c, mask=channel_mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=channel_mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=channel_mask, other=0.0)
    
    # Apply BatchNorm: (x - mean) * weight / sqrt(var + eps) + bias
    inv_std = tl.math.rsqrt(var + eps)
    x_norm = (x - mean) * inv_std
    x_bn = x_norm * weight + bias
    
    # Element-wise addition with in_5
    x_add = x_bn + in_5_val
    
    # Apply ReLU
    x_relu = tl.maximum(x_add, 0.0)
    
    # Store ReLU output
    tl.store(relu_out_ptr + idx_bchw, x_relu, mask=mask)
    
    # Accumulate per-channel sum for mean calculation
    tl.atomic_add(mean_sum_ptr + c, x_relu)

@torch.fx.wrap
def fused_batch_norm_add_relu(in_4, in_0, in_1, in_3, in_2, in_5):
    batch_size, num_channels, height, width = in_4.shape
    
    # Allocate output tensors
    relu_out = torch.empty_like(in_5)
    
    # Initialize mean accumulation buffer
    mean_buffer = torch.zeros(num_channels, dtype=in_4.dtype, device=in_4.device)
    
    # Block size for the main computation
    BLOCK_SIZE = 1024
    
    # Launch main kernel
    num_programs = (batch_size * num_channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_batch_norm_add_relu_kernel[(num_programs,)](
        in_4, in_5,
        in_0, in_1, in_3, in_2,  # BatchNorm parameters
        relu_out, mean_buffer,
        batch_size, num_channels, height, width,
        1e-05,  # eps
        BLOCK_SIZE,
    )
    
    # Calculate mean from accumulated sums
    # For spatial mean: divide by (height * width)
    scale = 1.0 / (height * width)
    mean_result = mean_buffer * scale
    
    # Reshape mean to match expected format: [batch_size, num_channels, 1, 1]
    mean_reshaped = mean_result.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_channels, 1, 1)
    
    return relu_out, mean_reshaped

def replacement_func():
    return fused_batch_norm_add_relu