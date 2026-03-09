import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    # Match BatchNorm -> ReLU sequence
    batch_norm_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu_out = torch.nn.functional.relu(batch_norm_out, inplace=True)
    return relu_out

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.jit
def fused_batchnorm_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    batch_size,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs for 3D grid (batch, channel, spatial)
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Broadcast to spatial dimensions
    spatial_offset = spatial_offset[:, None]
    
    # Calculate input address offset
    input_offset = (batch_id * n_channels * height * width + 
                   channel_id * height * width + 
                   spatial_offset)
    
    # Bounds checking
    spatial_mask = spatial_offset < width
    
    # Load input data
    input_val = tl.load(input_ptr + input_offset, mask=spatial_mask, other=0.0).to(tl.float32)
    
    # Load batch norm parameters (constants for this kernel)
    mean = tl.load(running_mean_ptr + channel_id, mask=spatial_mask & tl.arange(0, 1), other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + channel_id, mask=spatial_mask & tl.arange(0, 1), other=0.0).to(tl.float32)
    weight_val = tl.load(weight_ptr + channel_id, mask=spatial_mask & tl.arange(0, 1), other=0.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + channel_id, mask=spatial_mask & tl.arange(0, 1), other=0.0).to(tl.float32)
    
    # BatchNorm computation
    # Normalize
    normalized = (input_val - mean) / tl.sqrt(var + eps)
    
    # Scale and shift
    batch_norm_out = weight_val * normalized + bias_val
    
    # ReLU activation
    relu_out = tl.where(batch_norm_out > 0, batch_norm_out, 0.0)
    
    # Store result
    tl.store(output_ptr + input_offset, relu_out, mask=spatial_mask)

@torch.fx.wrap
def fused_batchnorm_relu(input_tensor, running_mean, running_var, weight, bias):
    # Get tensor dimensions
    batch_size, n_channels, height, width = input_tensor.shape
    
    # Calculate grid dimensions
    grid_z = batch_size
    grid_y = n_channels
    grid_x = (width * height + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    output = torch.empty_like(input_tensor)
    
    fused_batchnorm_relu_kernel[(
        grid_z,
        grid_y, 
        grid_x
    )](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_channels=n_channels,
        height=height,
        width=width,
        batch_size=batch_size,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE=256  # Optimal for 8x8 and 64x64 spatial sizes
    )
    
    return output

@torch.fx.wrap
def fused_batchnorm_relu(input_tensor, running_mean, running_var, weight, bias):
    # Get tensor dimensions
    batch_size, n_channels, height, width = input_tensor.shape
    
    # Calculate grid dimensions
    grid_z = batch_size
    grid_y = n_channels
    grid_x = (width * height + 256 - 1) // 256
    
    # Launch kernel
    output = torch.empty_like(input_tensor)
    
    fused_batchnorm_relu_kernel[(
        grid_z,
        grid_y, 
        grid_x
    )](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_channels=n_channels,
        height=height,
        width=width,
        batch_size=batch_size,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE=256  # Optimal for 8x8 and 64x64 spatial sizes
    )
    
    return output

def replacement_func():
    return fused_batchnorm_relu