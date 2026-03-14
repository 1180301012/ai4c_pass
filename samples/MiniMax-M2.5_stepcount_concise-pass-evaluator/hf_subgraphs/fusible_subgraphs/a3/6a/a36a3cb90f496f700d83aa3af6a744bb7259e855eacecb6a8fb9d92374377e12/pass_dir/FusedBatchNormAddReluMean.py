import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the pattern: batch_norm -> add -> relu -> mean
    - in_0: running_mean
    - in_1: running_var
    - in_2: bias
    - in_3: weight
    - in_4: input tensor
    - in_5: residual input
    """
    # BatchNorm operation
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Add residual
    tmp_5 = in_5 + tmp_4
    
    # ReLU activation
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    
    # Mean over spatial dimensions (2, 3) with keepdim=True
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    
    return tmp_6, tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# Fused kernel: batch_norm + add + relu + mean
@triton.jit
def fused_bn_add_relu_mean_kernel(
    # Input pointers
    in_ptr, residual_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output pointers
    out_relu_ptr, out_mean_ptr,
    # Sizes
    N, C, H, W,
    # BN parameters
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    2. Add residual
    3. ReLU: max(0, x)
    4. Mean over spatial dimensions (H, W)
    """
    # Get batch and channel indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Channel offset
    c_offset = pid_c * BLOCK_SIZE_C
    
    # Load running mean and var for the channel
    running_mean = tl.load(running_mean_ptr + c_offset)
    running_var = tl.load(running_var_ptr + c_offset)
    
    # Compute standard deviation
    std = tl.sqrt(running_var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + c_offset)
    bias = tl.load(bias_ptr + c_offset)
    
    # Compute normalized weight factor for fast computation
    # BN formula: (x - mean) / sqrt(var + eps) * weight + bias
    # = x * (weight / std) + (bias - mean * weight / std)
    weight_factor = weight / std
    bias_factor = bias - running_mean * weight_factor
    
    # Accumulate for mean calculation
    sum_val = 0.0
    
    # Process each spatial position
    for h in range(H):
        for w in range(W):
            # Calculate flat index
            offset = pid_batch * N * C * H * W + c_offset * H * W + h * W + w
            
            # Load input value
            x = tl.load(in_ptr + offset)
            
            # Apply BatchNorm
            x_norm = x * weight_factor + bias_factor
            
            # Load residual and add
            residual = tl.load(residual_ptr + offset)
            x_add = x_norm + residual
            
            # Apply ReLU
            x_relu = tl.maximum(x_add, 0.0)
            
            # Store output
            tl.store(out_relu_ptr + offset, x_relu)
            
            # Accumulate for mean
            sum_val += x_relu
    
    # Store mean output
    # Mean output shape: [batch, channel, 1, 1]
    mean_offset = pid_batch * C + pid_c
    # Since we have BLOCK_SIZE_C channels per program, need to scale
    num_elements = H * W
    mean_val = sum_val / num_elements
    tl.store(out_mean_ptr + mean_offset, mean_val)


@torch.fx.wrap
def fused_bn_add_relu_mean_kernel_wrapper(
    running_mean, running_var, weight, bias, input_tensor, residual, eps=1e-05
):
    """
    Wrapper for the fused kernel.
    Computes: BN(input) + residual -> ReLU -> mean
    """
    batch_size, num_channels, H, W = input_tensor.shape
    
    # Allocate output tensors
    output_relu = torch.empty_like(input_tensor)
    # Mean output shape: [batch, channels, 1, 1]
    output_mean = torch.empty((batch_size, num_channels, 1, 1), 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Define block sizes
    # Process channels in blocks
    BLOCK_SIZE_C = 1  # Process 1 channel at a time for simplicity
    num_channel_blocks = num_channels
    
    # Grid: (batch_size, num_channels)
    grid = (batch_size, num_channel_blocks)
    
    fused_bn_add_relu_mean_kernel[grid](
        input_tensor, residual, running_mean, running_var, weight, bias,
        output_relu, output_mean,
        batch_size, num_channels, H, W,
        eps,
        BLOCK_SIZE_C,
        H * W,
    )
    
    return output_relu, output_mean


def replacement_func():
    return fused_bn_add_relu_mean_kernel_wrapper