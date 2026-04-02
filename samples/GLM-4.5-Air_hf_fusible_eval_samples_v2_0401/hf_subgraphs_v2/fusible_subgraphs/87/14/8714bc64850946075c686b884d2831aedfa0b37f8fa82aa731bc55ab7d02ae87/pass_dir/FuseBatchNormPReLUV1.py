import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_batchnorm_prelu_kernel(
    # Input tensors
    x_ptr,              # Input [N, C, H, W]
    running_mean_ptr,   # Running mean [C]
    running_var_ptr,    # Running var [C]
    weight_ptr,         # BN weight [C]
    bias_ptr,           # BN bias [C]
    prelu_weight_ptr,   # PReLU weight [C]
    # Output tensor
    out_ptr,            # Output [N, C, H, W]
    # Metadata
    n_batch,            # Batch size N
    n_channels,         # Number of channels C
    height,             # Height H
    width,              # Width W
    # Constants
    eps,                # BN epsilon
    momentum,           # BN momentum
    # Block size config
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Program indices
    pid_n = tl.program_id(0)  # Batch dimension
    pid_c = tl.program_id(1)  # Channel dimension
    
    # Create ranges for spatial dimensions
    hw_offset = tl.arange(0, BLOCK_SIZE_HW)
    hw_indices = hw_offset[:, None] * width + hw_offset[None, :]
    
    # Mask for spatial dimensions
    hw_mask = (hw_offset[:, None] < height) & (hw_offset[None, :] < width)
    
    # Calculate batch and channel offsets
    batch_offset = pid_n * n_channels * height * width
    channel_offset = pid_c * height * width
    
    # Load running statistics
    running_mean = tl.load(running_mean_ptr + pid_c)
    running_var = tl.load(running_var_ptr + pid_c)
    
    # Load BN parameters
    bn_weight = tl.load(weight_ptr + pid_c)
    bn_bias = tl.load(bias_ptr + pid_c)
    
    # Load PReLU weight
    prelu_weight = tl.load(prelu_weight_ptr + pid_c)
    
    # Compute fused batch norm and PReLU
    # Load input block
    input_index = batch_offset + channel_offset + hw_indices
    x_val = tl.load(x_ptr + input_index, mask=hw_mask, other=0.0)
    
    # BatchNorm: (x - running_mean) / sqrt(running_var + eps)
    sqrt_var = tl.sqrt(running_var + eps)
    normalized = (x_val - running_mean) / sqrt_var
    
    # Affine transform: weight * normalized + bias
    bn_out = bn_weight * normalized + bn_bias
    
    # PReLU: max(0, bn_out) + prelu_weight * min(0, bn_out)
    # This can be written as: bn_out * (bn_out > 0) + prelu_weight * bn_out * (bn_out <= 0)
    positive_mask = bn_out > 0
    negative_mask = ~positive_mask
    
    relu_out = bn_out * positive_mask
    prelu_out = prelu_weight * bn_out * negative_mask
    
    # Final result
    fused_out = relu_out + prelu_out
    
    # Store output
    output_index = input_index
    tl.store(out_ptr + output_index, fused_out, mask=hw_mask)

@torch.fx.wrap
def fused_batchnorm_prelu(x, running_mean, running_var, weight, bias, prelu_weight):
    # Get shapes
    n_batch, n_channels, height, width = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Configure block sizes
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 64  # Process multiple channels simultaneously
    BLOCK_SIZE_HW = 16 * 16  # 16x16 block for spatial dimensions
    
    # Calculate grid dimensions
    grid_n = (n_batch + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (n_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch kernel
    fused_batchnorm_prelu_kernel[(grid_n, grid_c, grid_hw)](
        x, running_mean, running_var, weight, bias, prelu_weight, out,
        n_batch, n_channels, height, width,
        0.001, 0.1,  # eps and momentum from model.py
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    
    return out

def pattern(tmp_7, in_1, in_2, in_4, in_3, in_0):
    """Match BatchNorm + PReLU pattern"""
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.prelu(tmp_8, in_0)
    return tmp_9

def replacement_args(tmp_7, in_1, in_2, in_4, in_3, in_0):
    """Extract arguments for replacement"""
    return (tmp_7, in_1, in_2, in_4, in_3, in_0)

def replacement_func():
    """Return the fused function"""
    return fused_batchnorm_prelu