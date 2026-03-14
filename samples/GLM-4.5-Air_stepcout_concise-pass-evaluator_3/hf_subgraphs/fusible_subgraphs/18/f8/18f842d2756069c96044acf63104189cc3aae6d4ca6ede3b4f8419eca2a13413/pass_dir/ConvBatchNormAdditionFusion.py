import torch
import triton
import triton.language as tl
from typing import Tuple

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Pattern matching for Conv2D + BatchNorm + Addition fusion
    
    This pattern focuses on the core computation flow without variable assignments,
    as the framework might be working with the computational graph directly.
    """
    # Core computation: Conv2D -> BatchNorm -> Addition
    conv_result = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    batchnorm_result = torch.nn.functional.batch_norm(conv_result, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    final_result = batchnorm_result + in_5
    
    # Return the final result as a tuple
    return (final_result,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Extract arguments needed for the fused kernel
    Returns running_mean, running_var, weight, bias, conv_weight, addition_input, conv_input
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def conv_batchnorm_add_kernel_1x1(
    # Input pointers
    conv_input_ptr,
    conv_weight_ptr,
    addition_input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Batch, output channels, height, width
    N: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # Input channels (from conv weight shape)
    C_in: tl.constexpr,
    # Batch norm parameters
    eps: tl.constexpr,
):
    """
    Optimized kernel for 1x1 Conv + BatchNorm + Add fusion
    """
    # Program ID maps to (batch, output_channel, spatial_position)
    pid = tl.program_id(0)
    batch_idx = pid // (C_out * H * W)
    spatial_idx = pid % (C_out * H * W)
    channel_idx = spatial_idx // (H * W)
    hw_idx = spatial_idx % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # Load 1x1 conv weights for this output channel (shape: C_in)
    conv_weights = tl.load(
        conv_weight_ptr + 
        channel_idx * C_in + tl.arange(0, C_in),
        mask=tl.arange(0, C_in) < C_in,
        other=0.0
    )
    
    # Load input feature map at this spatial position (shape: C_in)
    input_features = tl.load(
        conv_input_ptr + 
        batch_idx * C_in * H * W + 
        hw_idx * C_in + tl.arange(0, C_in),
        mask=tl.arange(0, C_in) < C_in,
        other=0.0
    )
    
    # Compute 1x1 convolution: matrix multiplication (dot product)
    conv_output = tl.dot(input_features, conv_weights)
    
    # Load batch norm parameters for this output channel
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    gamma = tl.load(weight_ptr + channel_idx)
    beta = tl.load(bias_ptr + channel_idx)
    
    # Apply batch normalization: y = (x - mean) * (gamma / sqrt(var + eps)) + beta
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn_output = (conv_output - mean) * gamma * inv_std + beta
    
    # Load addition input for this position and add
    add_input = tl.load(
        addition_input_ptr + 
        batch_idx * C_out * H * W + 
        channel_idx * H * W + hw_idx,
        mask=(hw_idx < H * W),
        other=0.0
    )
    
    # Final result: conv_output + addition_input
    final_output = bn_output + add_input
    
    # Store result
    tl.store(
        output_ptr + 
        batch_idx * C_out * H * W + 
        channel_idx * H * W + hw_idx,
        final_output,
        mask=(hw_idx < H * W)
    )

@torch.fx.wrap
def fused_conv_batchnorm_add(
    conv_input: torch.Tensor,
    conv_weight: torch.Tensor,
    addition_input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # Get tensor shapes
    N, C_in, H, W = conv_input.shape
    C_out = conv_weight.shape[0]
    
    # Verify this is a 1x1 convolution
    assert conv_weight.shape[1] == C_in, "Input channels must match"
    assert conv_weight.shape[2] == 1 and conv_weight.shape[3] == 1, "Only 1x1 convolution supported"
    
    # Verify output dimensions match addition input
    assert addition_input.shape[0] == N, "Batch size mismatch"
    assert addition_input.shape[1] == C_out, "Output channels mismatch"
    assert addition_input.shape[2] == H, "Height mismatch"
    assert addition_input.shape[3] == W, "Width mismatch"
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), device=conv_input.device, dtype=conv_input.dtype)
    
    # Calculate grid size: total number of (batch, channel, spatial_position) combinations
    grid_size = (N * C_out * H * W,)
    
    # Launch optimized kernel
    conv_batchnorm_add_kernel_1x1[grid_size](
        conv_input_ptr=conv_input,
        conv_weight_ptr=conv_weight,
        addition_input_ptr=addition_input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_out=C_out, H=H, W=W,
        C_in=C_in,
        eps=1e-05,
    )
    
    return output

def replacement_func():
    """
    Return the fused kernel function
    """
    return fused_conv_batchnorm_add