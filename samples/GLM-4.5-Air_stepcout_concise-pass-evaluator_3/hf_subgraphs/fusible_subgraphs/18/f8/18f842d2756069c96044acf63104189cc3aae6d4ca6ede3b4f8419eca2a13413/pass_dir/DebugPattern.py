import torch
import triton
import triton.language as tl

def pattern(conv_result, in_0, in_1, in_3, in_2):
    """
    Pattern matching for BatchNorm operation only
    Matches: tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    """
    batchnorm_result = torch.nn.functional.batch_norm(conv_result, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return batchnorm_result

def replacement_args(conv_result, in_0, in_1, in_3, in_2):
    return (conv_result, in_0, in_1, in_3, in_2)

@triton.jit
def batchnorm_kernel(
    # Input tensors
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Input/output shapes
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # Batch norm parameters
    eps: tl.constexpr,
    momentum: tl.constexpr,
):
    """
    Optimized BatchNorm kernel
    Input shapes:
    - input: [N, C, H, W]
    - running_mean, running_var, weight, bias: [C]
    """
    # Calculate grid position
    pid = tl.program_id(0)
    batch_idx = pid // (C * H * W)
    spatial_idx = pid % (C * H * W)
    channel_idx = spatial_idx // (H * W)
    hw_idx = spatial_idx % (H * W)
    
    # Load input at this spatial position for this channel
    input_val = tl.load(
        input_ptr + batch_idx * C * H * W + channel_idx * H * W + hw_idx,
        mask=(channel_idx < C) & (hw_idx < H * W),
        other=0.0
    )
    
    # Load batch norm parameters for this channel
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    gamma = tl.load(weight_ptr + channel_idx)
    beta = tl.load(bias_ptr + channel_idx)
    
    # Apply batch normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    output_val = (input_val - mean) * gamma * inv_std + beta
    
    # Store result
    tl.store(
        output_ptr + batch_idx * C * H * W + channel_idx * H * W + hw_idx,
        output_val,
        mask=(channel_idx < C) & (hw_idx < H * W)
    )

@torch.fx.wrap
def optimized_batchnorm(input_tensor, running_mean, running_var, weight, bias):
    """
    Optimized BatchNorm implementation
    """
    N, C, H, W = input_tensor.shape
    
    # Create output tensor
    output = torch.empty((N, C, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate grid size
    grid_size = (N * C * H * W,)
    
    # Launch kernel
    batchnorm_kernel[grid_size](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        eps=1e-05,
        momentum=0.1,
    )
    
    return output

def replacement_func():
    return optimized_batchnorm