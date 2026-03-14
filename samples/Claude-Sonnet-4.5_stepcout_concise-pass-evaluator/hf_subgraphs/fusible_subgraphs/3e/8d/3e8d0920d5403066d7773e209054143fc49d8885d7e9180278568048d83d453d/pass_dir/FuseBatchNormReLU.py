import torch
import triton
import triton.language as tl


def pattern(input_tensor, running_mean, running_var, weight, bias):
    """Pattern to match: batch_norm followed by relu"""
    bn_out = torch.nn.functional.batch_norm(
        input_tensor, running_mean, running_var, weight, bias, False, 0.1, 0.001
    )
    relu_out = torch.nn.functional.relu(bn_out, inplace=False)
    return relu_out


def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)


@triton.jit
def fused_batchnorm_relu_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused batch normalization + ReLU kernel"""
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * HW
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index for each element
    channel_idx = (offsets // HW) % C
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + channel_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    # Batch normalization
    normalized = (x - mean) / tl.sqrt(var + eps)
    bn_out = w * normalized + b
    
    # ReLU
    output = tl.maximum(bn_out, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_batchnorm_relu(input_tensor, running_mean, running_var, weight, bias):
    """Wrapper for fused batch norm + relu"""
    N, C, H, W = input_tensor.shape
    HW = H * W
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    total_elements = N * C * HW
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_batchnorm_relu_kernel[grid](
        input_tensor,
        output,
        running_mean,
        running_var,
        weight,
        bias,
        N,
        C,
        HW,
        eps=0.001,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_batchnorm_relu