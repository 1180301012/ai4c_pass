import torch
import triton
import triton.language as tl

"""
Pattern: Optimizes conv2d + stack([x], dim=0).sum(dim=0) + cat

The pattern:
  conv2d_result = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
  stacked = torch.stack([conv2d_result], dim=0)
  summed = stacked.sum(dim=0)  # This is identity: sum over single-element dim
  output = torch.cat([summed, other_tensor], 1)

Since stack([x], dim=0).sum(dim=0) == identity(x), this simplifies to:
  output = torch.cat([conv2d_result, other_tensor], 1)

We optimize by:
1. Using PyTorch's optimized conv2d (cuDNN accelerated)
2. Using PyTorch's optimized cat
3. Eliminating unnecessary stack/sum overhead
"""

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: conv2d + stack + sum + cat
    
    Note: The operations use positional arguments for conv2d params to match the model exactly.
    """
    conv2d_result = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv2d_result], dim=0)
    summed = stacked.sum(dim=0)
    output = torch.cat([summed, in_3], 1)
    return output


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments: bias, weight, input, other_tensor for cat
    """
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def optimized_conv2d_cat_wrapper(bias, weight, input_tensor, other_tensor):
    """
    Optimized wrapper that eliminates stack/sum overhead.
    
    Args:
        bias: [C_out] - convolution bias
        weight: [C_out, C_in, 1, 1] - convolution weight (1x1 conv)
        input_tensor: [N, C_in, H, W] - input to conv2d
        other_tensor: [N, C_other, H, W] - tensor to concatenate
    
    Returns:
        output: [N, C_out + C_other, H, W] - concatenated result
    """
    # Use PyTorch's optimized conv2d (cuDNN accelerated for 1x1 convolutions)
    # Arguments: input, weight, bias, stride, padding, dilation, groups
    conv_result = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Use PyTorch's optimized cat
    output = torch.cat([conv_result, other_tensor], dim=1)
    
    return output


def replacement_func():
    """Return the optimized wrapper"""
    return optimized_conv2d_cat_wrapper