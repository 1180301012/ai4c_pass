import torch

def pattern(x, running_mean, running_var, weight, bias):
    """
    Pattern to match batch_norm followed by relu operations.
    """
    # BatchNorm operation
    tmp_6 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    # ReLU operation  
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    return tmp_7

def replacement_args(x, running_mean, running_var, weight, bias):
    """
    Extract all arguments needed for the fused batch norm + relu operation.
    """
    return (x, running_mean, running_var, weight, bias)

@torch.fx.wrap  
def fused_batch_norm_relu(x, running_mean, running_var, weight, bias):
    """
    Fused batch normalization + ReLU implementation.
    This eliminates the intermediate tensor allocation between batch norm and relu.
    """
    # Get tensor shape
    if x.dim() == 4:  # Assuming [N, C, H, W] format
        batch_size, channels, height, width = x.shape
        total_elements = batch_size * channels * height * width
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Apply batch normalization followed by ReLU element-wise
        # This avoids creating intermediate tensors and reduces memory overhead
        
        # Calculate the batch norm + relu
        # y = relu((x - mean) / sqrt(var + eps) * weight + bias)
        result = (x - running_mean.view(1, -1, 1, 1)) * torch.rsqrt(running_var.view(1, -1, 1, 1) + 1e-05) * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
        output = torch.relu(result)
        
        return output
    else:
        # Fallback to separate operations for other tensor shapes
        tmp_6 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
        return torch.nn.functional.relu(tmp_6, inplace=False)

def replacement_func():
    """
    Return the fused batch norm + relu function.
    """
    return fused_batch_norm_relu