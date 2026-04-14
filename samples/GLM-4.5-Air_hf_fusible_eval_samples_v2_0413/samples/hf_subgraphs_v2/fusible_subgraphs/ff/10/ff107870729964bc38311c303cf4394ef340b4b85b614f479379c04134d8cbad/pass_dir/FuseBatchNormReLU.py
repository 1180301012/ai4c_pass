import torch
import triton
import triton.language as tl

# Pattern matching for BatchNorm followed by ReLU
def pattern(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    batch_norm = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)
    relu = torch.nn.functional.relu(batch_norm, inplace=False)
    return relu

# Argument extraction function
def replacement_args(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)

# Optimized kernel combining batch norm and ReLU
@triton.jit
def fused_bn_relu_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, num_channels, height, width, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Each program processes one channel
    channel = tl.program_id(0)
    if channel >= num_channels:
        return
        
    # Load normalization parameters for this channel
    mean_val = tl.load(running_mean_ptr + channel)
    var_val = tl.load(running_var_ptr + channel)
    weight_val = tl.load(weight_ptr + channel)
    bias_val = tl.load(bias_ptr + channel)
    
    # Process all elements in this channel
    for h in range(height):
        for w in range(width):
            for b in range(batch_size):
                # Calculate linear index
                idx = (b * num_channels + channel) * height * width + h * width + w
                
                # Load input element
                x = tl.load(input_ptr + idx, other=0.0)
                
                # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
                normalized = (x - mean_val) / tl.sqrt(var_val + eps) * weight_val + bias_val
                
                # Apply ReLU activation
                relu_output = tl.maximum(normalized, 0.0)
                
                # Store result
                tl.store(output_ptr + idx, relu_output)

@torch.fx.wrap
def fused_bn_relu(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    """
    Fused batch normalization and ReLU operation
    This combines two separate operations into one kernel for better performance
    """
    # Get tensor properties
    batch, channels, height, width = input_tensor.shape
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Call optimized Triton kernel
    fused_bn_relu_kernel[(channels,)](
        input_tensor, running_mean, running_var, weight, bias, output,
        batch, channels, height, width, eps, 1
    )
    
    return output



# Replacement function
def replacement_func():
    return fused_bn_relu