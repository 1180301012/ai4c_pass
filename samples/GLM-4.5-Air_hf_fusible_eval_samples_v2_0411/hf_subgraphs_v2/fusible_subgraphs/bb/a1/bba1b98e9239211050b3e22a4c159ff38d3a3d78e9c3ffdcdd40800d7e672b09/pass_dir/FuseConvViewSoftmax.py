import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_reshape_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_batch, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for depthwise 1x1 convolution + reshape
    """
    pid = tl.program_id(0)
    batch_id = pid // (channels * height * width)
    channel_id = (pid // (height * width)) % channels
    spatial_id = pid % (height * width)
    
    # Load input, weight, and bias for depthwise convolution
    input_idx = batch_id * channels * height * width + channel_id * height * width + spatial_id
    input_val = tl.load(input_ptr + input_idx)
    weight_val = tl.load(weight_ptr + channel_id)
    bias_val = tl.load(bias_ptr + 0)  # bias is shared across channels
    
    # Apply depthwise convolution (1x1 convolution = element-wise multiply + bias)
    conv_out = input_val * weight_val + bias_val
    
    # Store in flattened format (directly in the output tensor)
    flattened_idx = batch_id * channels * height * width + channel_id * height * width + spatial_id
    tl.store(output_ptr + flattened_idx, conv_out)

@torch.fx.wrap  
def fused_conv_reshape(input, weight, bias):
    """
    Fused implementation of conv2d + reshape operations
    """
    # Get input dimensions
    n_batch, channels, height, width = input.shape
    
    # Output shape after view operation: [n_batch, 1, channels*height*width]
    flattened_size = channels * height * width
    
    # Create output tensor with flattened shape
    output = torch.empty((n_batch, flattened_size), dtype=input.dtype, device=input.device)
    
    # Calculate grid size - one thread per element in the input tensor
    grid_size = n_batch * channels * height * width
    
    # Launch optimized convolution + reshape kernel
    fused_conv_reshape_kernel[grid_size](
        input, weight, bias, output,
        n_batch, channels, height, width,
        BLOCK_SIZE=1024
    )
    
    # Reshape output to match the expected [N, 1, flattened_size] format
    final_output = output.view(n_batch, 1, flattened_size)
    
    return final_output

def pattern(in_0, in_1, in_2):
    """
    Match the pattern: Conv2D → View → Softmax
    This matches all the variations seen in the model files
    """
    # Conv2D operation with exact same parameters as in the model
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Determine the view shape based on the batch dimension of input
    batch_size = in_2.shape[0]
    tmp_3 = tmp_2.view(batch_size, 1, -1)
    
    # Softmax operation on the last dimension
    tmp_4 = tmp_3.softmax(dim=-1)
    
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the fused operation
    """
    return (in_0, in_1, in_2)

def replacement_func():
    """
    Return the fused function as a callable
    This implements conv2d + view fusion, then applies softmax separately
    """
    def fused_function(input, weight, bias):
        # Step 1: Conv2D + View fusion using optimized Triton kernel
        fused_output = fused_conv_reshape(input, weight, bias)
        
        # Step 2: Apply softmax on the last dimension
        # Use torch's optimized softmax for the large dimension
        final_output = fused_output.softmax(dim=-1)
        
        return final_output
    
    return fused_function