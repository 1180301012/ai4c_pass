import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Match ReLU followed by flatten operations"""
    tmp_0 = torch.nn.functional.relu(input_tensor, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(input_tensor):
    """Extract arguments for the replacement kernel"""
    return (input_tensor,)

@triton.jit
def relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for ReLU + flatten on small 1x1 spatial tensors"""
    # Each program handles a channel
    channel_id = tl.program_id(0)
    
    # Calculate the offset for this channel in input (assuming [N, C, 1, 1] layout)
    input_offset = channel_id * n_features
    
    # Load the entire channel data
    channel_data = tl.load(input_ptr + input_offset, mask=input_offset + tl.arange(0, n_features) < n_channels * n_features, other=0.0)
    
    # Apply ReLU
    relu_result = tl.maximum(channel_data, 0.0)
    
    # Store directly to flattened output layout
    tl.store(output_ptr + input_offset, relu_result)

@torch.fx.wrap
def fused_relu_flatten(input_tensor):
    """Kernel wrapper for fused ReLU + flatten operation"""
    # Get input tensor properties
    n_channels = input_tensor.size(1)  # Dimension 1 is channels for [N, C, 1, 1]
    n_features = input_tensor.numel() // n_channels  # Total elements per channel
    
    # Calculate grid size
    num_channels = n_channels
    
    # Create output tensor with same dtype as input
    output_tensor = torch.empty_like(input_tensor)
    
    # Special handling for small tensors - use efficient block size
    if n_features <= 128:
        BLOCK_SIZE = 128
    elif n_features <= 256:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    # Launch kernel
    relu_flatten_kernel[(num_channels,)](
        input_tensor,
        output_tensor,
        n_channels,
        n_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor.flatten(1, -1)

def replacement_func():
    """Return the optimized kernel function"""
    return fused_relu_flatten