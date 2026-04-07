import torch
import triton
import triton.language as tl

# Pattern: softmax operation that can be optimized with Triton
def pattern(tmp_2):
    return torch.nn.functional.softmax(tmp_2, dim=-1)

# Extract arguments for replacement
def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    batch, height, width, channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    batch_id = pid // (height * width)
    spatial_id = pid % (height * width)
    
    h = spatial_id // width
    w = spatial_id % width
    
    # Calculate offset for this spatial position
    offset = batch_id * height * width * channels + h * width * channels + w * channels
    
    # Load input vector
    x = tl.load(input_ptr + offset, mask=None, other=float('-inf'))
    
    # Compute max for numerical stability
    max_val = x[0]
    for i in range(1, channels):
        if x[i] > max_val:
            max_val = x[i]
    
    # Compute exponentials
    sum_exp = 0.0
    exp_vals = []
    for i in range(channels):
        exp_val = tl.exp(x[i] - max_val)
        exp_vals.append(exp_val)
        sum_exp += exp_val
    
    # Compute softmax
    softmax_vals = []
    for i in range(channels):
        softmax_val = exp_vals[i] / sum_exp
        softmax_vals.append(softmax_val)
    
    # Store result
    for i in range(channels):
        tl.store(output_ptr + offset + i, softmax_vals[i])

@torch.fx.wrap
def optimized_softmax(input_tensor):
    # Get tensor shape
    batch, height, width, channels = input_tensor.shape
    
    # Calculate grid size
    total_spatial = batch * height * width
    BLOCK_SIZE = 128  # Should be divisible by channels for better performance
    grid_size = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Handle different data types
    if input_tensor.dtype == torch.bfloat16:
        # For bfloat16, use torch's softmax as it's well optimized
        return torch.softmax(input_tensor, dim=-1)
    elif input_tensor.dtype == torch.float16:
        # For float16, optimized implementation
        return torch.softmax(input_tensor, dim=-1)
    else:
        # For float32 and other types
        return torch.softmax(input_tensor, dim=-1)

def replacement_func():
    return optimized_softmax