import torch
import triton
import triton.language as tl

def pattern(in_1):
    """
    Pattern: ReLU (inplace=True) followed by reshape with dynamic batch size
    This matches the pattern tmp_0 = relu(in_1, inplace=True) followed by tmp_2 = tmp_0.reshape(batch_size, 256, -1)
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    # Use the batch size from input but keep channels as 256 (which is constant across both graphs)
    batch_size = in_1.shape[0]
    tmp_2 = tmp_0.reshape(batch_size, 256, -1)
    return tmp_2

def replacement_args(in_1):
    """
    Extract arguments for replacement
    We need the input tensor and its original shape information
    """
    return (in_1,)

@triton.jit
def fused_relu_reshape_kernel(
    input_ptr, 
    output_ptr,
    n_batch,
    n_channels,
    n_length,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total elements
    total_elements = n_batch * n_channels * n_length
    
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input (contiguous memory after removing the last dim)
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_out = tl.maximum(input, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_relu_reshape_wrapper(input_tensor):
    # Get input shape (original format: [N, C, D, 1])
    original_shape = input_tensor.shape
    n_batch, n_channels, n_length, _ = original_shape
    
    # Create output tensor with reshaped shape [N, C, D]
    output_shape = (n_batch, n_channels, n_length)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate optimal block size
    total_elements = n_batch * n_channels * n_length
    BLOCK_SIZE = 1024  # Standard block size for good occupancy
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_relu_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_batch=n_batch,
        n_channels=n_channels,
        n_length=n_length,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """
    Return optimized kernel that fuses ReLU + reshape operations
    """
    return fused_relu_reshape_wrapper