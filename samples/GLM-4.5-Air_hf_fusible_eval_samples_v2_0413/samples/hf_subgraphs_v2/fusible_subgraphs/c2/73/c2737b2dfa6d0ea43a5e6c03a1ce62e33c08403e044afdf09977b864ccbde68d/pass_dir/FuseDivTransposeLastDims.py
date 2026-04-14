import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation in model.py
def pattern(input_tensor):
    tmp_0 = input_tensor / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Constants from the computation
DIVISOR = 1.6817928305074292

@triton.jit
def fused_div_transpose_kernel(
    input_ptr,
    output_ptr,
    stride_0, stride_1, stride_2, stride_3,
    n_0, n_1, n_2, n_3,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies division by constant and transpose last two dimensions
    """
    # Get program IDs for parallelization (limited to 3 dimensions in Triton)
    pid_0 = tl.program_id(0)  # batch dimension
    pid_1 = tl.program_id(1)  # sequence length dimension
    pid_2 = tl.program_id(2)  # combined heads and head_dim
    
    # Unpack pid_2 into heads and head_dim indices
    head_idx = pid_2 // n_2  # n_2 is head_dim
    head_dim_idx = pid_2 % n_2  # n_2 is head_dim
    
    # Create offsets within the block for parallel processing
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_2 * n_3)  # Total elements in the last two dimensions
    
    # Load input data: input[batch, seq, head_dim, num_heads]
    # Calculate linear index for the last two dimensions
    linear_idx = offsets
    head_dim_idx_local = linear_idx % n_2
    head_idx_local = linear_idx // n_2
    
    input_ptr_local = input_ptr + pid_0 * stride_0 + pid_1 * stride_1 + head_dim_idx_local * stride_2 + head_idx_local * stride_3
    input_data = tl.load(input_ptr_local, mask=mask, other=0.0)
    
    # Apply division by constant
    divided_data = input_data / divisor
    
    # Store to output with transpose: output[batch, seq, num_heads, head_dim]
    # Swap the order of the last two dimensions in the output
    output_ptr_local = output_ptr + pid_0 * stride_0 + pid_1 * stride_1 + head_idx_local * stride_2 + head_dim_idx_local * stride_3
    tl.store(output_ptr_local, divided_data, mask=mask)

@torch.fx.wrap
def fused_div_transpose(input_tensor):
    """
    Optimized function that eliminates intermediate tensor allocation
    by using a custom Triton kernel to perform division and transpose in one operation.
    """
    # Get tensor properties
    shape = input_tensor.shape
    strides = input_tensor.stride()
    dtype = input_tensor.dtype
    device = input_tensor.device
    
    # Handle tensors with exactly 4 dimensions [batch, seq_len, head_dim, num_heads]
    if len(shape) == 4:
        batch, seq_len, head_dim, num_heads = shape
        
        # Create output tensor with last two dimensions transposed
        output_shape = [batch, seq_len, num_heads, head_dim]
        output_tensor = torch.empty(output_shape, dtype=dtype, device=device)
        
        # Choose block size for optimal GPU utilization
        BLOCK_SIZE = 256  # Number of elements to process per program
        
        # Calculate grid dimensions - using only 3D grid as Triton requires
        grid = (
            batch,
            seq_len,
            num_heads * head_dim,
        )
        
        # Launch the kernel
        fused_div_transpose_kernel[grid](
            input_tensor, output_tensor,
            strides[0], strides[1], strides[2], strides[3],
            batch, seq_len, head_dim, num_heads,
            DIVISOR,
            BLOCK_SIZE,
        )
        
        return output_tensor
    else:
        # Fallback for other tensor shapes - use separate operations
        # but avoid intermediate tensor where possible
        divided = input_tensor / DIVISOR
        return divided.transpose(-1, -2)

# Replacement function - returns the fused function
def replacement_func():
    return fused_div_transpose