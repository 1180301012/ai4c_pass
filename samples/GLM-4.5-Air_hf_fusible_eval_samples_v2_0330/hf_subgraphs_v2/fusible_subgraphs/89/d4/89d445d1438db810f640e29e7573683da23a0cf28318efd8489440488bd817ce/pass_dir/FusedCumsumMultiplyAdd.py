import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation sequence
def pattern(in_0):
    # tmp_0 = in_0 (don't include assignment to avoid matching noise)
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    # tmp_1 = tmp_0 = None (don't include cleanup operations)
    tmp_3 = tmp_2 - 1
    # tmp_2 = None (don't include cleanup operations)
    tmp_4 = tmp_3.long()
    # tmp_3 = None (don't include cleanup operations)
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    # tmp_4 = None (don't include cleanup operations)
    tmp_6 = tmp_5 + 2
    # tmp_5 = None (don't include cleanup operations)
    return (tmp_6,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel - auto-tuned for small inputs
@triton.jit
def fused_cumsum_multiply_add_kernel(
    input_ptr,
    output_ptr,
    stride_dim0,
    dim1_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which row this program will process
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim1_size
    
    # Load input data for this row
    input_ptr_row = input_ptr + row_idx * stride_dim0
    x = tl.load(input_ptr_row + offsets, mask=mask, other=0)
    
    # Apply the fused operation directly: cumsum * input + 1
    # Use in-place computation for better performance
    cumsum_val = tl.cumsum(x, axis=0)
    result = cumsum_val * x + 1
    
    # Store the result directly to memory
    output_ptr_row = output_ptr + row_idx * stride_dim0
    tl.store(output_ptr_row + offsets, result, mask=mask)

# Optimized Triton kernel with better performance for small inputs
@triton.jit
def fused_cumsum_multiply_add_kernel_optimized(
    input_ptr,
    output_ptr,
    stride_dim0,
    dim1_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which row this program will process
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim1_size
    
    # Load input data with optimized vectorization for small sizes
    input_ptr_row = input_ptr + row_idx * stride_dim0
    
    # For small arrays, use efficient computation pattern
    x = tl.load(input_ptr_row + offsets, mask=mask, other=0)
    
    # Optimized fused computation with minimal operations
    result = x * (tl.cumsum(x, axis=0)) + 1
    
    # Store result directly
    output_ptr_row = output_ptr + row_idx * stride_dim0
    tl.store(output_ptr_row + offsets, result, mask=mask)

# Kernel wrapper with performance optimizations
@torch.fx.wrap
def fused_cumsum_multiply_add(input_tensor):
    # Handle the input tensor shape and properties - store original shape first
    original_shape = input_tensor.shape
    
    if input_tensor.dim() != 2:
        # If not 2D, pad to 2D to match pattern expectation
        max_dim = max(original_shape)
        padded_shape = [1, max_dim] if len(original_shape) == 1 else original_shape
        input_tensor = input_tensor.reshape(padded_shape)
    
    dim0, dim1 = input_tensor.shape
    stride_dim0 = input_tensor.stride(0)
    
    # Create output tensor with same dtype as input
    output = torch.empty_like(input_tensor)
    
    # Highly optimized block size selection for small inputs
    total_elements = dim0 * dim1
    
    if total_elements <= 32:
        # Very small inputs - use the most efficient block size
        BLOCK_SIZE = 32
    elif total_elements <= 256:
        # Small to medium inputs
        BLOCK_SIZE = 64
    else:
        # Larger inputs
        BLOCK_SIZE = 256
    
    # Calculate grid dimensions optimized for occupancy
    num_rows = dim0
    num_cols = (dim1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use optimized kernel for better small-tensor performance
    fused_cumsum_multiply_add_kernel_optimized[(num_rows, num_cols)](
        input_tensor,
        output,
        stride_dim0,
        dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # If input was originally 1D, reshape back
    if len(original_shape) == 1:
        output = output.reshape(original_shape)
    
    return output

# Replacement function that returns the optimized kernel wrapper
def replacement_func():
    return fused_cumsum_multiply_add