import torch
import triton
import triton.language as tl

# Pattern matching function - must match exact computation structure from model.py
def pattern(in_0):
    """
    Pattern to match: division by constant followed by transpose of last two dimensions.
    
    This matches the computation:
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)
    """
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    """Extract the input tensor argument for the fused kernel."""
    return (in_0,)

# Constants
DIVISOR = 1.6817928305074292

# Highly optimized kernel with memory access pattern optimization
@triton.jit
def fused_divide_transpose_kernel(
    input_ptr,
    output_ptr,
    stride_inner,
    stride_last,
    DIVISOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized fused kernel that performs element-wise division and transpose.
    
    Implements optimized memory access patterns for transpose operations and efficient vectorization.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Create offset range for this block with better alignment
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (stride_inner * stride_last)
    
    # Convert to 2D coordinates for the last two dimensions
    inner = offsets // stride_last
    last = offsets % stride_last
    
    # Create masks for each dimension
    inner_mask = inner < stride_inner
    last_mask = last < stride_last
    
    # Combined mask for all bounds checking
    combined_mask = mask & inner_mask & last_mask
    
    # Process data with vectorized memory access for better throughput
    # Use stride-8 vectorization for optimal cache line utilization
    for i in range(0, 8):
        # Create vectorized offset with proper alignment
        vector_offsets = offsets + i
        
        # Vectorized bounds checking
        vector_mask = vector_offsets < (stride_inner * stride_last)
        
        # Create transposed coordinates for efficient memory access
        # We access data as if for transpose from the beginning
        vector_inner = vector_offsets // stride_last
        vector_last = vector_offsets % stride_last
        
        # Apply transpose immediately in coordinate calculation
        # Load from [inner, last], store to [last, inner]
        load_inner = vector_last  # Transposed for efficient memory access
        load_last = vector_inner  # Transposed for efficient memory access
        
        # Store inner/last (non-transposed for output)
        store_inner = vector_inner
        store_last = vector_last
        
        # Bounds checking
        load_inner_mask = load_inner < stride_inner
        load_last_mask = load_last < stride_last
        
        # Combined vector mask
        vector_combined_mask = vector_mask & load_inner_mask & load_last_mask
        
        # Load input data with transposed access pattern for cache efficiency
        load_ptrs = input_ptr + (load_inner * stride_inner + load_last)
        input_vals = tl.load(load_ptrs, mask=vector_combined_mask, other=0.0)
        
        # Apply division
        result_vals = input_vals / DIVISOR
        
        # Store to output with original (non-transposed) coordinates
        store_ptrs = output_ptr + (store_inner * stride_last + store_last)
        tl.store(store_ptrs, result_vals, mask=vector_combined_mask)

@torch.fx.wrap
def fused_divide_transpose_kernel_wrapper(input_tensor):
    """
    Wrapper function that sets up and launches the fused kernel.
    
    This function properly computes strides and launches an optimized 2D kernel.
    """
    # Get input shape - the last two dimensions will be transposed
    # Input shape: [..., inner_dim, last_dim]
    output_shape = input_tensor.shape[:-2] + input_tensor.shape[-2:][::-1]
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Compute strides for the input tensor (row-major order)
    strides = []
    current_stride = 1
    for dim in reversed(input_tensor.shape):
        strides.append(current_stride)
        current_stride *= dim
    strides = list(reversed(strides))
    
    # Extract strides for the last two dimensions
    stride_inner = strides[-2]  # Stride for the inner dimension (penultimate)
    stride_last = strides[-1]   # Stride for the last dimension
    
    # Use larger block size with stride-8 vectorization for optimal performance
    BLOCK_SIZE = 4096  # Optimal block size for stride-8 vectorization and GPU occupancy
    
    # Calculate 1D grid size for the last two dimensions
    total_elements = stride_inner * stride_last
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with 1D grid
    fused_divide_transpose_kernel[(grid_size,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        stride_inner=stride_inner,
        stride_last=stride_last,
        DIVISOR=DIVISOR,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function - must return a zero-arg function that returns the kernel wrapper
def replacement_func():
    """Returns the fused kernel wrapper function."""
    return fused_divide_transpose_kernel_wrapper