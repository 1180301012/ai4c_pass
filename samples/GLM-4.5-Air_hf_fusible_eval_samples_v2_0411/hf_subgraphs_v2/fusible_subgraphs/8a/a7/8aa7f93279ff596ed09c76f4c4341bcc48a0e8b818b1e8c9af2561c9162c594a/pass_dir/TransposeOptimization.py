import torch
import triton
import triton.language as tl

# Pattern matching function - matches the complete forward computation structure
def pattern(in_0, in_1):
    """Match complete forward computation: (tmp_0 = in_1 * scalar, tmp_1 = in_0.transpose(-2, -1))"""
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.transpose(-2, -1)
    return (tmp_0, tmp_1)

# Argument extraction function
def replacement_args(in_0, in_1):
    """Extract both input tensors for complete computation"""
    return (in_0, in_1)

# Corrected Triton transpose kernel for last two dimensions
@triton.jit
def correct_transpose_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    dim0_size,
    dim1_size,
    leading_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Corrected kernel for transpose of last two dimensions"""
    pid = tl.program_id(0)
    
    # Process multiple elements per thread
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose offsets into leading indices and local 2D coordinates
    local_offset = offsets % (dim0_size * dim1_size)
    leading_idx = offsets // (dim0_size * dim1_size)
    
    # Convert 2D local coordinates (original layout)
    orig_k = local_offset // dim0_size   # second-to-last dimension (49)
    orig_l = local_offset % dim0_size    # last dimension (32)
    
    # Convert 2D coordinates for transposed layout
    trans_k = orig_l                     # now becomes second-to-last dimension
    trans_l = orig_k                     # now becomes last dimension
    
    # Calculate linear indices in input and output
    # Input: [leading_indices..., orig_k, orig_l]  
    input_idx = leading_idx * (dim0_size * dim1_size) + orig_k * dim0_size + orig_l
    
    # Output: [leading_indices..., trans_k, trans_l] = [leading_indices..., orig_l, orig_k]
    output_idx = leading_idx * (dim0_size * dim1_size) + trans_k * dim0_size + trans_l
    
    # Load from input, store to output with swapped coordinates
    input_data = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + output_idx, input_data, mask=mask)

# Optimized combined computation kernel (scalar multiplication + transpose)
@triton.jit
def combined_kernel(
    input_0_ptr,  # in_0 (for transpose)
    input_1_ptr,  # in_1 (for scalar multiplication)
    output_0_ptr, # tmp_0 (result of scalar multiplication)
    output_1_ptr, # tmp_1 (result of transpose)
    total_elements,
    dim0_size,
    dim1_size,
    leading_size,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Combined kernel for scalar multiplication and transpose operations"""
    pid = tl.program_id(0)
    
    # Process multiple elements per thread
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # === Scalar multiplication part (on input_1) ===
    # Simply multiply by scalar value
    input_data_1 = tl.load(input_1_ptr + offsets, mask=mask, other=0.0)
    result_1 = input_data_1 * scalar_val
    tl.store(output_0_ptr + offsets, result_1, mask=mask)
    
    # === Transpose part (on input_0) ===
    # Decompose offsets for transpose operation
    local_offset = offsets % (dim0_size * dim1_size)
    leading_idx = offsets // (dim0_size * dim1_size)
    
    # Convert 2D local coordinates (original layout)
    orig_k = local_offset // dim0_size   # second-to-last dimension (49)
    orig_l = local_offset % dim0_size    # last dimension (32)
    
    # Convert 2D coordinates for transposed layout
    trans_k = orig_l                     # now becomes second-to-last dimension
    trans_l = orig_k                     # now becomes last dimension
    
    # Calculate linear indices in input and output for transpose
    input_idx = leading_idx * (dim0_size * dim1_size) + orig_k * dim0_size + orig_l
    output_idx = leading_idx * (dim0_size * dim1_size) + trans_k * dim0_size + trans_l
    
    # Load from input_0, store to output_1 with swapped coordinates
    input_data_0 = tl.load(input_0_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_1_ptr + output_idx, input_data_0, mask=mask)

# Kernel wrapper (optimized for combined operations)
@torch.fx.wrap
def optimized_combined_operations(in_0, in_1):
    """Perform optimized combined operations using Triton"""
    # Get original shape for transpose operation
    original_shape = in_0.shape
    
    # Get sizes of the two dimensions being transposed
    last_dim = len(original_shape) - 1
    dim0_size = original_shape[last_dim]   # Original last dimension size (32)
    dim1_size = original_shape[last_dim-1] # Original second-to-last dimension size (49)
    
    # Calculate leading size (product of dimensions before the last two)
    leading_size = 1
    for i in range(len(original_shape) - 2):
        leading_size *= original_shape[i]
    
    total_elements = leading_size * dim0_size * dim1_size
    
    # Use optimal block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    # For scalar multiplication result: same shape as in_1
    out_0 = torch.empty_like(in_1)
    
    # For transpose result: swapped last two dimensions
    new_shape = list(original_shape)
    new_shape[last_dim-1], new_shape[last_dim] = new_shape[last_dim], new_shape[last_dim-1]
    out_1 = torch.empty(new_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch combined Triton kernel
    combined_kernel[(num_programs,)](
        input_0_ptr=in_0,
        input_1_ptr=in_1,
        output_0_ptr=out_0,
        output_1_ptr=out_1,
        total_elements=total_elements,
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        leading_size=leading_size,
        scalar_val=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_0, out_1)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_combined_operations