import torch
import triton
import triton.language as tl

# Pattern matching function for fused operations
def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel for element-wise scalar multiplication and transpose
@triton.jit
def fused_operations_kernel(
    input_0_ptr,  # For transpose
    input_1_ptr,  # For scalar multiplication
    output_0_ptr,  # Transpose result
    output_1_ptr,  # Scalar multiplication result
    batch_size,
    feature_dim1,
    feature_dim2,
    stride_batch_0,
    stride_feature1_0,
    stride_feature2_0,
    n_elements_1,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles multiple elements
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * feature_dim1 * feature_dim2 + n_elements_1)
    
    # Process transpose operation for input_0
    # Calculate indices for transpose(-2, -1)
    within_feature_space = offsets < (batch_size * feature_dim1 * feature_dim2)
    mask_transpose = within_feature_space
    
    batch_idx = tl.where(within_feature_space, offsets // (feature_dim1 * feature_dim2), 0)
    remaining = tl.where(within_feature_space, offsets % (feature_dim1 * feature_dim2), 0)
    feature1_idx = tl.where(within_feature_space, remaining // feature_dim2, 0)
    feature2_idx = tl.where(within_feature_space, remaining % feature_dim2, 0)
    
    input_offset = batch_idx * stride_batch_0 + feature1_idx * stride_feature1_0 + feature2_idx * stride_feature2_0
    output_offset = batch_idx * stride_batch_0 + feature2_idx * stride_feature2_0 + feature1_idx * stride_feature1_0
    
    # Load input_0 and store to transposed location in output_0
    input_0_val = tl.load(input_0_ptr + input_offset, mask=mask_transpose, other=0.0)
    tl.store(output_0_ptr + output_offset, input_0_val, mask=mask_transpose)
    
    # Process scalar multiplication for input_1
    scalar_mask = offsets >= (batch_size * feature_dim1 * feature_dim2)
    scalar_offsets = offsets - (batch_size * feature_dim1 * feature_dim2)
    mask_scalar = scalar_offsets < n_elements_1
    
    # For scalar multiplication, process all elements that exceed the transpose range
    tl.store(output_1_ptr + scalar_offsets, 
             tl.load(input_1_ptr + offsets, mask=mask & mask_scalar, other=0.0) * scalar,
             mask=mask_scalar)

# Kernel wrapper
@torch.fx.wrap
def fused_operations_kernel_wrapper(in_0, in_1):
    # Get tensor shapes and strides
    shape_0 = in_0.shape
    strides_0 = in_0.stride()
    shape_1 = in_1.shape
    strides_1 = in_1.stride()
    
    # Extract dimensions for transpose
    batch_size = shape_0[:-2].numel() if len(shape_0) > 2 else 1
    feature_dim1 = shape_0[-2]
    feature_dim2 = shape_0[-1]
    
    # Calculate strides
    stride_batch_0 = strides_0[:-2][0] if len(shape_0) > 2 else 1
    stride_feature1_0 = strides_0[-2]
    stride_feature2_0 = strides_0[-1]
    
    n_elements_1 = in_1.numel()
    
    N = in_0.numel() + in_1.numel()  # Total elements to process
    BLOCK_SIZE = 1024  # Optimized for GPU occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_0 = torch.empty_like(in_0)
    output_1 = torch.empty_like(in_1)
    
    fused_operations_kernel[(num_programs,)](
        input_0_ptr=in_0,
        input_1_ptr=in_1,
        output_0_ptr=output_0,
        output_1_ptr=output_1,
        batch_size=batch_size,
        feature_dim1=feature_dim1,
        feature_dim2=feature_dim2,
        stride_batch_0=stride_batch_0,
        stride_feature1_0=stride_feature1_0,
        stride_feature2_0=stride_feature2_0,
        n_elements_1=n_elements_1,
        scalar=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_0, output_1

# Replacement function (returns kernel wrapper function)
def replacement_func():
    return fused_operations_kernel_wrapper