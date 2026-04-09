import torch
import triton
import triton.language as tl

# Pattern matching function for transpose operation
def pattern(in_0):
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel for transpose of last two dimensions
@triton.jit
def transpose_last_two_dims_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    feature_dim1,
    feature_dim2,
    stride_batch,
    stride_feature1,
    stride_feature2,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * feature_dim1 * feature_dim2)
    
    # Calculate multi-dimensional indices
    batch_idx = offsets // (feature_dim1 * feature_dim2)
    remaining = offsets % (feature_dim1 * feature_dim2)
    feature1_idx = remaining // feature_dim2
    feature2_idx = remaining % feature_dim2
    
    # Transpose: swap feature1 and feature2 indices
    # For transpose(-2, -1), we swap the last two dimensions
    input_offset = batch_idx * stride_batch + feature1_idx * stride_feature1 + feature2_idx * stride_feature2
    output_offset = batch_idx * stride_batch + feature2_idx * stride_feature2 + feature1_idx * stride_feature1
    
    # Load input data
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to transposed location
    tl.store(output_ptr + output_offset, input_val, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def transpose_last_two_dims_kernel_wrapper(input_tensor):
    # Get tensor shape and strides
    shape = input_tensor.shape
    strides = input_tensor.stride()
    
    # Extract dimensions for last two transpose (-2, -1)
    batch_size = shape[:-2].numel() if len(shape) > 2 else 1
    feature_dim1 = shape[-2]
    feature_dim2 = shape[-1]
    
    # Calculate strides for multi-dimensional access
    stride_batch = strides[:-2][0] if len(shape) > 2 else 1
    stride_feature1 = strides[-2]
    stride_feature2 = strides[-1]
    
    N = input_tensor.numel()
    BLOCK_SIZE = 1024  # Optimized for GPU occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_tensor = torch.empty_like(input_tensor)
    
    transpose_last_two_dims_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        batch_size=batch_size,
        feature_dim1=feature_dim1,
        feature_dim2=feature_dim2,
        stride_batch=stride_batch,
        stride_feature1=stride_feature1,
        stride_feature2=stride_feature2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function (returns kernel wrapper function)
def replacement_func():
    return transpose_last_two_dims_kernel_wrapper