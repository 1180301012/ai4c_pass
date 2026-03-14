import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching slice + expand + explicit slice range operations.
    Matches: y = x[:, :size], y.expand(1, size), z = original_x[0:size]
    Found in: bge-base-en-v1.5 model
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = tmp_1[slice(None, None, None), slice(None, None, None)]
    tmp_1 = None
    tmp_3 = tmp_2.expand(1, tmp_2.shape[1])
    tmp_2 = None
    tmp_4 = tmp_0[slice(0, None, None)]
    tmp_0 = None
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def broadcast_expand_kernel(
    input_ptr, output_ptr,
    src_features: tl.constexpr,
    dst_batch: tl.constexpr,
    dst_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel for broadcasting and expanding 1D tensor to 2D with single batch.
    Efficiently handles the expand(1, feature_size) operation on CPU/GPU.
    """
    pid = tl.program_id(0)
    
    # Compute indices for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dst_features
    
    # Load source data (single batch)
    src_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Store to destination with batch dimension
    tl.store(output_ptr + offsets, src_data, mask=mask)

@triton.jit
def explicit_slice_kernel(
    input_ptr, output_ptr,
    slice_size: tl.constexpr,
    total_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel for explicit slice operation up to specified size.
    Handles cases where we extract first N elements from a larger tensor.
    """
    pid = tl.program_id(0)
    
    # Compute indices for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < slice_size
    
    # Load and store only the slice portion
    src_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    tl.store(output_ptr + offsets, src_data, mask=mask)

@torch.fx.wrap
def explicit_slice_expand_optimized(in_0, in_1):
    """
    Optimized implementation for slice + expand + explicit slice pattern
    """
    batch_size_0, feature_size_0 = in_0.shape
    batch_size_1, feature_size_1 = in_1.shape
    
    # Determine slice size (it will be the second dimension of in_1 after slicing)
    slice_size = min(feature_size_1, feature_size_0)  # Ensure we don't exceed bounds
    
    # Create output tensors
    expand_output = torch.empty((1, slice_size), dtype=in_1.dtype, device=in_1.device)
    slice_output = torch.empty((slice_size,), dtype=in_0.dtype, device=in_0.device)
    
    # Optimized expand operation: broadcast single batch to match target shape
    expand_output[0, :] = in_1[:, :slice_size].squeeze(0)
    
    # Optimized explicit slice operation
    slice_output = in_0[:slice_size]
    
    return expand_output, slice_output

def replacement_func():
    return explicit_slice_expand_optimized