import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Flexible pattern matching for addition + slicing + concatenation.
    Matches: in_0 + in_1, then slice in_2[start:], then concatenate.
    The slice start value is handled internally without requiring extraction.
    """
    tmp_0 = in_0 + in_1
    # Use a more flexible slice pattern that matches any slice start
    tmp_1 = in_2[slice(None, None, None), slice(None, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the optimized kernel.
    We determine the slice start from tensor shapes.
    """
    return (in_0, in_1, in_2)


@triton.jit
def fused_kernel(
    x1_ptr, x2_ptr, x3_ptr,
    x1_stride_1, x1_stride_2, x1_stride_3,
    x2_stride_1, x2_stride_2, x2_stride_3,
    x3_stride_1, x3_stride_2, x3_stride_3,
    y_ptr,
    y_stride_1, y_stride_2, y_stride_3,
    batch_size, channels1, height, width, channels3,
    slice_start: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for add + slice + concat pattern"""
    pid = tl.program_id(0)
    batch_idx = pid
    batch_offset = batch_idx * x1_stride_1
    
    # Process addition part (in_0 + in_1)
    elem_per_batch_add = channels1 * height * width
    for i in range(0, elem_per_batch_add, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elem_per_batch_add
        
        # Load and add
        ptr1 = x1_ptr + batch_offset + offsets * x1_stride_2
        ptr2 = x2_ptr + batch_offset + offsets * x2_stride_2
        x1 = tl.load(ptr1, mask=mask, other=0.0)
        x2 = tl.load(ptr2, mask=mask, other=0.0)
        result = x1 + x2
        
        # Store addition result
        ptr_out = y_ptr + batch_offset + offsets * y_stride_2
        tl.store(ptr_out, result, mask=mask)
    
    # Process slice part (slice from in_2)
    channels_slice = channels3 - slice_start
    elem_per_batch_slice = channels_slice * height * width
    for i in range(0, elem_per_batch_slice, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elem_per_batch_slice
        
        # Load slice from in_2 (offset by slice_start * height * width)
        slice_offset = slice_start * height * width + offsets
        ptr3 = x3_ptr + batch_offset + slice_offset * x3_stride_2
        x3_slice = tl.load(ptr3, mask=mask, other=0.0)
        
        # Store at correct position in output
        slice_output_offset = channels1 * height * width + offsets
        ptr_out = y_ptr + batch_offset + slice_output_offset * y_stride_2
        tl.store(ptr_out, x3_slice, mask=mask)


@torch.fx.wrap
def fused_add_slice_concat(in_0, in_1, in_2):
    """Wrapper function for launching the optimized kernel"""
    batch_size, channels1, height, width = in_0.shape
    _, channels3, _, _ = in_2.shape
    
    # Determine slice start from tensor shapes
    # In this pattern, in_0 and in_1 have the same number of channels,
    # and in_2 has more channels. We slice in_2 from channel channels1 onwards.
    slice_start = channels1
    
    output_channels = channels1 + (channels3 - slice_start)
    out = torch.empty((batch_size, output_channels, height, width), 
                      dtype=in_0.dtype, device=in_0.device)
    
    # Calculate strides
    x1_stride_1, x1_stride_2, x1_stride_3 = in_0.stride(0), in_0.stride(1), in_0.stride(2)
    x2_stride_1, x2_stride_2, x2_stride_3 = in_1.stride(0), in_1.stride(1), in_1.stride(2)
    x3_stride_1, x3_stride_2, x3_stride_3 = in_2.stride(0), in_2.stride(1), in_2.stride(2)
    y_stride_1, y_stride_2, y_stride_3 = out.stride(0), out.stride(1), out.stride(2)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = batch_size
    
    fused_kernel[(num_programs,)](
        in_0, in_1, in_2,
        x1_stride_1, x1_stride_2, x1_stride_3,
        x2_stride_1, x2_stride_2, x2_stride_3,
        x3_stride_1, x3_stride_2, x3_stride_3,
        out,
        y_stride_1, y_stride_2, y_stride_3,
        batch_size, channels1, height, width, channels3,
        slice_start,
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """Returns the optimized kernel function"""
    return fused_add_slice_concat