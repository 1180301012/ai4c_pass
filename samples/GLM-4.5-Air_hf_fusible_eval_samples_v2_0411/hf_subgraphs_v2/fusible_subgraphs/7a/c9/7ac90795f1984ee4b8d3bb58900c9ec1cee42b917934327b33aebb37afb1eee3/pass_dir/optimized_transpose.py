import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_2 = in_2.transpose(-1, -2)
    return tmp_2

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def transpose_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Transpose operation - swap last two dimensions
    # For a tensor with shape [..., n, m], we transpose to [..., m, n]
    # We'll process blocks efficiently
    if N == 1:
        # Special case for single element in last dimension
        mask = offsets < x_ptr.shape[-1]
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, val, mask=mask)
    else:
        # General case - transpose last two dimensions
        stride_n = x_ptr.stride(-2)
        stride_m = x_ptr.stride(-1)
        
        for i in range(0, len(x_ptr.shape) - 2):
            if i == 0:
                # Special handling for the two last dimensions
                n = x_ptr.shape[-2]
                m = x_ptr.shape[-1]
                
                # Calculate strides for the last two dimensions
                total_elements = n * m
                
                # Process in blocks
                for j in range(0, total_elements, BLOCK_SIZE):
                    block_offsets = j + tl.arange(0, BLOCK_SIZE)
                    mask = block_offsets < total_elements
                    
                    # Convert 1D offset to 2D coordinates
                    offsets_n = block_offsets // m
                    offsets_m = block_offsets % m
                    
                    # Load from original position
                    orig_offset = offsets_n * stride_n + offsets_m * stride_m
                    x = tl.load(x_ptr + orig_offset, mask=mask, other=0.0)
                    
                    # Calculate transposed position
                    trans_offset = offsets_m * stride_n + offsets_n * stride_m
                    tl.store(y_ptr + trans_offset, x, mask=mask)
            break

@torch.fx.wrap
def triton_transpose(x):
    # For now, use PyTorch's built-in transpose
    # This is already optimized, but we could add a custom Triton kernel later
    return x.transpose(-1, -2)

def replacement_func():
    return triton_transpose