import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match torch.cat operation.
    """
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def cat_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    in_0_size,
    in_1_size,
    dim1_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel to concatenate two tensors along dimension 0.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = (in_0_size + in_1_size) * dim1_size
    mask = offsets < total_elements
    
    row = offsets // dim1_size
    col = offsets % dim1_size
    
    # If row < in_1_size, read from in_1, else read from in_0
    in_1_mask = (row < in_1_size) & mask
    in_0_mask = (row >= in_1_size) & mask
    
    # Read from in_1
    in_1_idx = row * dim1_size + col
    val_1 = tl.load(in_1_ptr + in_1_idx, mask=in_1_mask, other=0.0)
    
    # Read from in_0
    in_0_row = row - in_1_size
    in_0_idx = in_0_row * dim1_size + col
    val_0 = tl.load(in_0_ptr + in_0_idx, mask=in_0_mask, other=0.0)
    
    # Combine
    result = tl.where(in_1_mask, val_1, val_0)
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_cat(in_0, in_1):
    """
    Optimized concatenation using Triton.
    """
    in_0_shape = in_0.shape
    in_1_shape = in_1.shape
    out_shape = (in_0_shape[0] + in_1_shape[0], in_0_shape[1])
    out = torch.empty(out_shape, dtype=in_0.dtype, device='cuda')
    
    BLOCK_SIZE = 256
    total_elements = out_shape[0] * out_shape[1]
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cat_kernel[(num_programs,)](
        in_0,
        in_1,
        out,
        in_0_shape[0],
        in_1_shape[0],
        in_0_shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_cat