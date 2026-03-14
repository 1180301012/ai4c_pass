import torch
import triton
import triton.language as tl

@triton.jit
def view_transpose_kernel_1_64(
    input_ptr, output_ptr,
    batch_size, seq_len, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for view(..,1,64) + transpose(1,2) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Input is [batch_size, seq_len, 64]
    stride_batch = seq_len * 64
    stride_seq = 64
    batch_idx = offsets // stride_batch
    rem = offsets % stride_batch
    seq_idx = rem // stride_seq
    feat_idx = rem % stride_seq
    
    # Load input element
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute output position for transpose pattern: [batch_size, 1, seq_len, 64]
    # Transposing dim1 <-> dim2 means we swap the middle dimensions
    # Input dims: [batch_size, seq_len, 64] -> view to [batch_size, seq_len, 1, 64] -> transpose to [batch_size, 1, seq_len, 64]
    # Output linear index calculation
    output_stride_batch = seq_len * 64
    output_stride_seq = 64
    output_stride_batch2 = 1 * seq_len * 64
    
    # For [batch_size, 1, seq_len, 64]: 
    # batch_idx maps to batch_idx
    # seq_idx maps to seq_idx 
    # feat_idx maps to feat_idx
    # The middle dimension (1) is always 0
    output_offset = (batch_idx * output_stride_batch2 + 
                    0 * seq_len * 64 + 
                    seq_idx * 64 + 
                    feat_idx)
    
    tl.store(output_ptr + output_offset, val, mask=mask)

@triton.jit  
def view_transpose_kernel_5_32(
    input_ptr, output_ptr,
    batch_size, seq_len, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for view(..,5,32) + transpose(1,2) pattern."""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (total_elements + block_size - 1) // block_size
    
    if pid >= num_blocks:
        return
        
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Input is [batch_size, seq_len, 32]
    stride_batch = seq_len * 32
    stride_seq = 32
    batch_idx = offsets // stride_batch
    rem = offsets % stride_batch
    seq_idx = rem // stride_seq
    feat_idx = rem % stride_seq
    
    # Load input element
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # For [batch_size, 5, seq_len, 32] output:
    output_stride_batch = 5 * seq_len * 32
    output_stride_batch2 = 1 * seq_len * 32
    
    output_offset = (batch_idx * output_stride_batch2 + 
                    seq_idx * seq_len * 32 + 
                    feat_idx)
    
    tl.store(output_ptr + output_offset, val, mask=mask)

@torch.fx.wrap
def optimized_view_transpose_1_64(x):
    """Optimized view(..,1,64) + transpose(1,2) operation for pattern 1."""
    batch_size, seq_len, _ = x.shape
    total_elements = batch_size * seq_len * 64
    
    # Compute output shape: [batch_size, 1, seq_len, 64]  
    out_shape = (batch_size, 1, seq_len, 64)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    view_transpose_kernel_1_64[grid](
        x, out,
        batch_size, seq_len, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def optimized_view_transpose_5_32(x):
    """Optimized view(..,5,32) + transpose(1,2) operation for pattern 2."""
    batch_size, seq_len, _ = x.shape
    total_elements = batch_size * seq_len * 32
    
    # Compute output shape: [batch_size, 5, seq_len, 32]
    out_shape = (batch_size, 5, seq_len, 32) 
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    view_transpose_kernel_5_32[grid](
        x, out,
        batch_size, seq_len, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(x, view_args, transpose_dims):
    """Match view + transpose pattern."""
    tmp_0 = x.view(*view_args)
    tmp_1 = tmp_0.transpose(*transpose_dims)
    return tmp_1

def replacement_args(x, view_args=(32, -1, 1, 64), transpose_dims=(1, 2)):
    """Extract arguments for replacement."""
    # Just return the arguments needed, no conditional logic
    return (x, view_args, transpose_dims)

def replacement_func():
    """Return optimized function."""
    def optimized_wrapper(x, view_args=(32, -1, 1, 64), transpose_dims=(1, 2)):
        # Determine which kernel to use based on feature dimension
        if view_args[-1] == 64:
            return optimized_view_transpose_1_64(x)
        else:
            return optimized_view_transpose_5_32(x)
    
    return optimized_wrapper