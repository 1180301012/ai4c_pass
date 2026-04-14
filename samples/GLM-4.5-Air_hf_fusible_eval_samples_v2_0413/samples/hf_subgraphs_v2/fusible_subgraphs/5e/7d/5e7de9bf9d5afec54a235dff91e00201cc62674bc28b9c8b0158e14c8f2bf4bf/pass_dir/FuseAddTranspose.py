import torch
import triton
import triton.language as tl

# Pattern matching function - Addition + transpose fusion
def pattern(a, b):
    # Addition operation
    add_out = a + b
    # Transpose operation
    trans_out = add_out.transpose(1, 2)
    return trans_out

# Argument extraction function
def replacement_args(a, b):
    return (a, b)

# Optimized kernel - Addition + transpose fusion
@triton.jit
def add_transpose_kernel(
    a_ptr, b_ptr, out_ptr,
    batch_size, dim1, dim2,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_DIM1: tl.constexpr,
    BLOCK_SIZE_DIM2: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    dim1_id = tl.program_id(1)
    dim2_id = tl.program_id(2)
    
    # Calculate output pointers (transposed order)
    out_offset = batch_id * dim1 * dim2 + dim1_id * dim2 + dim2_id
    
    # Calculate input pointers for both tensors
    a_offset = batch_id * dim1 * dim2 + dim1_id * dim2 + dim2_id
    b_offset = batch_id * dim1 * dim2 + dim1_id * dim2 + dim2_id
    
    if batch_id < batch_size and dim1_id < dim1 and dim2_id < dim2:
        # Load elements from both tensors
        a_val = tl.load(a_ptr + a_offset)
        b_val = tl.load(b_ptr + b_offset)
        
        # Add elements
        add_val = a_val + b_val
        
        # Store result in transposed position
        tl.store(out_ptr + out_offset, add_val)

# Kernel wrapper
@torch.fx.wrap
def add_transpose_fused(a, b):
    # Get tensor shapes (should be same)
    assert a.shape == b.shape, "Input tensors must have same shape"
    batch_size, dim1, dim2 = a.shape
    
    # Optimal block sizes for better GPU occupancy
    BLOCK_SIZE_BATCH = 1  # Usually batch is small
    BLOCK_SIZE_DIM1 = 32
    BLOCK_SIZE_DIM2 = 32
    
    # Grid configuration
    grid = (batch_size, (dim1 + BLOCK_SIZE_DIM1 - 1) // BLOCK_SIZE_DIM1,
            (dim2 + BLOCK_SIZE_DIM2 - 1) // BLOCK_SIZE_DIM2)
    
    # Output tensor with transposed dimensions (swap dim1 and dim2)
    out = torch.empty((batch_size, dim2, dim1), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    add_transpose_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch_size,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_DIM1=BLOCK_SIZE_DIM1,
        BLOCK_SIZE_DIM2=BLOCK_SIZE_DIM2
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return add_transpose_fused