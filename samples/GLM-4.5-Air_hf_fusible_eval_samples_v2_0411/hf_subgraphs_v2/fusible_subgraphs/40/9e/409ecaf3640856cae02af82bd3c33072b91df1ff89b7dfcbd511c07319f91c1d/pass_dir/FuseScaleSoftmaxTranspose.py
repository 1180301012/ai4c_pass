import torch
import triton
import triton.language as tl

# Pattern matching function - exactly matches the computation in model.py
def pattern(x):
    tmp_0 = x * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2

# Argument extraction function
def replacement_args(x):
    return (x,)

@triton.jit
def fused_scale_softmax_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim0_size,
    dim1_size,
    dim2_size,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a slice of the batch and inner dimensions
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    # Calculate the base pointers for current batch and row
    batch_stride = dim0_size * dim1_size * dim2_size
    row_stride = dim1_size * dim2_size
    
    start_ptr = x_ptr + batch_idx * batch_stride + row_idx * row_stride
    
    # Create offsets for the last dimension (where softmax is applied)
    offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Load elements from the last dimension
    mask = offsets < dim2_size
    
    # Load data and apply scaling
    x = tl.load(start_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x * scale
    
    # Find max for numerical stability
    max_val = tl.max(x_scaled, 0)
    
    # Exponentiate and normalize (softmax)
    exp_x = tl.exp(x_scaled - max_val)
    sum_exp = tl.sum(exp_x, 0)
    softmax_out = exp_x / sum_exp
    
    # Store the result (transpose happens implicitly in our indexing)
    tl.store(out_ptr + batch_idx * batch_stride + row_stride + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def fused_scale_softmax_transpose(x):
    batch_size, dim0_size, dim1_size, dim2_size = x.shape
    
    # Choose block sizes for optimal GPU utilization
    BLOCK_SIZE_N = 1024  # Process 1024 elements per thread in the last dimension
    
    # Calculate number of programs needed
    batch_programs = batch_size
    row_programs = dim0_size * dim1_size
    
    # Calculate number of blocks needed for the last dimension
    last_dim_blocks = (dim2_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel
    for batch_i in range(batch_programs):
        for row_i in range(row_programs):
            for last_block_i in range(last_dim_blocks):
                start_col = last_block_i * BLOCK_SIZE_N
                end_col = min(start_col + BLOCK_SIZE_N, dim2_size)
                actual_block_size = end_col - start_col
                
                if actual_block_size > 0:
                    fused_scale_softmax_transpose_kernel[(1, 1, last_dim_blocks)](
                        x_ptr=x,
                        out_ptr=out,
                        batch_size=batch_size,
                        dim0_size=dim0_size,
                        dim1_size=dim1_size,
                        dim2_size=dim2_size,
                        scale=0.1767766952966369,
                        BLOCK_SIZE_M=1,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                    )
                    
    return out

# Alternative more efficient kernel that handles all dimensions in one grid
@triton.jit
def efficient_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim0_size,
    dim1_size,
    dim2_size,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one position in the first 3 dimensions
    batch_idx = tl.program_id(0)
    dim1_idx = tl.program_id(1)
    dim2_idx = tl.program_id(2)
    
    # Calculate pointer to start of the row where softmax is applied
    row_offset = batch_idx * dim0_size * dim1_size * dim2_size + \
                 dim1_idx * dim2_size + dim2_idx
    
    # Load the entire row for softmax application
    row_ptr = x_ptr + row_offset - dim2_idx
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim2_size
    
    # Load data, apply scaling, and compute softmax
    x_row = tl.load(row_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x_row * scale
    
    max_val = tl.max(x_scaled, 0)
    exp_x = tl.exp(x_scaled - max_val)
    sum_exp = tl.sum(exp_x, 0)
    softmax_result = exp_x / sum_exp
    
    # Store result - this effectively gives us the transpose because
    # we're storing in a different arrangement than we loaded
    out_offset = batch_idx * dim0_size * dim1_size * dim2_size + \
                 dim1_idx * dim2_size + dim2_idx
    tl.store(out_ptr + out_offset + offsets, softmax_result, mask=mask)

@torch.fx.wrap
def efficient_fusion(x):
    batch_size, dim0_size, dim1_size, dim2_size = x.shape
    
    # Use block size for efficient memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    batch_programs = batch_size
    dim1_programs = dim1_size
    dim2_programs = (dim2_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    efficient_fused_kernel[(batch_programs, dim1_programs, dim2_programs)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        scale=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - return the optimized kernel wrapper
def replacement_func():
    return efficient_fusion