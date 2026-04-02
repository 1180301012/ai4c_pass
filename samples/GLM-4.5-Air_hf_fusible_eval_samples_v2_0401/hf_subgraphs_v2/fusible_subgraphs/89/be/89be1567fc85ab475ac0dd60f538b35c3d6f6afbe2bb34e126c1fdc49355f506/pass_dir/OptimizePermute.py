import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # The exact computation from the graphs - permute dims (0, 2, 1)
    tmp_6 = input_tensor.permute(0, 2, 1)
    return tmp_6

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_permute_kernel(
    input_ptr,
    output_ptr, 
    dim0_size,
    dim1_size, 
    dim2_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a tile of the output matrix
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Create offsets within the tile
    row_offsets = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure we don't go out of bounds
    row_mask = row_offsets < dim0_size
    col_mask = col_offsets < dim2_size  # Note: permute(0,2,1) - dim2 becomes the new dim1
    
    # Create 2D indices for memory access
    # Original shape: [dim0, dim1, dim2] -> permute: [dim0, dim2, dim1]
    # We need to map (m, n) in output to (m, ?, n) in input
    input_row_indices = row_offsets[:, None]  # dim0
    input_col_indices = col_offsets[None, :]  # dim2
    input_middle_slice = tl.arange(0, dim1_size)  # dim1
    
    # Flatten indices for memory access
    input_indices = (input_row_offsets[:, None] * dim1_size * dim2_size + 
                    input_middle_slice[None, None, :] * dim2_size + 
                    input_col_offsets[None, None, :])
    
    # We'll process this differently - use a simpler approach for the permute
    # Since we know the exact permutation (0,2,1), we can optimize memory access patterns
    
    # Create output indices
    output_indices = (row_offsets[:, None] * dim1_size * dim2_size + 
                     input_middle_slice[None, None, :] * dim2_size + 
                     col_offsets[None, None, :])
    
    # Calculate batch size (this is the dim0 size which stays in place)
    batch_size = dim0_size
    
    # Process each slice in the batch
    if m < batch_size and n < dim2_size:
        # Process a block for each batch element
        for batch_idx in range(batch_size):
            # Calculate input and output indices for this batch element
            base_input_offset = batch_idx * dim1_size * dim2_size
            base_output_offset = batch_idx * dim1_size * dim2_size
            
            # For permute(0,2,1): 
            # input[batch, i, j] -> output[batch, j, i]
            for i in range(BLOCK_SIZE_N):
                for j in range(BLOCK_SIZE_M):
                    if batch_idx < dim0_size and i < dim2_size and j < dim1_size:
                        # Load input data
                        src_offset = base_input_offset + j * dim2_size + i
                        data = tl.load(input_ptr + src_offset, other=0.0)
                        
                        # Store to permuted location
                        dst_offset = base_output_offset + i * dim1_size + j 
                        tl.store(output_ptr + dst_offset, data)

@torch.fx.wrap
def optimized_permute(input_tensor):
    # Get input tensor shape
    dim0_size, dim1_size, dim2_size = input_tensor.shape
    
    # Allocate output tensor with permuted shape: [dim0, dim2, dim1]
    output_shape = [dim0_size, dim2_size, dim1_size]
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set block size for dim1 processing 
    BLOCK_SIZE_D1 = 256  # Process this many elements in dim1 direction
    
    # More efficient implementation with proper memory layout
    @triton.jit
    def optimized_permute_kernel_efficient(
        input_ptr,
        output_ptr, 
        dim0_size,
        dim1_size, 
        dim2_size,
        BLOCK_SIZE_D1: tl.constexpr,
    ):
        # Get program IDs
        pid = tl.program_id(0)
        
        # Map to batch and column coordinates
        col_idx = pid // dim0_size  # dim2 position
        batch_idx = pid % dim0_size   # dim0 position
        
        if batch_idx >= dim0_size or col_idx >= dim2_size:
            return
            
        # Process entire row in dim1 dimension with fixed block size
        offsets = tl.arange(0, BLOCK_SIZE_D1)
        mask = offsets < dim1_size
        
        # Calculate memory offsets
        input_base = batch_idx * dim1_size * dim2_size + col_idx * dim1_size
        output_base = batch_idx * dim1_size * dim2_size + col_idx * dim1_size
        
        # Load input data: input[batch_idx, :, col_idx]
        input_data = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0)
        
        # Store to output: output[batch_idx, col_idx, :] (which is the same layout after permutation)
        tl.store(output_ptr + output_base + offsets, input_data, mask=mask)
    
    # Launch kernel
    total_work_items = dim0_size * dim2_size
    optimized_permute_kernel_efficient[(total_work_items,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        BLOCK_SIZE_D1=BLOCK_SIZE_D1,
    )
    
    return output_tensor

def replacement_func():
    return optimized_permute