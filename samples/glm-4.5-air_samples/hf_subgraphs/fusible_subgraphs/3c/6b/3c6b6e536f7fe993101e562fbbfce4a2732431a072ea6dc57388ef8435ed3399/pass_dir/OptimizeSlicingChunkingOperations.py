import torch
import triton
import triton.language as tl

# Pattern matching function for slicing and chunking operations
def pattern(in_0, in_1, in_3):
    # Slicing operations - select the appropriate number of elements based on actual tensor size
    # This pattern handles variable slice sizes across different graphs
    seq_len = in_0.shape[2]  # Get the actual sequence length
    
    tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, seq_len, None), slice(None, None, None)]
    tmp_6 = in_1[slice(None, None, None), slice(None, None, None), slice(None, seq_len, None), slice(None, None, None)]
    tmp_7 = in_3 * tmp_5
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return (tmp_6, tmp_7, tmp_9, tmp_10)

# Argument extraction function
def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

# Optimized kernel for slicing and chunking operations
@triton.jit
def slice_chunk_kernel(
    in_0_ptr, in_1_ptr, in_3_ptr,
    out_6_ptr, out_7_ptr, out_9_ptr, out_10_ptr,
    batch_size_0, seq_len_0, hidden_dim_0,
    batch_size_3, seq_len_3, hidden_dim_3,
    slice_size: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Calculate program IDs
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Process hidden dimensions
    hidden_offsets = tl.arange(0, BLOCK_SIZE_Y)
    mask = hidden_offsets < hidden_dim_3
    
    # Base pointers for this batch/seq position
    base_0 = batch_id * seq_len_0 * hidden_dim_0 + seq_id * hidden_dim_0
    base_1 = batch_id * seq_len_0 * hidden_dim_0 + seq_id * hidden_dim_0
    base_3 = batch_id * seq_len_3 * hidden_dim_3 + seq_id * hidden_dim_3
    
    # Load input tensors with proper masking
    in_0_val = tl.load(in_0_ptr + base_0 + hidden_offsets, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + base_1 + hidden_offsets, mask=mask, other=0.0)
    in_3_val = tl.load(in_3_ptr + base_3 + hidden_offsets, mask=mask, other=0.0)
    
    # tmp_6 = sliced in_1 (directly from in_1 since we're taking all)
    # tmp_7 = in_3 * in_0 (broadcasting in_0 to match in_3 shape)
    out_7_val = in_3_val * in_0_val
    
    # For chunking: split in_3 along last dimension (dim=-1)
    chunk_size = hidden_dim_3 // 2
    
    # Store outputs in their respective output tensors
    tl.store(out_6_ptr + base_1 + hidden_offsets, in_1_val, mask=mask) 
    tl.store(out_7_ptr + base_3 + hidden_offsets, out_7_val, mask=mask)
    
    # Handle chunking by selectively storing data
    first_half_mask = hidden_offsets < chunk_size
    second_half_mask = hidden_offsets >= chunk_size
    
    # First chunk: store elements 0:chunk_size to out_9[0:chunk_size]
    first_half_data = tl.where(first_half_mask, in_3_val, 0.0)
    tl.store(out_9_ptr + base_3 + hidden_offsets, first_half_data, mask=first_half_mask)
    
    # Second chunk: store elements chunk_size:hidden_dim_3 to out_10[0:hidden_dim_3-chunk_size]
    second_half_data = tl.where(second_half_mask, in_3_val, 0.0)
    second_half_output_offset = hidden_offsets - chunk_size
    tl.store(out_10_ptr + base_3 + second_half_output_offset, second_half_data, mask=second_half_mask)

@torch.fx.wrap 
def optimized_slice_chunk(in_0, in_1, in_3):
    # Get tensor shapes
    shape_0 = in_0.shape
    shape_3 = in_3.shape
    
    # Create output tensors
    tmp_6 = in_1  # This will be the sliced in_1
    tmp_7 = torch.empty_like(in_3)  # in_3 * in_0 (broadcasted)
    
    # For chunking: split in_3 into 2 chunks along last dimension
    chunk_size = in_3.shape[-1] // 2
    tmp_9 = torch.empty((in_3.shape[0], in_3.shape[1], in_3.shape[2], chunk_size), dtype=in_3.dtype, device=in_3.device)
    tmp_10 = torch.empty((in_3.shape[0], in_3.shape[1], in_3.shape[2], chunk_size), dtype=in_3.dtype, device=in_3.device)
    
    # Set up Triton kernel launch parameters
    BLOCK_SIZE_Y = 64   # Hidden dimension chunk size
    
    # Create grid: (batch_size_3, seq_len_3) - each program handles one batch and sequence position
    grid = (shape_3[0], shape_3[2])
    
    # Launch kernel
    slice_chunk_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_3_ptr=in_3,
        out_6_ptr=tmp_6,
        out_7_ptr=tmp_7,
        out_9_ptr=tmp_9,
        out_10_ptr=tmp_10,
        batch_size_0=shape_0[0],
        seq_len_0=shape_0[2],
        hidden_dim_0=shape_0[3],
        batch_size_3=shape_3[0],
        seq_len_3=shape_3[2],
        hidden_dim_3=shape_3[3],
        slice_size=512,  # Only used for clarity
        BLOCK_SIZE_X=1,  # Unused but required
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return (tmp_6, tmp_7, tmp_9, tmp_10)

# Replacement function
def replacement_func():
    return optimized_slice_chunk