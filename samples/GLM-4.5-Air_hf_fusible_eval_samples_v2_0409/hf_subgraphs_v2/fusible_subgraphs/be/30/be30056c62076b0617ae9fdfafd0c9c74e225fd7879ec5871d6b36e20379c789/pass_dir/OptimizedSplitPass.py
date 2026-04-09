import torch
import triton
import triton.language as tl

# Pattern matching for the split operation - this is allowed since it's in pattern() function
def pattern(tmp_1):
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1] 
    tmp_5 = split[2]
    return tmp_3, tmp_4, tmp_5

# Extract arguments 
def replacement_args(tmp_1):
    return (tmp_1,)

# Optimized Triton kernel for split operation
@triton.jit
def split_optimized_kernel(
    input_ptr,
    output3_ptr,
    output4_ptr,
    output5_ptr,
    batch_size,
    seq_len,
    total_dims,
    block_size: tl.constexpr,
):
    # Calculate total elements
    total_elements = batch_size * seq_len * total_dims
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate dimension index to determine which chunk this element belongs to
    dim_idx = offsets % total_dims
    
    # Determine which chunk and calculate output index
    if dim_idx < 512:  # First chunk
        chunk_idx = offsets // (total_dims // 512)
        output_idx = (chunk_idx // 512) * (512 * 512) + (chunk_idx % 512) * 512 + dim_idx
        tl.store(output3_ptr + output_idx, input_data, mask=mask)
    elif dim_idx < 1024:  # Second chunk (512+512)
        adjusted_idx = dim_idx - 512
        chunk_idx = offsets // (total_dims // 512)
        output_idx = (chunk_idx // 512) * (512 * 512) + (chunk_idx % 512) * 512 + adjusted_idx
        tl.store(output4_ptr + output_idx, input_data, mask=mask)
    else:  # Third chunk (1024+128)
        adjusted_idx = dim_idx - 1024
        chunk_idx = offsets // (total_dims // 128)
        output_idx = (chunk_idx // 128) * (512 * 128) + (chunk_idx % 128) * 512 + adjusted_idx
        tl.store(output5_ptr + output_idx, input_data, mask=mask)

@torch.fx.wrap
def optimized_split_function(tmp_1):
    batch_size, seq_len, total_dims = tmp_1.shape
    
    # Allocate output tensors using allowed APIs
    tmp_3 = torch.empty(batch_size, seq_len, 512, dtype=tmp_1.dtype, device=tmp_1.device)
    tmp_4 = torch.empty(batch_size, seq_len, 512, dtype=tmp_1.dtype, device=tmp_1.device)
    tmp_5 = torch.empty(batch_size, seq_len, 128, dtype=tmp_1.dtype, device=tmp_1.device)
    
    total_elements = batch_size * seq_len * total_dims
    block_size = 1024
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Launch Triton kernel
    split_optimized_kernel[(num_programs,)](
        input_ptr=tmp_1,
        output3_ptr=tmp_3,
        output4_ptr=tmp_4,
        output5_ptr=tmp_5,
        batch_size=batch_size,
        seq_len=seq_len,
        total_dims=total_dims,
        block_size=block_size,
    )
    
    # Return the split results in the same order as the original
    return tmp_3, tmp_4, tmp_5

def replacement_func():
    return optimized_split_function