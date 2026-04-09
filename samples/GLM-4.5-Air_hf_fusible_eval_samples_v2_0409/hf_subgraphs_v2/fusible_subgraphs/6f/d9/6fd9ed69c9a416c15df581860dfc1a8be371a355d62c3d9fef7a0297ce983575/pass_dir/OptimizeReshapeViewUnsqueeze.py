import torch
import triton
import triton.language as tl

def pattern(in_4, tmp_6):
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_13 = tmp_6.unsqueeze(-2)
    return tmp_9, tmp_13

def replacement_args(in_4, tmp_6):
    return (in_4, tmp_6)

@triton.jit
def reshape_unsqueeze_kernel(
    input_ptr,      # in_4: [1, 150, 1, 512]  
    unsqueeze_ptr,  # tmp_6: [300, 256]
    out_reshape_ptr, # tmp_9: [300, 1, 256]
    out_unsqueeze_ptr, # tmp_13: [300, 1, 256]
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program ID for processing rows
    pid = tl.program_id(0)
    
    # Number of elements to process (we have 300 rows total)
    total_rows = 300
    row_start = pid * BLOCK_SIZE_M
    row_end = min((pid + 1) * BLOCK_SIZE_M, total_rows)
    rows = row_end - row_start
    
    # Process reshape: [1, 150, 1, 512] -> [300, 1, 256]
    output_offset = row_start * 256  # Each row has 256 elements
    
    for i in range(rows):
        # Calculate original position in [1, 150, 1, 512] tensor
        original_row = row_start + i
        # Original structure: [1, 150, 1, 512] -> [150, 512] 
        # Then reshape to [300, 1, 256] means:
        # Each original [150, 512] row becomes 2 rows in output
        orig_pos_in_150x512 = original_row // 2
        orig_channel_offset = (original_row % 2) * 256
        
        # Load channel data from original position
        orig_offset = orig_pos_in_150x512 * 512 + orig_channel_offset
        input_data = tl.load(input_ptr + orig_offset, mask=orig_offset < 150 * 512, other=0.0)
        
        # Store in reshaped position
        reshape_offset = (row_start + i) * 256
        tl.store(out_reshape_ptr + reshape_offset, input_data, mask=reshape_offset < 300 * 256)
    
    # Process unsqueeze: [300, 256] -> [300, 1, 256]
    unsqueeze_offset = row_start * 256
    for i in range(rows):
        row_offset = unsqueeze_offset + i * 256
        row_data = tl.load(unsqueeze_ptr + row_offset, mask=row_offset < 300 * 256, other=0.0)
        # Store with implicit middle dimension of size 1
        tl.store(out_unsqueeze_ptr + row_offset, row_data, mask=row_offset < 300 * 256)

@torch.fx.wrap
def optimized_reshape_unsqueeze(in_4, tmp_6):
    # Calculate output shapes
    tmp_9_shape = (300, 1, 256)
    tmp_13_shape = (300, 1, 256)
    
    # Create output tensors
    out_reshape = torch.empty(tmp_9_shape, dtype=in_4.dtype, device=in_4.device)
    out_unsqueeze = torch.empty(tmp_13_shape, dtype=tmp_6.dtype, device=tmp_6.device)
    
    # Launch kernel
    grid = (lambda meta: (
        (300 + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
    ))
    
    reshape_unsqueeze_kernel[grid](
        in_4,
        tmp_6,
        out_reshape,
        out_unsqueeze,
        BLOCK_SIZE_M=64,  # Process 64 rows per block
    )
    
    return out_reshape, out_unsqueeze

def replacement_func():
    return optimized_reshape_unsqueeze