import torch
import triton
import triton.language as tl


# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2):
    # Compute cos and sin of in_1
    tmp_1 = in_1.cos()
    tmp_2 = in_1.sin()
    
    # Concatenate in_0 and in_2 on last dimension
    tmp_0 = torch.cat((in_0, in_2), dim=-1)
    
    # Concatenate cos and sin on last dimension
    tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
    
    # Stack the two concatenations along last dimension
    tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
    
    # Transpose last two dimensions
    tmp_5 = tmp_4.transpose(-1, -2)
    
    return tmp_5


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Optimized Triton kernel that fuses all operations
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    stride_in0_0, stride_in0_1,
    stride_in1_0, stride_in1_1,
    stride_in2_0, stride_in2_1,
    stride_out_0, stride_out_1, stride_out_2,
    M: tl.constexpr,  # rows
    N: tl.constexpr,  # columns per input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the output
    row_idx = tl.program_id(0)
    col_group_idx = tl.program_id(1)  # 0 or 1
    
    # The output has shape [M, 2, N*2] = [M, 2, 128]
    # col_group_idx = 0: output = concat(in_0, in_2) -> columns 0-127
    # col_group_idx = 1: output = concat(cos, sin) -> columns 0-127
    
    # Calculate offsets for output columns
    # Each program handles columns 0-127 of its output slice
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < (N * 2)  # 128
    
    if col_group_idx == 0:
        # Output = concat(in_0, in_2)
        # Columns 0-63: in_0[row, col]
        # Columns 64-127: in_2[row, col-64]
        
        # For in_0: columns 0-63
        in0_col = col_offsets
        in0_mask = (in0_col < N) & mask
        in0_offset = row_idx * stride_in0_0 + in0_col * stride_in0_1
        in0_val = tl.load(in_0_ptr + in0_offset, mask=in0_mask, other=0.0)
        
        # For in_2: columns 64-127
        in2_col = col_offsets - N
        in2_mask = (in2_col >= 0) & (in2_col < N) & mask
        in2_offset = row_idx * stride_in2_0 + in2_col * stride_in2_1
        in2_val = tl.load(in_2_ptr + in2_offset, mask=in2_mask, other=0.0)
        
        # Select based on column index: first half from in_0, second half from in_2
        result = tl.where(col_offsets < N, in0_val, in2_val)
    else:
        # Output = concat(cos(in_1), sin(in_1))
        # Columns 0-63: cos(in_1)[row, col]
        # Columns 64-127: sin(in_1)[row, col-64]
        
        # The in_1 values we use for cos/sin: for columns 0-63 in output, 
        # we use in_1 columns 0-63. Same for columns 64-127.
        
        # Use col_offsets % N to get the in_1 column index (0-63)
        # When col_offsets is 0-63: col_offsets % 64 = 0-63 -> use in_1[:, 0:64]
        # When col_offsets is 64-127: col_offsets % 64 = 0-63 -> use in_1[:, 0:64]
        in1_col = col_offsets % N
        in1_mask = (in1_col < N) & mask
        in1_offset = row_idx * stride_in1_0 + in1_col * stride_in1_1
        in1_val = tl.load(in_1_ptr + in1_offset, mask=in1_mask, other=0.0)
        
        # Compute cos and sin of the same in1_val
        cos_val = tl.cos(in1_val)
        sin_val = tl.sin(in1_val)
        
        # Select: first half is cos, second half is sin
        # col_offsets < N means columns 0-63 -> cos
        # col_offsets >= N means columns 64-127 -> sin
        result = tl.where(col_offsets < N, cos_val, sin_val)
    
    # Store result to output
    # Output shape: [M, 2, N*2], strides: [2*N, N*2, 1]
    out_offset = row_idx * stride_out_0 + col_group_idx * stride_out_1 + col_offsets * stride_out_2
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    # Get input shapes
    M = in_0.shape[0]  # 64
    N = in_0.shape[1]  # 64
    
    # Output shape: [M, 2, N*2] = [64, 2, 128]
    out = torch.empty((M, 2, N * 2), dtype=in_0.dtype, device=in_0.device)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid: (M rows, 2 for the stack dimension)
    grid = (M, 2)
    
    fused_kernel[grid](
        in_0, in_1, in_2, out,
        in_0.stride(0), in_0.stride(1),
        in_1.stride(0), in_1.stride(1),
        in_2.stride(0), in_2.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        M, N,
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_kernel_wrapper