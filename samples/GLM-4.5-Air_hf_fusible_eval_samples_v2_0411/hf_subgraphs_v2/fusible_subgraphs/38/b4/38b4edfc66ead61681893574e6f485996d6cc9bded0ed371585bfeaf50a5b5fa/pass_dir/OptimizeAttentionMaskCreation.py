import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0):
    # Create causal mask foundation  
    tmp_1 = torch.full((in_0.shape[1], in_0.shape[1]), -3.4028234663852886e+38, device = in_0.device)
    tmp_2 = torch.arange(in_0.shape[1], device = in_0.device)
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(in_0.shape[1], 1)
    tmp_5 = tmp_2 < tmp_4
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    
    # Process causal mask
    tmp_7 = tmp_6.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, in_0.shape[1], in_0.shape[1])
    
    # Process input attention mask
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, in_0.shape[1], in_0.shape[1])
    tmp_12 = tmp_11.to(torch.float32)
    tmp_13 = torch.tensor(1.0, dtype = torch.float32)
    tmp_14 = tmp_13 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    
    # Combine masks
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return (tmp_19,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def attention_mask_kernel(
    causal_mask_ptr,
    input_mask_ptr,
    output_ptr,
    seq_len,
    mask_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a column of the matrix
    col_idx = tl.program_id(0)
    row_start = tl.program_id(1) * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, seq_len)
    
    # Load current column of input mask
    input_mask_col = tl.load(input_mask_ptr + col_idx * seq_len)
    
    # Process this column
    for row_idx in range(row_start, row_end):
        # Calculate causal mask value: -inf for j <= i, 0 for j > i
        causal_val = mask_value if col_idx <= row_idx else 0.0
        
        # Load input attention mask value
        input_val = input_mask_col[row_idx]
        
        # Convert input mask to attention pattern
        # 1.0 means attend, 0 means don't attend
        # So we want -inf where input_val == 0
        input_effective = 0.0 if input_val == 0 else 1.0
        
        # Combine masks: apply -inf where either mask says to mask
        final_val = causal_val if causal_val == mask_value else (mask_value if input_effective == 0 else 0.0)
        
        # Store result
        output_ptr[row_idx * seq_len + col_idx] = final_val

@torch.fx.wrap
def optimized_attention_mask(in_0):
    seq_len = in_0.shape[1]
    mask_value = -3.4028234663852886e+38
    
    # Create output tensor in float32 (same as original computation)
    output = torch.empty((1, 1, seq_len, seq_len), dtype=torch.float32, device=in_0.device)
    
    # Grid configuration: one program per column, programs distribute across rows
    BLOCK_SIZE = 128  # Adjust based on performance
    num_row_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (seq_len, num_row_blocks)
    
    # Launch kernel for each head (only one head in this case)
    attention_mask_kernel[grid](
        causal_mask_ptr=torch.empty((seq_len, seq_len), dtype=torch.float32, device=in_0.device).data_ptr(),
        input_mask_ptr=in_0.data_ptr(),
        output_ptr=output.data_ptr(),
        seq_len=seq_len,
        mask_value=mask_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_attention_mask