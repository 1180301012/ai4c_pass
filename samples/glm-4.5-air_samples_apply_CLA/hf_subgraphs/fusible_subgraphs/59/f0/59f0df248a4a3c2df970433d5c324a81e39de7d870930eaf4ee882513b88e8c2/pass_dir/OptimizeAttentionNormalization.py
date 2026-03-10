import torch
import triton
import triton.language as tl

def pattern(attention_mask, input_ids):
    # Get attention mask sum (count non-padded tokens)
    attention_sum = attention_mask.sum(-1)
    
    # Count padding tokens (redundant with original pattern)
    padding_counts = input_ids.__eq__(2).sum(-1).float()
    
    # Calculate normalization factor: 1 - (padding_counts / attention_sum)
    # This represents the proportion of non-padded tokens
    padding_ratio = padding_counts / attention_sum
    normalization_factor = 1 - padding_ratio
    
    # Add broadcast dimensions for compatibility
    result = normalization_factor[slice(None, None, None), None, None]
    
    return result

def replacement_args(attention_mask, input_ids):
    return (attention_mask, input_ids)

@triton.jit
def normalization_kernel(
    attention_mask_ptr,
    input_ids_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized attention normalization kernel"""
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Process one row at a time
    row_offset = pid * seq_len
    
    # Load attention mask for this sample
    attention_sum = 0
    for i in range(0, seq_len, BLOCK_SIZE):
        end = min(i + BLOCK_SIZE, seq_len)
        mask_data = tl.load(attention_mask_ptr + row_offset + i, mask=tl.arange(i < seq_len) < (end - i))
        attention_sum += tl.sum(mask_data)
    
    # Count padding tokens (token == 2)
    padding_count = 0
    for i in range(0, seq_len, BLOCK_SIZE):
        end = min(i + BLOCK_SIZE, seq_len)
        input_ids_data = tl.load(input_ids_ptr + row_offset + i, mask=tl.arange(i < seq_len) < (end - i))
        is_padding = (input_ids_data == 2).to(tl.int32)
        padding_count += tl.sum(is_padding)
    
    # Compute normalization factor
    if attention_sum > 0:
        padding_ratio = padding_count / attention_sum
        normalization_factor = 1.0 - padding_ratio
    else:
        normalization_factor = 1.0
    
    # Store result with broadcast dimensions (1, 1, 1)
    tl.store(output_ptr + pid * 3, normalization_factor)

@torch.fx.wrap
def optimized_attention_normalization(attention_mask, input_ids):
    batch_size, seq_len = attention_mask.shape
    
    # Create output tensor with proper broadcasting dimensions
    output = torch.empty(batch_size, 1, 1, dtype=torch.float32, device=attention_mask.device)
    
    # Set block size
    BLOCK_SIZE = 512
    num_programs = batch_size
    
    # Launch kernel
    normalization_kernel[(num_programs,)](
        attention_mask_ptr=attention_mask,
        input_ids_ptr=input_ids,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_attention_normalization