import torch
import triton
import triton.language as tl

def pattern(attention_mask, padding_mask, scaling_factor):
    """
    Pattern: Exact operations from model for normalization:
    tmp_7 = tmp_0.sum(-1)
    tmp_9 = tmp_8.sum(-1)  
    tmp_10 = tmp_9.float()
    tmp_11 = tmp_10 / tmp_7
    tmp_13 = 1 - tmp_11
    tmp_14 = tmp_13[slice(None, None, None), None, None]
    
    Returns exactly what the model uses: tmp_14 (expanded ratio)
    """
    # tmp_7 = tmp_0.sum(-1)
    tmp_7 = attention_mask.sum(-1)
    
    # tmp_9 = tmp_8.sum(-1)  
    tmp_9 = padding_mask.sum(-1)
    
    # tmp_10 = tmp_9.float()
    tmp_10 = tmp_9.float()
    
    # tmp_11 = tmp_10 / tmp_7
    tmp_11 = tmp_10 / tmp_7
    
    # tmp_13 = 1 - tmp_11
    tmp_13 = 1 - tmp_11
    
    # tmp_14 = tmp_13[slice(None, None, None), None, None]
    tmp_14 = tmp_13[slice(None, None, None), None, None]
    
    return tmp_14


def replacement_args(attention_mask, padding_mask, scaling_factor):
    return (attention_mask, padding_mask, scaling_factor)


@triton.jit
def normalization_kernel(
    attention_mask_ptr,
    padding_mask_ptr,
    output_ratio_ptr,
    batch_size,
    seq_len,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Each program handles one batch item
    
    # Step 1: Compute attention mask sum for this batch item
    attention_sum = 0
    seq_start = pid * seq_len
    for i in range(seq_len):
        idx = seq_start + i
        if idx < batch_size * seq_len:
            mask_val = tl.load(attention_mask_ptr + idx)
            attention_sum += mask_val
    
    # Step 2: Count padding tokens for this batch item
    padding_count = 0
    for i in range(seq_len):
        idx = seq_start + i
        if idx < batch_size * seq_len:
            is_padding = tl.load(padding_mask_ptr + idx)
            padding_count += is_padding
    
    # Step 3: Calculate ratio
    # Avoid division by zero
    attention_sum = tl.maximum(attention_sum, 1)
    padding_ratio = tl.cast(padding_count, tl.float32) / tl.cast(attention_sum, tl.float32)
    complement_ratio = 1.0 - padding_ratio
    
    # Store ratio result (already expanded shape)
    tl.store(output_ratio_ptr + pid * seq_len, complement_ratio)


@torch.fx.wrap
def fused_normalization_scaling(attention_mask, padding_mask, scaling_factor):
    batch_size, seq_len = attention_mask.shape
    
    # Allocate output
    expanded_ratio = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=attention_mask.device)
    
    # Launch kernel
    BLOCK_SIZE = 256  # Larger block size for this simpler computation
    num_programs = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    normalization_kernel[(num_programs,)](
        attention_mask_ptr=attention_mask,
        padding_mask_ptr=padding_mask,
        output_ratio_ptr=expanded_ratio,
        batch_size=batch_size,
        seq_len=seq_len,
        scaling_factor=float(scaling_factor),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return expanded_ratio


def replacement_func():
    return fused_normalization_scaling