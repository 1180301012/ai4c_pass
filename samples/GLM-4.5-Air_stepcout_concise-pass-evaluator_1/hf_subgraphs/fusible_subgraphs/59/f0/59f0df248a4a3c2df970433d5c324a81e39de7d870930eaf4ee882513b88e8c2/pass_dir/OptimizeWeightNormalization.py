import torch
import triton
import triton.language as tl

def pattern(attention_mask, input_ids):
    # Re-compute padding mask (already computed earlier, but we'll optimize this)
    padding_mask = input_ids.__eq__(2)
    
    # Count tokens and padding
    token_counts = attention_mask.sum(-1).float()
    padding_counts = padding_mask.sum(-1).float()
    
    # Compute padding ratio and create weight
    padding_ratio = padding_counts / token_counts
    weight = 1 - padding_ratio
    weight_expanded = weight[slice(None, None, None), None, None]
    
    return weight_expanded

def replacement_args(attention_mask, input_ids):
    return (attention_mask, input_ids)

@triton.jit
def sum_reduction_kernel(
    input_ptr,
    output_ptr,
    input_shape0,
    input_shape1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= input_shape0:
        return
    
    # Load and sum the row
    row_offset = row_idx * input_shape1
    sum_val = 0.0
    
    for i in range(0, input_shape1, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < input_shape1
        val = tl.load(input_ptr + row_offset + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(val)
    
    # Store the sum
    tl.store(output_ptr + row_idx, sum_val)

@torch.fx.wrap
def optimized_weight_normalization(attention_mask, input_ids):
    batch_size, seq_len = attention_mask.shape
    
    # Reuse padding mask (avoid recomputing)
    padding_mask = input_ids.__eq__(2)
    
    # Allocate output tensors
    token_counts = torch.empty(batch_size, dtype=torch.float32, device=attention_mask.device)
    padding_counts = torch.empty(batch_size, dtype=torch.float32, device=attention_mask.device)
    
    # Launch kernels for simultaneous summation
    block_size = 1024
    grid_token = (batch_size,)
    grid_padding = (batch_size,)
    
    sum_reduction_kernel[grid_token](
        attention_mask,
        token_counts,
        batch_size,
        seq_len,
        block_size
    )
    
    sum_reduction_kernel[grid_padding](
        padding_mask,
        padding_counts,
        batch_size,
        seq_len,
        block_size
    )
    
    # Compute final weight
    padding_ratio = padding_counts / token_counts.clamp(min=1.0)  # avoid division by zero
    
    weight = 1 - padding_ratio
    weight_expanded = weight.unsqueeze(-1).unsqueeze(-1)  # Create broadcastable shape
    
    return weight_expanded

def replacement_func():
    return optimized_weight_normalization