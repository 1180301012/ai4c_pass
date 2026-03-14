import torch
import triton
import triton.language as tl

def pattern(attention_mask, input_ids, embedding_output):
    """Pattern: optimal mask computation + normalization operations"""
    # Compute mask once (avoiding redundant computation)
    mask = input_ids.__eq__(2)
    
    # Compute normalized attention weights
    attention_sum = attention_mask.sum(-1)
    mask_sum = mask.sum(-1).float()
    normalization_factor = mask_sum / attention_sum
    complement = 1 - normalization_factor
    
    # Scale output
    scaled_output = embedding_output * 0.88
    
    # Expand complement for broadcasting
    complement_expanded = complement[slice(None, None, None), None, None]
    
    return complement_expanded, scaled_output

def replacement_args(attention_mask, input_ids, embedding_output):
    return (attention_mask, input_ids, embedding_output)

@triton.jit
def compute_mask_kernel(
    input_ids_ptr,
    mask_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute input_ids == 2 mask efficiently"""
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load input IDs
    input_vals = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    
    # Compute mask (input_id == 2)
    mask_vals = (input_vals == 2)
    
    # Store result
    tl.store(mask_ptr + offsets, mask_vals, mask=mask)

@triton.jit
def normalization_kernel(
    attention_mask_ptr,
    mask_ptr,
    complement_ptr,
    mask_sum_ptr,
    attention_sum_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute normalization factors"""
    batch_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load attention and mask data
    attention_vals = tl.load(attention_mask_ptr + batch_id * seq_len + offsets, mask=mask, other=0)
    mask_vals = tl.load(mask_ptr + batch_id * seq_len + offsets, mask=mask, other=0)
    
    # Compute sums for this batch
    local_attention_sum = tl.sum(attention_vals)
    local_mask_sum = tl.sum(mask_vals.to(tl.float32))
    
    # Store partial sums
    attention_sum_offset = batch_id * num_blocks + tl.program_id(1)
    mask_sum_offset = batch_id * num_blocks + tl.program_id(1)
    
    tl.store(attention_sum_ptr + attention_sum_offset, local_attention_sum)
    tl.store(mask_sum_ptr + mask_sum_offset, local_mask_sum)

@triton.jit
def final_norm_kernel(
    attention_sum_ptr,
    mask_sum_ptr,
    complement_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Final normalization computation"""
    batch_id = tl.program_id(0)
    
    # Load partial sums
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    local_attention_sum = 0.0
    local_mask_sum = 0.0
    
    for block_id in range(num_blocks):
        attention_offset = batch_id * num_blocks + block_id
        mask_offset = batch_id * num_blocks + block_id
        
        local_attention_sum += tl.load(attention_sum_ptr + attention_offset)
        local_mask_sum += tl.load(mask_sum_ptr + mask_offset)
    
    # Compute normalization factor
    if local_attention_sum > 0:
        norm_factor = local_mask_sum / local_attention_sum
    else:
        norm_factor = 0.0
    
    # Compute complement (1 - norm_factor)
    complement = 1.0 - norm_factor
    
    # Store result
    tl.store(complement_ptr + batch_id, complement)

@torch.fx.wrap
def optimized_mask_norm(attention_mask, input_ids, embedding_output):
    batch_size, seq_len = attention_mask.shape
    
    # Create output tensors
    scaled_output = embedding_output * 0.88
    complement = torch.zeros(batch_size, dtype=torch.float32, device=attention_mask.device)
    
    # Block size optimization
    if seq_len <= 128:
        BLOCK_SIZE = 128
    elif seq_len <= 256:
        BLOCK_SIZE = 256
    elif seq_len <= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Compute mask
    mask = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=input_ids.device)
    grid = ((batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    compute_mask_kernel[grid](
        input_ids_ptr=input_ids,
        mask_ptr=mask,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape mask
    mask = mask.reshape(batch_size, seq_len)
    
    # Compute normalization factors
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    attention_sum = torch.zeros(batch_size * num_blocks, dtype=torch.float32, device=attention_mask.device)
    mask_sum = torch.zeros(batch_size * num_blocks, dtype=torch.float32, device=attention_mask.device)
    
    grid = (batch_size, num_blocks)
    normalization_kernel[grid](
        attention_mask_ptr=attention_mask,
        mask_ptr=mask,
        complement_ptr=complement,
        mask_sum_ptr=mask_sum,
        attention_sum_ptr=attention_sum,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final normalization
    grid = (batch_size,)
    final_norm_kernel[grid](
        attention_sum_ptr=attention_sum,
        mask_sum_ptr=mask_sum,
        complement_ptr=complement,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Expand complement for broadcasting
    complement_expanded = complement[:, None, None]
    
    return complement_expanded, scaled_output

def replacement_func():
    return optimized_mask_norm