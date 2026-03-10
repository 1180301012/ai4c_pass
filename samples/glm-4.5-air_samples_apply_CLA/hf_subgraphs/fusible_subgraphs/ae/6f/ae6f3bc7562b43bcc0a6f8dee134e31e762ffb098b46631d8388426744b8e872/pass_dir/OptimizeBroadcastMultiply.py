import torch
import triton
import triton.language as tl

def pattern(tensor, mask_tensor):
    # Pattern: unsqueeze + multiplication (broadcasting)
    # This matches: tmp_8 = tmp_0.unsqueeze(-1); tmp_9 = tmp_7 * tmp_8
    unsqueezed_mask = mask_tensor.unsqueeze(-1)
    result = tensor * unsqueezed_mask
    return result

def replacement_args(tensor, mask_tensor):
    return (tensor, mask_tensor)

@triton.jit
def broadcast_multiply_kernel(
    tensor_ptr,
    mask_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    # Calculate program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2)
    
    # Compute global offsets
    batch_offset = batch_idx * seq_len
    tensor_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + hidden_idx
    
    # Ensure we're within bounds
    if batch_idx >= batch_size or seq_idx >= seq_len or hidden_idx >= hidden_dim:
        return
    
    # Load values
    tensor_val = tl.load(tensor_ptr + tensor_offset)
    mask_val = tl.load(mask_ptr + batch_offset + seq_idx)
    
    # Apply broadcasting multiplication
    result = tensor_val * mask_val
    
    # Store result
    tl.store(output_ptr + tensor_offset, result)

@torch.fx.wrap
def optimized_broadcast_multiply(tensor, mask_tensor):
    batch_size, seq_len, hidden_dim = tensor.shape
    mask_batch, mask_seq = mask_tensor.shape
    
    # Output tensor
    output = torch.empty_like(tensor)
    
    # Optimize block size based on hidden dimension
    if hidden_dim <= 512:
        BLOCK_SIZE_HIDDEN = 64
    elif hidden_dim <= 1024:
        BLOCK_SIZE_HIDDEN = 128
    else:
        BLOCK_SIZE_HIDDEN = 256
    
    # Calculate grid size: (batch_size, seq_len, hidden_dim // BLOCK_SIZE_HIDDEN)
    grid = (
        batch_size,
        seq_len,
        (hidden_dim + BLOCK_SIZE_HIDDEN - 1) // BLOCK_SIZE_HIDDEN
    )
    
    # Launch kernel
    broadcast_multiply_kernel[grid](
        tensor_ptr=tensor,
        mask_ptr=mask_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE_HIDDEN=BLOCK_SIZE_HIDDEN,
    )
    
    return output

def replacement_func():
    return optimized_broadcast_multiply