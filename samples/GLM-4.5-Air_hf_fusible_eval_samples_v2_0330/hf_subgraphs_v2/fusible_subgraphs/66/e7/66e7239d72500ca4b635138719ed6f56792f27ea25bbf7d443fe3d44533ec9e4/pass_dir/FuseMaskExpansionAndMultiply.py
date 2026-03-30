import torch
import triton
import triton.language as tl

def pattern(attention_mask, normalized_tensor):
    """Pattern: fuse unsqueeze -> expand_as -> float -> multiply operations"""
    tmp_5 = attention_mask.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(normalized_tensor)
    tmp_7 = tmp_6.float()
    tmp_8 = normalized_tensor * tmp_7
    return tmp_7, tmp_8, normalized_tensor

def replacement_args(attention_mask, normalized_tensor):
    """Extract arguments for the replacement function"""
    return attention_mask, normalized_tensor

@triton.jit
def fused_mask_multiply_kernel(
    attention_mask_ptr,
    normalized_tensor_ptr,
    out_mask_expanded_ptr,
    out_multiplied_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for attention mask expansion and multiplication"""
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len * hidden_size)
    
    # Load normalized tensor (batch, seq, hidden)
    normalized_val = tl.load(normalized_tensor_ptr + offsets, mask=mask)
    
    # Load attention mask and broadcast manually
    batch_idx = offsets // (seq_len * hidden_size)
    seq_idx = (offsets // hidden_size) % seq_len
    
    # Load attention mask value for this position
    attention_val = tl.load(attention_mask_ptr + batch_idx * seq_len + seq_idx, mask=batch_idx < batch_size, other=0.0)
    
    # Broadcast mask to full feature dimension and convert to float
    mask_expanded = tl.broadcast_to(tl.cast(attention_val, tl.float32), (hidden_size,))
    
    # Perform multiplication
    result = normalized_val * mask_expanded
    
    # Store results
    tl.store(out_mask_expanded_ptr + offsets, mask_expanded, mask=mask)
    tl.store(out_multiplied_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mask_multiply(attention_mask, normalized_tensor):
    """Fused function combining mask expansion and multiplication"""
    batch_size, seq_len, hidden_size = normalized_tensor.shape
    
    # Output tensors
    out_mask_expanded = torch.empty((batch_size, seq_len, hidden_size), dtype=torch.float32, device=normalized_tensor.device)
    out_multiplied = torch.empty_like(normalized_tensor)
    
    # Calculate grid size
    total_elements = batch_size * seq_len * hidden_size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_mask_multiply_kernel[(num_programs,)](
        attention_mask,
        normalized_tensor,
        out_mask_expanded,
        out_multiplied,
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_SIZE,
    )
    
    return out_mask_expanded, out_multiplied, normalized_tensor

def replacement_func():
    """Return the fused function"""
    return fused_mask_multiply