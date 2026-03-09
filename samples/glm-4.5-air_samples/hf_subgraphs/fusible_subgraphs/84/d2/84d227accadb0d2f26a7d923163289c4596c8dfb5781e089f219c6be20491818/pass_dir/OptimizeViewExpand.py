import torch
import triton
import triton.language as tl

def pattern(tmp_0, expand_shape):
    # Pattern: slice + expand operations
    tmp_9 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_10 = tmp_9.expand(expand_shape)
    return tmp_10

def replacement_args(tmp_0, expand_shape):
    return (tmp_0, expand_shape)

@torch.fx.wrap
def optimized_view_expand(attention_mask, expand_shape):
    """
    Optimized version of slicing and expanding attention mask
    This combines the slice and expand operations into a single more efficient operation
    """
    # Handle cases where expand_shape varies but operation is similar
    original_shape = attention_mask.shape
    
    # Common pattern: attention_mask is [batch_size, seq_len] and we want [batch_size, 1, 1, seq_len]
    # or variations like [batch_size, 1, seq_len, seq_len]
    
    if len(original_shape) == 2 and len(expand_shape) == 4:
        batch_size, seq_len = original_shape
        
        # Analyze the target expand shape to determine the optimal operation
        # Patterns we typically see:
        # [batch_size, 1, 1, seq_len] 
        # [batch_size, 1, seq_len, seq_len]
        # [batch_size, 1, seq_len, 1]
        
        if expand_shape[1] == 1 and expand_shape[2] == 1 and expand_shape[3] == seq_len:
            # Pattern: expand to [batch_size, 1, 1, seq_len]
            # Use torch's view operations which are optimized
            result = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            return result.expand_as(torch.empty(expand_shape))
            
        elif expand_shape[1] == 1 and expand_shape[2] == seq_len and expand_shape[3] == seq_len:
            # Pattern: expand to [batch_size, 1, seq_len, seq_len] for attention patterns
            # This is common in transformer attention mechanisms
            expanded = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            # Create the full [batch_size, 1, seq_len, seq_len] using broadcasting tricks
            result = expanded.unsqueeze(-1).expand(batch_size, 1, seq_len, seq_len)
            return result
            
        elif expand_shape[1] == 1 and expand_shape[2] == seq_len and expand_shape[3] == 1:
            # Pattern: expand to [batch_size, 1, seq_len, 1]
            return attention_mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, seq_len, 1]
    
    # For other patterns, fall back to original operation
    tmp_9 = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
    return tmp_9.expand(expand_shape)

@triton.jit
def expand_mask_kernel(
    mask_ptr,
    out_ptr,
    batch_size, 
    seq_len,
    out_batch, 
    out_seq_dim1,
    out_seq_dim2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for expanding attention masks efficiently
    This expands 2D masks to 4D tensors for attention computations
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_batch * out_seq_dim1 * out_seq_dim2
    
    if mask:
        # Calculate indices
        batch_idx = offsets // (out_seq_dim1 * out_seq_dim2) % out_batch
        seq_idx1 = (offsets % (out_seq_dim1 * out_seq_dim2)) // out_seq_dim2
        seq_idx2 = offsets % out_seq_dim2
        
        valid_batch = batch_idx < batch_size
        valid_seq1 = seq_idx1 < seq_len
        valid_seq2 = seq_idx2 < seq_len
        
        # Load original mask value using broadcasting rules
        # Original mask has shape [batch_size, seq_len]
        orig_mask_offset = batch_idx * seq_len + seq_idx1
        
        mask_val = tl.load(
            mask_ptr + orig_mask_offset,
            mask=valid_batch & valid_seq1,
            other=0.0
        )
        
        # Store expanded value (with attention mask logic)
        out_offset = offsets
        expanded_val = mask_val  # For positive attention mask as in our case
        
        tl.store(out_ptr + out_offset, expanded_val, mask=valid_batch & valid_seq1 & valid_seq2)

@torch.fx.wrap  
def triton_expand_attention_mask(attention_mask, expand_shape):
    """
    Triton-optimized version for attention mask expansion
    This is particularly efficient for large sequences and batch sizes
    """
    original_shape = attention_mask.shape
    if len(original_shape) != 2 or len(expand_shape) != 4:
        # Fallback for unsupported shapes
        return optimized_view_expand(attention_mask, expand_shape)
    
    batch_size, seq_len = original_shape
    out_batch, dim2, out_seq_dim1, out_seq_dim2 = expand_shape
    
    # Special case: batch_size might be different, use broadcasting
    expected_batch = max(batch_size, out_batch)
    
    output = torch.empty(expand_shape, dtype=attention_mask.dtype, device=attention_mask.device)
    
    BLOCK_SIZE = 1024
    grid_size = (out_batch * out_seq_dim1 * out_seq_dim2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    expand_mask_kernel[grid_size](
        attention_mask,
        output,
        expected_batch,  # Original batch size
        seq_len,          # Original sequence length  
        out_batch,        # Target batch size
        out_seq_dim1,     # First target sequence dimension
        out_seq_dim2,     # Second target sequence dimension
        BLOCK_SIZE
    )
    
    return output

def optimized_view_expand_wrapper(attention_mask, expand_shape):
    """
    Smart wrapper that chooses the best optimization strategy
    """
    # Use simple view operations for small sizes
    total_elements = attention_mask.numel() * expand_shape[-1]
    if total_elements < 10000:  # For small tensors, use PyTorch's optimized operations
        return optimized_view_expand(attention_mask, expand_shape)
    
    # For larger tensors, use Triton-optimized kernel
    return triton_expand_attention_mask(attention_mask, expand_shape)

def replacement_func():
    return optimized_view_expand_wrapper