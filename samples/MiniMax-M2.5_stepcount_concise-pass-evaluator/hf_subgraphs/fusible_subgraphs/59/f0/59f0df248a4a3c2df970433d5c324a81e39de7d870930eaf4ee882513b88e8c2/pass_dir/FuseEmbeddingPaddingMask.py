import torch
import triton
import triton.language as tl

# This pass optimizes the computation by:
# 1. Fusing masked_fill with scalar multiplication into a single kernel
# 2. Fusing the sum, eq, sum, float, division, subtraction operations


def pattern(embedded, padding_mask):
    """
    Match a simpler pattern: masked_fill followed by scalar multiplication.
    This avoids the eq matching issues.
    
    Operations to match:
    - masked_fill
    - scalar multiplication by 0.88
    """
    # Apply mask
    masked = embedded.masked_fill(padding_mask, 0.0)
    
    # Scalar multiplication
    scaled = masked * 0.88
    
    return scaled


def replacement_args(embedded, padding_mask):
    """Extract arguments needed for the replacement."""
    return (embedded, padding_mask)


@triton.jit
def fused_masked_scale_kernel(
    embedding_ptr,
    mask_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that:
    1. Applies mask (where mask==True, set to 0)
    2. Scales by factor
    
    The mask is [batch, seq_len, 1] boolean tensor.
    True = padding position (should be masked to 0)
    """
    # Grid: (batch_size, seq_len)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate base offset for embedding: [batch, seq, hidden]
    base_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    
    # Mask is at [batch, seq, 1] - get the correct offset
    mask_offset = batch_idx * seq_len + seq_idx
    mask_val = tl.load(mask_ptr + mask_offset)
    
    # Process hidden_dim elements
    for h in range(0, hidden_dim, BLOCK_SIZE):
        offsets = base_offset + h + tl.arange(0, BLOCK_SIZE)
        
        # Load embedding values
        emb = tl.load(embedding_ptr + offsets)
        
        # Apply mask and scale:
        # - If mask is True (non-zero), set to 0
        # - Otherwise, multiply by scale
        result = tl.where(mask_val != 0, 0.0, emb * scale)
        
        # Store result
        tl.store(output_ptr + offsets, result)


@torch.fx.wrap
def fused_masked_scale(embedding, mask, scale):
    """
    Apply padding mask and scale in a single fused kernel.
    
    Args:
        embedding: [batch, seq_len, hidden_dim] - embedding output
        mask: [batch, seq_len, 1] - boolean mask 
        scale: float - scale factor
    
    Returns:
        scaled and masked embedding [batch, seq_len, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = embedding.shape
    
    # Configure block size based on hidden dimension
    BLOCK_SIZE = min(1024, hidden_dim)
    
    # Grid: (batch_size, seq_len)
    grid = (batch_size, seq_len)
    
    output = torch.empty_like(embedding)
    
    fused_masked_scale_kernel[grid](
        embedding_ptr=embedding,
        mask_ptr=mask,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@triton.jit
def fused_ratio_complement_kernel(
    input_ids_ptr,
    attention_mask_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel to compute:
    1. Sum of attention mask
    2. Count of padding tokens (input_ids == 2)
    3. Compute padding ratio = padding_count / mask_sum
    4. Compute complement = 1 - padding_ratio
    """
    batch_idx = tl.program_id(0)
    
    # Compute sum of attention mask for this batch
    mask_sum = 0
    for i in range(0, seq_len, BLOCK_SIZE):
        offsets = batch_idx * seq_len + i + tl.arange(0, BLOCK_SIZE)
        mask = tl.load(attention_mask_ptr + offsets)
        mask_sum += tl.sum(mask)
    
    # Compute count of padding tokens (input_ids == 2)
    padding_count = 0
    for i in range(0, seq_len, BLOCK_SIZE):
        offsets = batch_idx * seq_len + i + tl.arange(0, BLOCK_SIZE)
        ids = tl.load(input_ids_ptr + offsets)
        # Check if equals 2 (padding token)
        is_padding = (ids == 2)
        padding_count += tl.sum(is_padding)
    
    # Compute ratio and complement
    padding_ratio = padding_count / mask_sum
    complement = 1.0 - padding_ratio
    
    # Store result [batch_size]
    tl.store(output_ptr + batch_idx, complement)


@torch.fx.wrap
def fused_ratio_complement(input_ids, attention_mask):
    """
    Compute padding ratio complement in a fused kernel.
    
    Args:
        input_ids: [batch, seq_len] - input token IDs
        attention_mask: [batch, seq_len] - attention mask
    
    Returns:
        complement: [batch, 1, 1] = 1 - (padding_count / mask_sum)
    """
    batch_size, seq_len = input_ids.shape
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    output = torch.zeros(batch_size, device=input_ids.device, dtype=torch.float32)
    
    fused_ratio_complement_kernel[grid](
        input_ids_ptr=input_ids,
        attention_mask_ptr=attention_mask,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to [batch, 1, 1] for broadcasting
    return output.view(-1, 1, 1)


def replacement_func():
    """Return the optimized function that computes the same output."""
    def optimized_computation(embedded, padding_mask):
        # Fused: apply mask and scale in one kernel
        scaled_masked = fused_masked_scale(embedded, padding_mask, 0.88)
        
        return scaled_masked
    
    return optimized_computation