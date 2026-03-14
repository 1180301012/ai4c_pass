import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern: add + softmax + dropout + to(float32)
    
    This matches the attention score computation pattern:
    - Add attention_scores + extended_attention_mask (with broadcasting)
    - Apply softmax on the last dimension
    - Apply dropout
    - Convert to float32
    """
    tmp_0 = in_0 + in_1  # Addition with broadcasting
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    tmp_3 = tmp_2.to(torch.float32)
    return tmp_3


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1)


@triton.jit
def fused_add_softmax_dropout_cast_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    # Tensor dimensions
    batch_size, num_heads, seq_len, mask_seq_len,
    # Dropout probability (scaled for kernel)
    dropout_p_scaled: tl.constexpr,
    # Kernel configuration
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: add + softmax + dropout + typecast.
    
    This kernel performs:
    1. Add in_0 + in_1 (with broadcasting for mask)
    2. Softmax on last dimension (seq_len)
    3. Dropout (if p > 0)
    4. Cast to float32
    """
    # Get position
    row_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate base offsets
    # in_0 shape: [batch, heads, seq, seq] -> stride: [heads*seq*seq, seq*seq, seq, 1]
    # in_1 shape: [batch, 1, 1, mask_seq] -> broadcasts to [batch, heads, seq, mask_seq]
    
    batch_idx = row_idx // num_heads
    actual_head_idx = row_idx % num_heads
    
    # For softmax, we need to load the full row and compute exp(x - max(x))
    # Then sum exp values and normalize
    
    # Offsets for in_0: [batch, head, seq, :]
    in_0_base = (batch_idx * num_heads * seq_len * seq_len + 
                 actual_head_idx * seq_len * seq_len + 
                 seq_idx * seq_len)
    
    # Offsets for in_1: [batch, 0, 0, :] - broadcasts
    in_1_base = (batch_idx * mask_seq_len + 
                 seq_idx * mask_seq_len)  # Assuming mask is [batch, 1, 1, seq]
    
    # Determine actual mask length (could be smaller than seq_len due to broadcasting)
    mask_len = mask_seq_len
    
    # First pass: compute max for numerical stability
    row_max = float('-inf')
    for j in range(0, seq_len, BLOCK_SIZE):
        offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load from in_0
        in_0_offsets = in_0_base + offsets
        x = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=float('-inf'))
        
        # Load from in_1 (broadcasting - only first mask_len elements)
        mask_offsets = offsets % mask_len
        in_mask = offsets < mask_len
        y = tl.load(in_1_ptr + in_1_base + mask_offsets, mask=in_mask, other=0.0)
        
        # Add
        val = x + y
        
        # Update max
        row_max = tl.max(row_max, val)
    
    # Second pass: compute exp(x - max) and sum
    row_sum = 0.0
    for j in range(0, seq_len, BLOCK_SIZE):
        offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load from in_0
        in_0_offsets = in_0_base + offsets
        x = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=float('-inf'))
        
        # Load from in_1 (broadcasting)
        mask_offsets = offsets % mask_len
        in_mask = offsets < mask_len
        y = tl.load(in_1_ptr + in_1_base + mask_offsets, mask=in_mask, other=0.0)
        
        # Add
        val = x + y
        
        # Exp with numerical stability
        exp_val = tl.exp(val - row_max)
        row_sum += exp_val
    
    # Third pass: normalize and apply dropout
    # Generate random values for dropout (use triton random)
    # Note: In inference mode, dropout is effectively disabled
    # For correctness, we apply dropout but with training=False behavior
    # Since training=False in the pattern, we skip dropout
    dropout_p_scaled_val = dropout_p_scaled
    
    for j in range(0, seq_len, BLOCK_SIZE):
        offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load from in_0
        in_0_offsets = in_0_base + offsets
        x = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=float('-inf'))
        
        # Load from in_1 (broadcasting)
        mask_offsets = offsets % mask_len
        in_mask = offsets < mask_len
        y = tl.load(in_1_ptr + in_1_base + mask_offsets, mask=in_mask, other=0.0)
        
        # Add
        val = x + y
        
        # Compute softmax
        exp_val = tl.exp(val - row_max)
        softmax_val = exp_val / row_sum
        
        # Dropout: since training=False, we don't apply dropout
        # Just keep the softmax value
        result = softmax_val
        
        # Cast to float32 (x is already float32, but ensure)
        result = result.to(tl.float32)
        
        # Store
        out_offsets = in_0_base + offsets
        tl.store(out_ptr + out_offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_softmax_dropout_cast_wrapper(in_0, in_1):
    """Wrapper function that launches the fused kernel.
    
    Handles the case where dropout p=0 or training=False (which is the case in all graphs).
    """
    # Get input shape
    batch_size, num_heads, seq_len, _ = in_0.shape
    _, _, _, mask_seq_len = in_1.shape
    
    # Output
    out = torch.empty_like(in_0)
    
    # Determine block size based on seq_len
    BLOCK_SIZE = 128 if seq_len <= 128 else 64
    
    # Grid: (batch * heads, seq_len)
    grid = (batch_size * num_heads, num_heads, seq_len)
    
    # dropout_p_scaled = 0 because training=False in original pattern
    # This effectively skips dropout
    dropout_p_scaled = 0
    
    fused_add_softmax_dropout_cast_kernel[grid](
        in_0, in_1, out,
        batch_size, num_heads, seq_len, mask_seq_len,
        dropout_p_scaled,
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_add_softmax_dropout_cast_wrapper