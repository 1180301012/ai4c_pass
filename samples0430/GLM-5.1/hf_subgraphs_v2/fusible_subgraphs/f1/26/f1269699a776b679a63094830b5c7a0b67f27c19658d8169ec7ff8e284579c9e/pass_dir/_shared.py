import torch
import triton
import triton.language as tl


@triton.jit
def fused_embedding_layernorm_kernel(
    input_ids_ptr,
    position_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    batch_seq_len,
    hidden_dim: tl.constexpr,
    eps,
    BLOCK_SIZE_H: tl.constexpr,
):
    """Fused embedding lookup + addition + layer_norm kernel.
    
    Each program handles one token (one row of the output).
    """
    token_idx = tl.program_id(0)
    if token_idx >= batch_seq_len:
        return
    
    # Get input and position ids for this token
    input_id = tl.load(input_ids_ptr + token_idx)
    position_id = tl.load(position_ids_ptr + token_idx)
    
    # First pass: compute mean
    sum_val = 0.0
    for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offsets < hidden_dim
        
        # Load embedding rows
        word_offsets = input_id * hidden_dim + h_offsets
        pos_offsets = position_id * hidden_dim + h_offsets
        
        word_val = tl.load(word_emb_ptr + word_offsets, mask=h_mask, other=0.0)
        pos_val = tl.load(pos_emb_ptr + pos_offsets, mask=h_mask, other=0.0)
        
        # Cast to float32 and add
        word_float = tl.cast(word_val, tl.float32)
        pos_float = tl.cast(pos_val, tl.float32)
        emb_sum = word_float + pos_float
        
        # Accumulate for mean
        sum_val += tl.sum(emb_sum, axis=0)
    
    mean = sum_val / hidden_dim
    
    # Second pass: compute variance using (x - mean)^2 for numerical stability
    var_val = 0.0
    for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offsets < hidden_dim
        
        # Load embedding rows again
        word_offsets = input_id * hidden_dim + h_offsets
        pos_offsets = position_id * hidden_dim + h_offsets
        
        word_val = tl.load(word_emb_ptr + word_offsets, mask=h_mask, other=0.0)
        pos_val = tl.load(pos_emb_ptr + pos_offsets, mask=h_mask, other=0.0)
        
        word_float = tl.cast(word_val, tl.float32)
        pos_float = tl.cast(pos_val, tl.float32)
        emb_sum = word_float + pos_float
        
        # Compute (x - mean)^2 for variance
        diff = emb_sum - mean
        var_val += tl.sum(diff * diff, axis=0)
    
    var = var_val / hidden_dim
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Third pass: normalize and store
    output_base = token_idx * hidden_dim
    
    for h_start in range(0, hidden_dim, BLOCK_SIZE_H):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offsets < hidden_dim
        
        # Load embedding rows again
        word_offsets = input_id * hidden_dim + h_offsets
        pos_offsets = position_id * hidden_dim + h_offsets
        
        word_val = tl.load(word_emb_ptr + word_offsets, mask=h_mask, other=0.0)
        pos_val = tl.load(pos_emb_ptr + pos_offsets, mask=h_mask, other=0.0)
        
        word_float = tl.cast(word_val, tl.float32)
        pos_float = tl.cast(pos_val, tl.float32)
        emb_sum = word_float + pos_float
        
        # Normalize: (x - mean) * rstd
        x_norm = (emb_sum - mean) * rstd
        
        # Apply weight and bias
        ln_w = tl.load(ln_weight_ptr + h_offsets, mask=h_mask, other=0.0)
        ln_b = tl.load(ln_bias_ptr + h_offsets, mask=h_mask, other=0.0)
        ln_w_float = tl.cast(ln_w, tl.float32)
        ln_b_float = tl.cast(ln_b, tl.float32)
        
        result = x_norm * ln_w_float + ln_b_float
        
        # Store result (auto-cast to output dtype)
        tl.store(output_ptr + output_base + h_offsets, result, mask=h_mask)


@triton.jit
def attention_mask_kernel(
    output_ptr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused attention mask computation kernel for MPNet.
    
    Each program handles one row (position i) of the [seq_len, seq_len] mask.
    
    The mask computation:
    - For position (i, j):
      - distance = |i - j|
      - base = 16 if j > i (future position), else 0
      - if distance < 8: penalty = distance
      - else: penalty = min(floor(log(distance/8) / log(16) * 8) + 8, 15)
      - mask[i, j] = base + penalty
    """
    row_idx = tl.program_id(0)
    
    # Compute the entire row at once (seq_len is small, so one iteration is enough for most cases)
    for col_start in range(0, seq_len, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < seq_len
        
        # Relative position: j - i (col - row)
        diff = col_offsets - row_idx
        
        # Negate: i - j = -diff
        neg_diff = -diff
        
        # Base: 16 for future positions (j > i means neg_diff < 0)
        is_future = neg_diff < 0
        base = tl.where(is_future, 16, 0).to(tl.int64)
        
        # Distance: |i - j| = |neg_diff|
        distance = tl.abs(neg_diff)
        
        # Within window (distance < 8)?
        within_window = distance < 8
        
        # For the logarithmic branch, avoid log(0) by using max(distance, 1)
        safe_distance = tl.where(distance >= 1, distance, 1)
        dist_float = tl.cast(safe_distance, tl.float32)
        
        # Compute logarithmic penalty: floor(log(dist/8) / log(16) * 8) + 8
        log_ratio = tl.log(dist_float / 8.0) / 2.772588722239781
        bucket_float = log_ratio * 8.0
        bucket_int = tl.cast(bucket_float, tl.int64)
        penalty_outside = bucket_int + 8
        penalty_capped = tl.minimum(penalty_outside, 15)
        
        # Choose penalty based on window
        penalty = tl.where(within_window, distance, penalty_capped)
        
        # Final mask value
        result = base + penalty
        
        # Store
        output_offsets = row_idx * seq_len + col_offsets
        tl.store(output_ptr + output_offsets, result, mask=col_mask)


# Helper function to compute attention mask using Triton kernel
@torch.fx.wrap
def compute_attention_mask(seq_len, device):
    """Compute the MPNet attention mask using a fused Triton kernel."""
    output_mask = torch.empty((seq_len, seq_len), dtype=torch.int64, device=device)
    BLOCK_SIZE_MASK = 64
    grid_mask = (seq_len,)
    attention_mask_kernel[grid_mask](
        output_ptr=output_mask,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE_MASK,
    )
    return output_mask


# The dispatch wrapper - shared by all pass files
@torch.fx.wrap
def fused_embedding_layernorm_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, route):
    """Dispatch wrapper that routes to appropriate kernel configurations.
    
    This wrapper computes the embedding + layer_norm output (tmp_9).
    The attention mask (tmp_32) is NOT computed here - it remains in the original graph.
    """
    
    # Route-specific parameters
    if route == "bf16_mpnet_768_11":
        seq_len = 11
        hidden_dim = 768
        eps = 1e-05
        BLOCK_SIZE_H = 256
    elif route == "bf16_tiny_64_45":
        seq_len = 45
        hidden_dim = 64
        eps = 1e-12
        BLOCK_SIZE_H = 64
    elif route == "f16_mpnet_768_7":
        seq_len = 7
        hidden_dim = 768
        eps = 1e-05
        BLOCK_SIZE_H = 256
    elif route == "f32_mpnet_768_7":
        seq_len = 7
        hidden_dim = 768
        eps = 1e-05
        BLOCK_SIZE_H = 256
    elif route == "f16_tiny_64_45":
        seq_len = 45
        hidden_dim = 64
        eps = 1e-12
        BLOCK_SIZE_H = 64
    else:
        raise ValueError(f"Unknown route: {route}")
    
    batch_size = in_0.shape[0]
    total_tokens = batch_size * seq_len
    
    # Allocate output tensor for embedding + layer_norm result
    # tmp_9 shape: [batch_size, seq_len, hidden_dim], same dtype as ln_weight (in_2)
    output_emb = torch.empty((batch_size, seq_len, hidden_dim), dtype=in_2.dtype, device=in_2.device)
    
    # Launch embedding + layer_norm kernel
    grid_emb = (total_tokens,)
    fused_embedding_layernorm_kernel[grid_emb](
        input_ids_ptr=in_0,
        position_ids_ptr=in_5,
        word_emb_ptr=in_4,
        pos_emb_ptr=in_3,
        ln_weight_ptr=in_2,
        ln_bias_ptr=in_1,
        output_ptr=output_emb,
        batch_seq_len=total_tokens,
        hidden_dim=hidden_dim,
        eps=eps,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output_emb