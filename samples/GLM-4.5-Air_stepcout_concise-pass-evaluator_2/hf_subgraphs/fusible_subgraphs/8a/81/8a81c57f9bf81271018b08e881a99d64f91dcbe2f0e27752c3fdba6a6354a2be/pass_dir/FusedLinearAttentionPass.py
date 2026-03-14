import torch
import triton
import triton.language as tl

def pattern(hidden_states, weight, bias, attention_mask, key_layer, query_layer):
    """
    Pattern that matches the fused linear transformation + attention preparation.
    All input tensors are torch.Tensor objects.

    Pattern:
    1. Linear transformation: torch.nn.functional.linear(hidden_states, weight, bias)  
    2. View and transpose for multi-head attention format: [batch, seq_len, hidden_dim] -> [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    3. Scaled dot product attention with existing Q and K
    4. Transpose and reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
    """
    # Linear transformation to get values
    values = torch.nn.functional.linear(hidden_states, weight, bias)
    
    # Determine tensor dimensions to support variable input shapes
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1] 
    hidden_dim = hidden_states.shape[2]
    
    # Infer number of heads and head dimension from query_layer or key_layer shape
    # query_layer shape: [batch_size, num_heads, seq_len, head_dim]
    num_heads = query_layer.shape[1]
    
    # Calculate head dimension 
    head_dim = hidden_dim // num_heads
    
    # Reshape values to multi-head format: [batch, seq_len, hidden_dim] -> [batch, num_heads, seq_len, head_dim]
    values_multihead = values.view(batch_size, seq_len, num_heads, head_dim)
    
    # Transpose for attention: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim] 
    values_for_attention = values_multihead.transpose(1, 2)
    
    # Scaled dot product attention
    attended_values = torch.nn.functional.scaled_dot_product_attention(
        query=query_layer, 
        key=key_layer, 
        value=values_for_attention, 
        attn_mask=attention_mask, 
        dropout_p=0.0, 
        is_causal=False
    )
    
    # Transpose back: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    attended_multihead = attended_values.transpose(1, 2)
    
    # Reshape back to original hidden dimension: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
    output = attended_multihead.reshape(batch_size, seq_len, hidden_dim)
    
    return output

def replacement_args(hidden_states, weight, bias, attention_mask, key_layer, query_layer):
    """Extract arguments for the fused kernel"""
    return (hidden_states, weight, bias, attention_mask, key_layer, query_layer)

@triton.jit
def fused_linear_attention_kernel(
    hidden_states_ptr, weight_ptr, bias_ptr,
    key_layer_ptr, query_layer_ptr, attention_mask_ptr,
    output_ptr,
    batch_size: tl.constexpr, seq_len: tl.constexpr, hidden_dim: tl.constexpr,
    num_heads: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Fused kernel that combines linear projection, multi-head attention preparation,
    and attention computation for improved performance.
    """
    # Determine grid coordinates for this thread
    m = tl.program_id(0)
    n = tl.program_id(1) 
    k_block_id = tl.program_id(2)
    
    # Calculate memory bounds
    K = head_dim
    bounds_m = batch_size * seq_len
    bounds_n = num_heads * head_dim  # Output dimension per head
    bounds_k = head_dim
    
    # Linear projection: hidden_states -> values (with bias)
    # Load weight matrix for this K block
    weight_offset = k_block_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    weight = tl.load(weight_ptr + weight_offset, mask=weight_offset < K, other=0.0)
    
    # Process multiple elements in parallel
    output_offset = m * BLOCK_SIZE_M * bounds_n + n * BLOCK_SIZE_N
    hidden_states_offset = m * BLOCK_SIZE_M * hidden_dim + k_block_id * BLOCK_SIZE_K
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load hidden states values
        hidden_states_vals = tl.load(
            hidden_states_ptr + hidden_states_offset + k_offset,
            mask=k_offset < K, 
            other=0.0
        )
        
        # Load bias values (only need to load once for this K block)
        if k == 0:
            bias_vals = tl.load(
                bias_ptr + k_offset,
                mask=k_offset < K,
                other=0.0
            )
        
        # Matrix multiplication accumulator update
        # hidden_states_vals: [BLOCK_SIZE_K]
        # weight: [BLOCK_SIZE_K] 
        # Broadcasting to compute outer product for each element in BLOCK_SIZE_M
        hidden_states_expanded = hidden_states_vals[None, :]  # [1, BLOCK_SIZE_K]
        weight_transposed = weight[:, None]  # [BLOCK_SIZE_K, 1]
        
        # Compute partial result
        partial_result = hidden_states_expanded * weight_transposed  # [BLOCK_SIZE_K, BLOCK_SIZE_K]
        
        # Add to accumulator
        acc += partial_result
        
        # Advance to next K block
        hidden_states_offset += BLOCK_SIZE_K
    
    # Add bias
    if k_block_id == 0:
        bias_expanded = bias_vals[None, :]
        acc += bias_expanded
    
    # Apply attention mechanism
    if m < batch_size and n < num_heads:
        # Get head and sequence indices
        head_idx = n
        seq_i = m % seq_len
        
        # Extract attention patterns for this head and sequence position
        key_offset = m * seq_len * head_dim + k_block_id * BLOCK_SIZE_K + (head_idx * seq_len + seq_i) * K
        query_offset = m * seq_len * head_dim + k_block_id * BLOCK_SIZE_K + (head_idx * seq_len + seq_i) * K
        
        # Load query and key values
        key_vals = tl.load(key_layer_ptr + key_offset, mask=k_offset < head_dim, other=0.0)
        query_vals = tl.load(query_layer_ptr + query_offset, mask=k_offset < head_dim, other=0.0)
        
        # Compute attention scores (dot product)
        attention_scores = tl.sum(key_vals * query_vals)
        
        # Load attention mask if applicable
        if seq_i < attention_mask_ptr.shape[2]:
            mask_val = tl.load(attention_mask_ptr + seq_i * attention_mask_ptr.shape[3], mask=mask_val < attention_mask_ptr.shape[3], other=0.0)
            attention_scores_masked = attention_scores * mask_val
        else:
            attention_scores_masked = attention_scores
        
        # Combine with accumulated values
        final_output = acc + attention_scores_masked
        
        # Store result
        tl.store(output_ptr + output_offset, final_output, mask=True)

@torch.fx.wrap  
def fused_linear_attention(hidden_states, weight, bias, attention_mask, key_layer, query_layer):
    """
    Optimized fused kernel wrapper for linear transformation + attention.
    
    This function combines the linear projection, multi-head attention preparation,
    and attention computation into a single efficient operation.
    """
    # Determine tensor dimensions
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1] 
    hidden_dim = hidden_states.shape[2]
    num_heads = query_layer.shape[1]
    head_dim = hidden_dim // num_heads
    
    # Output shape matches input hidden_states shape
    output_shape = (batch_size, seq_len, hidden_dim)
    output = torch.empty(output_shape, dtype=torch.float32, device=hidden_states.device)
    
    # Set up Triton kernel launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    m_grid = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_grid = (num_heads * head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    k_grid = (head_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch the fused kernel
    grid = (m_grid, n_grid, k_grid)
    
    fused_linear_attention_kernel[grid](
        hidden_states_ptr=hidden_states,
        weight_ptr=weight,
        bias_ptr=bias,
        key_layer_ptr=key_layer,
        query_layer_ptr=query_layer,
        attention_mask_ptr=attention_mask,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    """Return the fused attention function for pattern replacement"""
    return fused_linear_attention