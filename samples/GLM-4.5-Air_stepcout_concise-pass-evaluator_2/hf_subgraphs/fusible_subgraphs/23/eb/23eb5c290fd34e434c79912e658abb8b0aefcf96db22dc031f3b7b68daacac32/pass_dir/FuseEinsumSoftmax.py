import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    # Match the exact computation structure from model.py
    tmp_0 = in_0
    tmp_1 = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    # Return the observable intermediates that appear in the model return
    return tmp_3, tmp_4

# Argument extraction function  
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized kernel using Triton
@triton.jit
def fused_einsum_softmax_kernel(
    energy_ptr,
    query_ptr, 
    key_ptr,
    out_full_ptr,
    out_slice_ptr,
    batch_size, seq_len, num_heads, embed_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one head in the batch
    b_idx, h_idx = tl.program_id(0), tl.program_id(1)
    
    # Compute pointers for current batch and head
    energy_base = energy_ptr + b_idx * seq_len * seq_len * num_heads
    query_base = query_ptr + b_idx * seq_len * seq_len * num_heads + h_idx * seq_len * seq_len
    key_base = key_ptr + b_idx * seq_len * seq_len * num_heads + h_idx * seq_len * seq_len
    
    # Create output pointers
    out_full_base = out_full_ptr + b_idx * seq_len * seq_len * (embed_dim + seq_len)
    out_slice_base = out_slice_ptr + b_idx * seq_len * seq_len * embed_dim
    
    # Iterate over query positions (M dimension)
    for m_i in range(0, seq_len, BLOCK_M):
        m_off = m_i * seq_len
        
        # Load energy values for this batch/head
        energy_vals = tl.load(energy_base + m_off + tl.arange(0, seq_len)).to(tl.float32)
        
        # Compute attention scores for this query position
        scores = tl.empty(seq_len, dtype=tl.float32)
        
        for n_i in range(0, seq_len, BLOCK_N):
            # Load key and query blocks
            key_block = tl.load(key_base + m_off + n_i + tl.arange(0, min(BLOCK_N, seq_len - n_i))).to(tl.float32)
            query_block = tl.load(query_base + m_i * seq_len + n_i + tl.arange(0, min(BLOCK_N, seq_len - n_i))).to(tl.float32)
            
            # Compute dot product between query and key
            score_block = tl.sum(query_block * key_block, axis=0)
            
            # Store scores
            mask = n_i + tl.arange(0, min(BLOCK_N, seq_len - n_i)) < seq_len
            tl.store(scores + n_i, score_block, mask=mask)
        
        # Add energy and apply softmax
        combined = energy_vals + scores
        
        # Simple softmax approximation for performance (max + exp normalization)
        max_val = tl.max(combined)
        shifted = combined - max_val
        exp_vals = tl.exp(shifted)
        sum_exp = tl.sum(exp_vals)
        softmax_vals = exp_vals / sum_exp
        
        # Store softmax result
        store_mask = tl.arange(0, seq_len) < seq_len
        tl.store(out_full_base + m_i * (embed_dim + seq_len) + h_idx * seq_len + tl.arange(0, seq_len), 
                softmax_vals, mask=store_mask)
        
        # Store sliced result (first embed_dim elements)
        store_range = min(seq_len, embed_dim)
        slice_mask = tl.arange(0, store_range) < store_range  
        tl.store(out_slice_base + m_i * embed_dim + h_idx * seq_len + tl.arange(0, store_range),
                softmax_vals + tl.arange(0, store_range - seq_len), mask=slice_mask)

# Kernel wrapper
@torch.fx.wrap
def fused_einsum_softmax_operation(energy, query, key, embed_dim=64):
    # Get input shapes
    batch_size, seq_len, num_heads, _ = query.shape
    
    # Check if we have the right shapes
    if energy.shape != (batch_size, seq_len, seq_len, num_heads):
        raise ValueError(f"Energy shape mismatch: expected {(batch_size, seq_len, seq_len, num_heads)}, got {energy.shape}")
    if key.shape != (batch_size, seq_len, seq_len, num_heads):
        raise ValueError(f"Key shape mismatch: expected {(batch_size, seq_len, seq_len, num_heads)}, got {key.shape}")
    
    # Create output tensors
    out_full_shape = (batch_size, seq_len, seq_len, num_heads)
    out_slice_shape = (batch_size, seq_len, seq_len, num_heads)
    
    out_full = torch.empty(out_full_shape, dtype=query.dtype, device=query.device)
    out_slice = torch.empty(out_slice_shape, dtype=query.dtype, device=query.device)
    
    # Define block sizes for optimal GPU utilization
    BLOCK_M = 32
    BLOCK_N = 32  
    BLOCK_K = 32
    
    # Calculate grid size: (batch_size, num_heads) per block
    grid_x = (batch_size + 1) // 1  # Each program handles one entire batch
    grid_y = (num_heads + 1) // 1   # Each program handles one entire head
    
    # Launch the kernel
    fused_einsum_softmax_kernel[grid_x, grid_y](
        energy,
        query,
        key,
        out_full,
        out_slice,
        batch_size, seq_len, num_heads, embed_dim,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return out_full, out_slice[..., :embed_dim]

# Replacement function returns the fused operation
def replacement_func():
    return fused_einsum_softmax_operation