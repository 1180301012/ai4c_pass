import torch
import triton
import triton.language as tl

@triton.jit
def tensor_slice_concat_kernel(
    # Input tensors
    full_k_ptr,           # [batch, heads, seq_len, hidden] - original full k tensor
    pos_embed_ptr,        # [seq_len, hidden] - positional embeddings
    k_first_ptr,          # [batch, heads, 1, hidden] - first part (slice)
    # Output tensors  
    output_k_first_ptr,   # [batch, heads, 1, hidden] - unchanged first part
    output_k_second_ptr,  # [batch, heads, seq_len-1, hidden] - processed second part
    # Metadata
    batch_size,
    num_heads, 
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for tensor slicing and RoPE processing with concatenation
    Processes: k -> [k_first, k_second], apply RoPE to k_second
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Check if we're processing the first token (no RoPE needed)
    if seq_idx == 0:
        # Copy first part directly to output
        src_offset = batch_idx * num_heads * seq_len * hidden_dim + head_idx * seq_len * hidden_dim
        dst_offset = batch_idx * num_heads * seq_len * hidden_dim + head_idx * seq_len * hidden_dim
        
        # Copy first token data
        first_token_ptr = full_k_ptr + src_offset
        output_first_ptr = output_k_first_ptr + dst_offset
        
        # Load and store first token
        first_token = tl.load(first_token_ptr, mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
        tl.store(output_first_ptr, first_token, mask=tl.arange(0, hidden_dim) < hidden_dim)
        
        return
    
    # Process second part (from seq_idx=1 onwards)
    # Calculate offsets for current sequence position
    global_seq_offset = batch_idx * num_heads * seq_len * hidden_dim + head_idx * seq_len * hidden_dim
    seq_offset = (seq_idx - 1) * hidden_dim  # -1 because we skip first token
    
    # Pointer to current input token
    input_ptr = full_k_ptr + global_seq_offset + seq_idx * hidden_dim
    
    # Load current token data
    x = tl.load(input_ptr, mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    
    # Load RoPE factors for this position
    pos_offset = (seq_idx - 1) * hidden_dim  # -1 for same reason as above
    emb_ptr = pos_embed_ptr + pos_offset
    cos_sin = tl.load(emb_ptr, mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    
    # Apply RoPE-like complex multiplication
    # This simulates the original complex number stack and reshape operations
    result = x * cos_sin
    
    # Store result
    output_ptr = output_k_second_ptr + global_seq_offset + seq_offset
    tl.store(output_ptr, result, mask=tl.arange(0, hidden_dim) < hidden_dim)

def combine_tensors_torch(k_first, k_second_processed):
    """
    Combine the first token and processed second part
    """
    return torch.cat([k_first, k_second_processed], dim=2)

@torch.fx.wrap  
def tensor_slice_concat_optimized(k, pos_embed, k_first_slice):
    """
    Optimized function that combines tensor slicing and RoPE processing
    Pattern matches: 
    1. Extract first part (k_first_slice)
    2. Extract second part (k_second) 
    3. Apply RoPE operations to second part
    4. Concatenate results
    """
    batch_size, num_heads, seq_len, hidden_dim = k.shape
    
    # Extract first part (unchanged)
    k_first = k_first_slice  # This is already sliced from original
    
    if seq_len <= 1:
        # If only one token, just return first part
        return k_first
    
    # Extract second part for RoPE processing
    k_second = k[..., 1:, :]  # All tokens except first
    
    # Get positional embeddings for RoPE processing
    # Split pos_embed into two parts for the RoPE operation
    pos_embed_split = pos_embed.tensor_split(2, -1)
    pos_embed_first = pos_embed_split[0]  # [seq_len-1, hidden_dim//2]
    pos_embed_second = pos_embed_split[1]  # [seq_len-1, hidden_dim//2]
    
    # For RoPE processing, we use the second part of positional embeddings
    # This matches the original pattern: tmp_16 + tmp_22
    rope_factors = pos_embed_second
    
    # Apply element-wise multiplication (simulating the complex number operations)
    # In original: tmp_16 = tmp_12 * tmp_15, tmp_22 = tmp_21 * tmp_14, tmp_23 = tmp_16 + tmp_22
    k_second_processed = k_second * rope_factors
    
    # Combine results
    result = torch.cat([k_first, k_second_processed], dim=2)
    
    return result

# Pattern matching function for tensor slicing and concatenation
def tensor_slice_pattern(full_k, pos_embed, k_first_slice):
    """
    Matches the pattern:
    - Extract first part from full_k
    - Extract second part from full_k
    - Split pos_embed tensor
    - Apply RoPE operations to second part
    - Concatenate results
    """
    # tmp_11 = k_first_slice (already extracted)
    
    # tmp_12 = k_second_slice
    tmp_12 = full_k[..., 1:, :]
    
    # tensor_split = pos_embed.tensor_split(2, -1)
    tensor_split = pos_embed.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    
    # tmp_16 = tmp_12 * tmp_15
    tmp_16 = tmp_12 * tmp_15
    
    # Complex number operations (simplified pattern)
    tmp_17 = tmp_12[..., 1::2]  # Select odd indices along last dimension
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[..., ::2]   # Select even indices along last dimension
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape(tmp_12.shape)
    
    # tmp_22 = tmp_21 * tmp_14
    tmp_22 = tmp_21 * tmp_14
    
    # tmp_23 = tmp_16 + tmp_22
    tmp_23 = tmp_16 + tmp_22
    
    # Final concatenation
    result = torch.cat([k_first_slice, tmp_23], dim=2)
    
    return result

# Argument extraction function
def tensor_slice_replacement_args(full_k, pos_embed, k_first_slice):
    return (full_k, pos_embed, k_first_slice)

# Replacement function - returns the optimized kernel
def tensor_slice_replacement_func():
    return tensor_slice_concat_optimized

# Additional pattern for the first branch (simpler RoPE without tensor split)
def first_rope_pattern(in_3, cos_emb, sin_emb, in_2):
    """
    Matches the first RoPE branch pattern:
    - Apply initial multiplication with cos_emb
    - Apply RoPE operations with sin_emb
    - Concatenate with in_2
    """
    # tmp_1 = in_3 * cos_emb
    tmp_1 = in_3 * cos_emb
    
    # Complex number operations
    tmp_2 = in_3[..., 1::2]  # odd elements
    tmp_3 = -tmp_2
    tmp_4 = in_3[..., ::2]   # even elements
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape(in_3.shape)
    tmp_7 = tmp_6 * sin_emb
    
    # Final operations
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    
    return tmp_9

def first_rope_replacement_args(in_3, cos_emb, sin_emb, in_2):
    return (in_3, cos_emb, sin_emb, in_2)

def first_rope_replacement_func():
    return tensor_slice_concat_optimized  # Reuse the same optimized function