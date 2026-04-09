import torch
import triton
import triton.language as tl

def pattern(energy_H_1, key, query):
    """
    Pattern matching for the entire computation chain:
    einsum = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    tmp_2 = torch.cat([energy_H_1, einsum], dim = -1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim = -1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    """
    einsum = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    tmp_2 = torch.cat([energy_H_1, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[..., :64]
    return tmp_3, tmp_4

def replacement_args(energy_H_1, key, query):
    """
    Extract arguments for the replacement function
    """
    return (energy_H_1, key, query)

@triton.jit
def fused_attention_kernel(
    energy_ptr, key_ptr, query_ptr, 
    concat_out_ptr,
    batch_size: tl.constexpr, height: tl.constexpr, width: tl.constexpr, feat_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for attention + concatenation operations
    
    Process:
    1. Compute einsum('bchw,bchj->bhwj', query, key) -> attention_weights
    2. Concat with energy: concat = [energy_H_1, attention_weights]
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    # Only process within bounds
    if b >= batch_size or h >= height or w >= width:
        return
    
    # Energy tensor [batch_size, height, width, feat_dim] - energy_H_1[b,h,w,c]
    energy_base = energy_ptr + (b * height * width * feat_dim + h * width * feat_dim + w * feat_dim)
    
    # Query tensor [batch_size, height, width, feat_dim] - query[b,c,h,w]
    query_base = query_ptr + (b * height * width * feat_dim + h * width * feat_dim + w * feat_dim)
    
    # Key tensor [batch_size, height, width, feat_dim] - key[b,c,h,j] 
    key_base = key_ptr + (b * height * width * feat_dim + h * width * feat_dim)
    
    # Concat output [batch_size, height, width, feat_dim*2] 
    concat_base = concat_out_ptr + (b * height * width * feat_dim * 2 + h * width * feat_dim * 2 + w * feat_dim * 2)
    
    # Process energy and attention computation for each j position
    for j_local in range(BLOCK_SIZE):
        j_global = w * BLOCK_SIZE + j_local
        
        if j_global >= width:
            continue
        
        # Store energy for first half
        energy_idx = energy_base + j_global
        energy_val = tl.load(energy_idx)
        concat_energy_idx = concat_base + j_global
        tl.store(concat_energy_idx, energy_val)
        
        # Compute attention for this j position
        attention_acc = 0.0
        
        # Unroll the feature dimension computation for better performance
        for c_chunk in range(0, feat_dim, 4):
            # Process 4 elements at a time for vectorization
            end_c = min(c_chunk + 4, feat_dim)
            
            # Load query values (query[b, c_chunk:end_c, h, w])
            query_vals = tl.load(query_base + c_chunk)
            for c_offset in range(1, end_c - c_chunk):
                if c_chunk + c_offset < feat_dim:
                    vals = tl.load(query_base + c_chunk + c_offset)
                    query_vals = tl.concatenate([query_vals, vals])
            
            # Load key values (key[b, c_chunk:end_c, h, j_global])
            key_vals = tl.load(key_base + c_chunk * width + j_global)  
            for c_offset in range(1, end_c - c_chunk):
                if c_chunk + c_offset < feat_dim:
                    vals = tl.load(key_base + (c_chunk + c_offset) * width + j_global)
                    key_vals = tl.concatenate([key_vals, vals])
            
            # Compute partial attention
            attention_acc += tl.sum(query_vals * key_vals)
        
        # Store attention result in second half
        concat_attention_idx = concat_base + feat_dim + j_global
        tl.store(concat_attention_idx, attention_acc)

@torch.fx.wrap
def fused_attention_computation(energy_H_1, key, query):
    """
    Fused computation for attention + concat + slicing
    Note: Softmax is kept separate due to API restrictions
    """
    batch_size, height, width, feat_dim = energy_H_1.shape
    
    # Output tensor
    concat_out = torch.empty((batch_size, height, width, feat_dim * 2), 
                           dtype=energy_H_1.dtype, device=energy_H_1.device)
    
    # Launch kernel for einsum + concatenation
    grid = (
        batch_size,
        height,
        triton.cdiv(width, 32)  # Use 32 as block size
    )
    
    fused_attention_kernel[grid](
        energy_H_1, key, query,
        concat_out,
        batch_size, height, width, feat_dim,
        32
    )
    
    # Apply softmax on concatenated result
    softmax_result = torch.nn.functional.softmax(concat_out, dim=-1)
    
    # Slice first 64 elements  
    sliced_result = softmax_result[..., :64]
    
    return softmax_result, sliced_result

def replacement_func():
    """
    Returns the optimized function
    """
    return fused_attention_computation