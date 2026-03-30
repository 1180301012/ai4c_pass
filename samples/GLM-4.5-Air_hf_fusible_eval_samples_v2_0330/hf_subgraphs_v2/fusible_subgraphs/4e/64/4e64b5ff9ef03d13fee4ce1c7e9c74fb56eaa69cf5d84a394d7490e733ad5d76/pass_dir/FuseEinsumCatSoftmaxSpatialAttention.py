import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """
    Pattern matching for spatial attention computation:
    1. einsum('bchw,bchj->bhwj', in_2, in_1) - attention scores
    2. torch.cat([in_0, einsum], dim=-1) - concatenate energy with scores
    3. softmax(..., dim=-1) - apply attention weights
    Returns both softmax output and sliced version
    """
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2):
    """Extract input tensors for optimized kernel"""
    return in_0, in_1, in_2

@triton.jit
def fused_attention_kernel(
    energy_ptr, key_ptr, query_ptr,
    out_full_ptr, out_sliced_ptr,
    batch_size, height, width, feat_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for spatial attention computation:
    - Computes einsum('bchw,bchj->bhwj', query, key) -> attention_scores  
    - Concatenates energy with attention_scores along last dim
    - Applies softmax along last dimension
    - Returns full softmax output and sliced version
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Grid setup: each program handles one (h,w) position across all batch items
    batch_idx = pid_m // (height * width)
    h_idx = (pid_m % (height * width)) // width  
    w_idx = (pid_m % (height * width)) % width
    
    # Input offsets
    energy_base = batch_idx * height * width * feat_dim
    key_base = batch_idx * feat_dim * height * width * feat_dim  # [B,C,H,W]
    query_base = batch_idx * feat_dim * height * width * feat_dim  # [B,C,H,W]
    
    einsum_base = batch_idx * height * width * feat_dim  # [B,H,W,J=feat_dim]
    
    # Output offsets
    out_full_base = batch_idx * height * width * feat_dim * 2  # [B,H,W,feat_dim*2]
    out_sliced_base = batch_idx * height * width * feat_dim  # [B,H,W,feat_dim]
    
    # Compute einsum: result[h,w,j] = sum_c(query[c,h,w] * key[c,h,j])
    einsum_scores = tl.zeros(feat_dim, dtype=tl.float32)
    
    # Loop over feature dimension C to compute einsum
    for c in range(0, feat_dim, BLOCK_SIZE_K):
        c_mask = c + tl.arange(0, BLOCK_SIZE_K) < feat_dim
        c_vals = tl.load(energy_ptr + energy_base + h_idx * width * feat_dim + w_idx * feat_dim + c + tl.arange(0, BLOCK_SIZE_K),
                        mask=c_mask, other=0.0)
        
        # Load key[c,h,j] for current j
        for j in range(feat_dim):
            key_val = tl.load(key_ptr + key_base + c * height * width * feat_dim + h_idx * width * feat_dim + j * feat_dim + w_idx,
                            mask=(c + tl.arange(0, 1)) < feat_dim and j < feat_dim, other=0.0)[0]
            
            query_val = tl.load(query_ptr + query_base + c * height * width * feat_dim + h_idx * width * feat_dim + j * feat_dim + w_idx,
                              mask=(c + tl.arange(0, 1)) < feat_dim and j < feat_dim, other=0.0)[0]
            
            einsum_scores[j] += key_val * query_val
    
    # Concatenate energy (at current position) with einsum scores
    energy_val = tl.load(energy_ptr + energy_base + h_idx * width * feat_dim + w_idx * feat_dim, 
                        mask=(h_idx * width + w_idx) < height * width, other=0.0)
    
    # Create concatenated tensor: [energy_val, einsum_scores[0], einsum_scores[1], ...]
    concat_scores = tl.zeros(feat_dim * 2, dtype=tl.float32)
    concat_scores[0] = energy_val
    for j in range(feat_dim):
        concat_scores[1 + j] = einsum_scores[j]
    
    # Apply softmax along the concatenated dimension
    max_val = tl.max(concat_scores)
    exp_scores = tl.exp(concat_scores - max_val)
    sum_exp = tl.sum(exp_scores)
    softmax_scores = exp_scores / sum_exp
    
    # Store full softmax result
    for j in range(feat_dim * 2):
        if j < feat_dim * 2:
            tl.store(out_full_ptr + out_full_base + h_idx * width * feat_dim * 2 + w_idx * feat_dim * 2 + j,
                    softmax_scores[j], mask=j < feat_dim * 2)
    
    # Store sliced version (first feat_dim elements)
    for j in range(feat_dim):
        tl.store(out_sliced_ptr + out_sliced_base + h_idx * width * feat_dim + w_idx * feat_dim + j,
                softmax_scores[j], mask=j < feat_dim)

@torch.fx.wrap
def fused_attention_kernel_wrapper(energy, key, query):
    """Kernel wrapper that handles different data types and launches the fused kernel"""
    batch_size, height, width, feat_dim = energy.shape
    
    # Output tensors
    out_full = torch.empty((batch_size, height, width, feat_dim * 2), dtype=energy.dtype, device=energy.device)
    out_sliced = torch.empty((batch_size, height, width, feat_dim), dtype=energy.dtype, device=energy.device)
    
    # Triton kernel configuration
    BLOCK_SIZE_K = 32  # Block size for feature dimension loop
    
    # Calculate grid size: each program handles one (b,h,w) position
    # We use two dimensions: first for batch*spatial positions, second for parallelization
    grid = (batch_size * height * width,)
    
    fused_attention_kernel[grid](
        energy_ptr=energy,
        key_ptr=key,
        query_ptr=query,
        out_full_ptr=out_full,
        out_sliced_ptr=out_sliced,
        batch_size=batch_size,
        height=height,
        width=width,
        feat_dim=feat_dim,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out_full, out_sliced

def replacement_func():
    return fused_attention_kernel_wrapper