import torch
from torch import device
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Pattern matching for core rotary embeddings components"""
    # Create simple pattern focused on the outer product + cos/sin computation
    tmp_0 = in_0
    tmp_1 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_2 = tmp_1.type_as(tmp_0)
    tmp_3 = torch.outer(tmp_2, tmp_0)
    tmp_5 = torch.cat((tmp_3, tmp_3), dim=-1)
    tmp_6 = tmp_5.cos()
    tmp_8 = tmp_5.sin()
    tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    
    return (tmp_5, tmp_6, tmp_8, tmp_7, tmp_9)

def replacement_args(in_0, in_1):
    """Extract arguments for the optimized kernel"""
    # For now, use a fixed sequence length, pattern matching will handle the actual size
    seq_len = 64  # This will be dynamically determined by the pattern matching
    return (in_0, seq_len)

@triton.jit
def simple_rotary_kernel(
    inv_freq_ptr,  # [inv_freq_size]
    seq_len: tl.constexpr,
    cat_result_ptr,  # [inv_freq_size, seq_len*2]
    cos_result_ptr,  # [inv_freq_size, seq_len*2]
    sin_result_ptr,  # [inv_freq_size, seq_len*2]
    cos_cached_ptr,  # [1, 1, seq_len, inv_freq_size]
    sin_cached_ptr,  # [1, 1, seq_len, inv_freq_size]
    inv_freq_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Simple kernel for basic outer product + cos/sin computation"""
    pid = tl.program_id(0)
    
    # Create range tensor
    pos_ids = tl.arange(0, seq_len)
    
    # Load inv_freq
    inv_freq = tl.load(inv_freq_ptr + tl.arange(0, inv_freq_size), 
                       mask=tl.arange(0, inv_freq_size) < inv_freq_size)
    
    # Compute outer product (freqs * pos)
    # Simple outer product without scaling
    outer_product = inv_freq[None, :] * pos_ids[:, None]  # [seq_len, inv_freq_size]
    
    # Concatenate with itself (double the sequence dimension)
    cat_result = tl.cat([outer_product, outer_product], dim=1)  # [seq_len, inv_freq_size*2]
    
    # Compute cos and sin
    cos_result = tl.cos(cat_result)
    sin_result = tl.sin(cat_result)
    
    # Store results
    cat_result_ptr += pid * seq_len * inv_freq_size * 2
    cos_result_ptr += pid * seq_len * inv_freq_size * 2
    sin_result_ptr += pid * seq_len * inv_freq_size * 2
    
    for i in range(0, seq_len):
        for j in range(0, inv_freq_size * 2, 4):
            mask_j = j + tl.arange(0, 4) < inv_freq_size * 2
            cat_vals = tl.load(cat_result + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                               mask=mask_j, other=0.0)
            cos_vals = tl.load(cos_result + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                               mask=mask_j, other=0.0)
            sin_vals = tl.load(sin_result + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                               mask=mask_j, other=0.0)
            tl.store(cat_result_ptr + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                    cat_vals, mask=mask_j)
            tl.store(cos_result_ptr + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                    cos_vals, mask=mask_j)
            tl.store(sin_result_ptr + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                    sin_vals, mask=mask_j)
    
    # Create cached versions with broadcasting
    cos_cached = cos_result[None, None, :, :]  # [1, 1, seq_len, inv_freq_size]
    sin_cached = sin_result[None, None, :, :]  # [1, 1, seq_len, inv_freq_size]
    
    cos_cached_ptr += pid * seq_len * inv_freq_size
    sin_cached_ptr += pid * seq_len * inv_freq_size
    
    for i in range(0, seq_len):
        for j in range(0, inv_freq_size, 4):
            mask_j = j + tl.arange(0, 4) < inv_freq_size
            cos_vals = tl.load(cos_result + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                               mask=mask_j, other=0.0)
            sin_vals = tl.load(sin_result + i * (inv_freq_size * 2) + j + tl.arange(0, 4),
                               mask=mask_j, other=0.0)
            tl.store(cos_cached_ptr + i * inv_freq_size + j + tl.arange(0, 4),
                    cos_vals, mask=mask_j)
            tl.store(sin_cached_ptr + i * inv_freq_size + j + tl.arange(0, 4),
                    sin_vals, mask=mask_j)

@torch.fx.wrap
def optimized_rotary_embeddings(in_0, seq_len):
    """Wrapper function for the optimized rotary embeddings kernel"""
    inv_freq_size = in_0.shape[0]
    
    # Create output tensors
    # Note: Adjust tensor shapes to match expected pattern outputs
    cat_result = torch.empty((inv_freq_size, seq_len * 2), dtype=in_0.dtype, device=in_0.device)
    cos_result = torch.empty((inv_freq_size, seq_len * 2), dtype=in_0.dtype, device=in_0.device)
    sin_result = torch.empty((inv_freq_size, seq_len * 2), dtype=in_0.dtype, device=in_0.device)
    cos_cached = torch.empty((1, 1, seq_len, inv_freq_size), dtype=in_0.dtype, device=in_0.device)
    sin_cached = torch.empty((1, 1, seq_len, inv_freq_size), dtype=in_0.dtype, device=in_0.device)
    
    # Set up Triton kernel launch configuration
    BLOCK_SIZE_M = 32
    grid = (triton.cdiv(inv_freq_size, BLOCK_SIZE_M),)
    
    # Launch the simple kernel
    simple_rotary_kernel[grid](
        in_0,
        seq_len,
        cat_result,
        cos_result,
        sin_result,
        cos_cached,
        sin_cached,
        inv_freq_size,
        BLOCK_SIZE_M,
    )
    
    return (cat_result, cos_result, sin_result, cos_cached, sin_cached)

def replacement_func():
    return optimized_rotary_embeddings