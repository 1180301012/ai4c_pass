import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_attention_energy_concat_kernel(
    energy_ptr, key_ptr, query_ptr, 
    out_fully_ptr, out_sliced_ptr,
    batch_size, feat_dim, height, width, 
    energy_width, attn_width, concat_width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Fused kernel for:
    1. Attention computation: query @ key.transpose(-1, -2)
    2. Concatenation with energy tensor  
    3. Softmax along concatenated dimension
    4. Slicing the first energy_width elements
    """
    pid_batch = tl.program_id(0)
    pid_height = tl.program_id(1)
    pid_width_col = tl.program_id(2)
    
    # Program handles one position in output [B, H, W, attn_width]
    out_offset = (pid_batch * height + pid_height) * width * attn_width + pid_width_col * attn_width
    
    max_val = -float('inf')
    exp_sum = 0.0
    
    # Compute attention weights for all attn_width positions
    attn_vals = tl.zeros(attn_width, dtype=tl.float32)
    
    for attn_j in range(attn_width):
        for k in range(0, feat_dim, BLOCK_SIZE_K):
            block_k = min(BLOCK_SIZE_K, feat_dim - k)
            query_ptr_local = query_ptr + (pid_batch * feat_dim * height * width + 
                                         pid_height * feat_dim * width + 
                                         k * width + pid_width_col)
            key_ptr_local = key_ptr + (pid_batch * feat_dim * height * width + 
                                      pid_height * feat_dim * width + 
                                      k * width + attn_j)
            
            # Load query and key elements
            query_vals = tl.load(query_ptr_local, mask=(k + tl.arange(0, block_k)) < feat_dim)
            key_vals = tl.load(key_ptr_local, mask=(k + tl.arange(0, block_k)) < feat_dim)
            
            # Compute dot product
            dot_val = 0.0
            for ki in range(block_k):
                dot_val += query_vals[ki] * key_vals[ki]
            
            if k == 0:
                attn_vals[attn_j] = dot_val
            else:
                attn_vals[attn_j] += dot_val
    
    # Get energy value for this position
    energy_ptr_local = energy_ptr + (pid_batch * height * energy_width + pid_height * energy_width + pid_width_col)
    energy_val = tl.load(energy_ptr_local)
    
    # Concatenate energy + attention for this position
    concat_tensor = tl.zeros(concat_width, dtype=tl.float32)
    concat_tensor[0] = energy_val
    for j in range(attn_width):
        concat_tensor[j + 1] = attn_vals[j]
    
    # Apply softmax across concatenated values
    # Find max for numerical stability
    for j in range(concat_width):
        if concat_tensor[j] > max_val:
            max_val = concat_tensor[j]
    
    # Compute exponentials and sum
    for j in range(concat_width):
        exp_val = tl.exp(concat_tensor[j] - max_val)
        exp_sum += exp_val
        concat_tensor[j] = exp_val
    
    # Normalize to get softmax
    if exp_sum > 0:
        concat_tensor = concat_tensor / exp_sum
    
    # Store results: full softmax output and sliced version
    for j in range(concat_width):
        out_idx = out_offset + j
        tl.store(out_fully_ptr + out_idx, concat_tensor[j])
    
    # Store sliced result (only first energy_width elements)
    for j in range(energy_width):
        out_idx = (pid_batch * height + pid_height) * width * energy_width + pid_width_col * energy_width + j
        tl.store(out_sliced_ptr + out_idx, concat_tensor[j])

@torch.fx.wrap
def fused_attention_energy_concat(energy, key, query, return_tuple=True):
    """Fused implementation of einsum + concat + softmax + slicing"""
    B, C, H, W = energy.shape
    
    # Input shapes: 
    # energy: [B, H, W, energy_width] 
    # key, query: [B, C, H, W]
    # Output shapes:
    # full result: [B, H, W, energy_width + W] 
    # sliced result: [B, H, W, energy_width]
    
    energy_width = energy.shape[-1]
    attn_width = W  # attention output width
    concat_width = energy_width + attn_width
    
    # Grid setup: one thread per [B, H, W] position
    block_size_m = 64
    block_size_n = 64  
    block_size_k = 32
    
    grid = (B, H, W)
    
    # Allocate outputs
    out_fully = torch.empty((B, H, W, concat_width), dtype=energy.dtype, device=energy.device)
    out_sliced = torch.empty((B, H, W, energy_width), dtype=energy.dtype, device=energy.device)
    
    # Launch kernel
    fused_attention_energy_concat_kernel[grid](
        energy, key, query,
        out_fully, out_sliced,
        B, C, H, W, 
        energy_width, attn_width, concat_width,
        block_size_m, block_size_n, block_size_k
    )
    
    if return_tuple:
        return out_fully, out_sliced
    else:
        return out_fully

def pattern(in_0, in_1, in_2):
    """Match the computation pattern:
    einsum('bchw,bchj->bhwj', in_2, in_1)
    concat([in_0, einsum], dim=-1)
    softmax(concat_input, dim=-1)
    slice(concat_output, first 64 elements)
    """
    # Note: Model uses cleanup statements like 'in_2 = in_1 = None'
    # but pattern matching should NOT include these cleanup statements
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64))]
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return fused_attention_energy_concat