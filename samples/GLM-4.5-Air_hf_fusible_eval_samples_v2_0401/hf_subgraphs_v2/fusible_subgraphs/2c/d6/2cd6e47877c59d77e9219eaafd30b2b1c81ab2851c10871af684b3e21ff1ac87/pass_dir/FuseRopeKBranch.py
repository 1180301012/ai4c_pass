import torch
import triton
import triton.language as tl

def pattern(in_0, in_4):
    # RoPE computation for K branch (second part of the computation)
    tmp_12 = in_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tensor_split = in_0.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[(Ellipsis, slice(1, None, 2))]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[(Ellipsis, slice(None, None, 2))]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, -1, 256, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    return tmp_23

def replacement_args(in_0, in_4):
    return (in_0, in_4)

@triton.jit
def rope_k_kernel(
    k_ptr, pos_embed_0_ptr, pos_embed_1_ptr,
    out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized RoPE kernel for K branch computation"""
    pid = tl.program_id(0)
    
    # Calculate which head and sequence position this program handles
    head_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    block_id = pid // head_dim_blocks
    local_block_id = pid % head_dim_blocks
    
    head_id = block_id // seq_len
    seq_id = block_id % seq_len
    dim_id = local_block_id * BLOCK_SIZE
    dim_end = min(dim_id + BLOCK_SIZE, head_dim)
    
    # Check bounds
    if head_id >= num_heads or seq_id >= seq_len or dim_id >= head_dim:
        return
    
    # For each dimension, process the RoPE transformation
    for dim in range(dim_id, dim_end):
        # Load K element (from the slice that has seq_len elements)
        k_val = tl.load(k_ptr + batch_size * num_heads * seq_len * head_dim + 
                       head_id * seq_len * head_dim + seq_id * head_dim + dim)
        
        # Get positional embedding values
        if dim_pos < head_dim // 2:
            pos_val_0 = tl.load(pos_embed_0_ptr + dim_id)
            pos_val_1 = tl.load(pos_embed_1_ptr + dim_id)
        else:
            pos_val_0 = 0.0
            pos_val_1 = 0.0
        
        # Apply rotation similar to Q branch but transformed
        if dim % 2 == 0:
            rotated_val = k_val * pos_val_0
        else:
            rotated_val = k_val * pos_val_1
        
        tl.store(out_ptr + batch_size * num_heads * seq_len * head_dim + 
                 head_id * seq_len * head_dim + seq_id * head_dim + dim, 
                 rotated_val)

@triton.jit
def rope_k_full_kernel(
    k_ptr, pos_embed_0_ptr, pos_embed_1_ptr,
    out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Full K branch RoPE implementation"""
    pid = tl.program_id(0)
    
    # Calculate which head and sequence position this program handles
    total_elements = num_heads * seq_len * head_dim
    elements_per_program = total_elements // triton.cdiv(total_elements, BLOCK_SIZE)
    
    start_idx = pid * elements_per_program
    end_idx = min(start_idx + elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        # Convert linear index to head, seq, dim
        head_id = idx // (seq_len * head_dim)
        seq_id = (idx % (seq_len * head_dim)) // head_dim
        dim_id = idx % head_dim
        
        # Load K element from the processed slice (seq_len length)
        k_val = tl.load(k_ptr + head_id * seq_len * head_dim + seq_id * head_dim + dim_id)
        
        # Get positional embedding values
        pos_embed_idx = dim_id // 2
        if pos_embed_idx < (pos_embed_0_ptr.size(0) if hasattr(pos_embed_0_ptr, 'size') else head_dim // 2):
            pos_val_0 = tl.load(pos_embed_0_ptr + pos_embed_idx)
            pos_val_1 = tl.load(pos_embed_1_ptr + pos_embed_idx)
        else:
            pos_val_0 = 0.0
            pos_val_1 = 0.0
        
        # Apply RoPE transformation
        if dim_id % 2 == 0:
            result_val = k_val * pos_val_0
        else:
            result_val = k_val * pos_val_1
        
        tl.store(out_ptr + head_id * seq_len * head_dim + seq_id * head_dim + dim_id, 
                 result_val)

@torch.fx.wrap
def rope_k_optimized(pos_split_tensor, k_tensor):
    """Optimized K branch RoPE computation"""
    # Get dimensions based on the actual tensor shapes
    batch_size = 1
    num_heads = k_tensor.size(1)
    seq_len = k_tensor.size(2)
    head_dim = k_tensor.size(3)
    
    BLOCK_SIZE = 64  # Optimal for head_dim = 64
    total_elements = num_heads * seq_len * head_dim
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    output = torch.empty_like(k_tensor)
    
    # Split position embedding tensor
    pos_0 = pos_split_tensor[0]  # First half
    pos_1 = pos_split_tensor[1]  # Second half
    
    rope_k_full_kernel[grid](
        k_tensor, pos_0, pos_1,
        output,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return rope_k_optimized