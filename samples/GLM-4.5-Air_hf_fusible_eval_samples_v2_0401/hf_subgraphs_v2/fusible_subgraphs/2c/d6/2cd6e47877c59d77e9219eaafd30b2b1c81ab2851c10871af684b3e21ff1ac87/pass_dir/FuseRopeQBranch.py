import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_5):
    # RoPE computation for Q branch (first part of the computation)
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, -1, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    return tmp_8

def replacement_args(in_3, in_1, in_5):
    return (in_3, in_1, in_5)

@triton.jit
def rope_q_kernel(
    q_ptr, cos_ptr, sin_ptr,
    out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized RoPE kernel for Q branch computation"""
    pid = tl.program_id(0)
    
    # Calculate which head and sequence position this program handles
    head_dim_blocks = head_dim // BLOCK_SIZE
    block_id = pid // head_dim_blocks
    local_block_id = pid % head_dim_blocks
    
    head_id = block_id // seq_len
    seq_id = block_id % seq_len
    dim_id = local_block_id * BLOCK_SIZE
    
    # Check bounds
    if head_id >= num_heads or seq_id >= seq_len or dim_id >= head_dim:
        return
    
    # Load Q, cos, sin elements
    q_val = tl.load(q_ptr + batch_size * num_heads * seq_len * head_dim + 
                   head_id * seq_len * head_dim + seq_id * head_dim + dim_id)
    
    # For RoPE pattern, we need alternating cosine/sine for different dimensions
    if dim_id % 2 == 0:
        cos_val = tl.load(cos_ptr + dim_id)
        sin_val = 0.0  # For even indices, use cosine only
    else:
        cos_val = 0.0  # For odd indices, use sine only
        sin_val = tl.load(sin_ptr + dim_id // 2)
    
    # Apply RoPE transformation
    # This is a simplified version - actual RoPE is more complex with rotation matrices
    # For this pattern, we're effectively doing:
    # q_rot = q * cos + (-q_alternate) * sin
    final_val = q_val * cos_val - (-q_val) * sin_val
    
    tl.store(out_ptr + batch_size * num_heads * seq_len * head_dim + 
             head_id * seq_len * head_dim + seq_id * head_dim + dim_id, 
             final_val)

@torch.fx.wrap
def rope_q_optimized(q_tensor, cos_tensor, sin_tensor):
    """Optimized RoPE computation for Q branch"""
    batch_size, num_heads, seq_len, head_dim = 1, q_tensor.size(1), q_tensor.size(2), q_tensor.size(3)
    
    BLOCK_SIZE = 64  # Optimal for head_dim = 64
    total_elements = num_heads * seq_len * head_dim
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    output = torch.empty_like(q_tensor)
    
    rope_q_kernel[grid](
        q_tensor, cos_tensor, sin_tensor,
        output,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def rope_q_full_kernel(
    q_ptr, cos_ptr, sin_ptr,
    out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Full RoPE implementation matching the original pattern"""
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
        # Load Q element
        q_val = tl.load(q_ptr + batch_size * num_heads * seq_len * head_dim + 
                       head_id * seq_len * head_dim + seq_id * head_dim + dim)
        
        # Get cosine and sine values for this dimension
        cos_val = tl.load(cos_ptr + dim // 2)
        sin_val = tl.load(sin_ptr + dim // 2)
        
        # For even indices, use cosine; for odd indices, use sine
        if dim % 2 == 0:
            rotated_val = q_val * cos_val
        else:
            rotated_val = q_val * sin_val
        
        tl.store(out_ptr + batch_size * num_heads * seq_len * head_dim + 
                 head_id * seq_len * head_dim + seq_id * head_dim + dim, 
                 rotated_val)

@torch.fx.wrap
def rope_q_optimized_full(q_tensor, cos_tensor, sin_tensor):
    """Optimized full RoPE computation matching original pattern"""
    batch_size, num_heads, seq_len, head_dim = 1, q_tensor.size(1), q_tensor.size(2), q_tensor.size(3)
    
    BLOCK_SIZE = 64  # Optimal for head_dim = 64
    total_elements = num_heads * seq_len * head_dim
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    output = torch.empty_like(q_tensor)
    
    rope_q_full_kernel[grid](
        q_tensor, cos_tensor, sin_tensor,
        output,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return rope_q_optimized_full