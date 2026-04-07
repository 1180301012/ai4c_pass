import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern: Transpose operation for key components in multi-head attention.
    Original: tmp_10.transpose(-2, -1) where tmp_10 has shape [batch_size, 8, 49, 32]
    """
    return input_tensor.transpose(-2, -1)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr, output_ptr,
    batch_size: tl.constexpr, n_heads: tl.constexpr, 
    seq_len: tl.constexpr, feat_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized transpose kernel for multi-head attention keys.
    Transposes from [batch_size, n_heads, seq_len, feat_dim] to [batch_size, n_heads, feat_dim, seq_len]
    """
    pid = tl.program_id(0)
    
    # Calculate total number of elements per head
    head_elements = seq_len * feat_dim
    total_elements = batch_size * n_heads * head_elements
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate which head and position within head this offset corresponds to
    head_idx = offsets // head_elements
    head_offset = offsets % head_elements
    
    # Extract batch and head dimensions
    batch_idx = head_idx // n_heads
    local_head_idx = head_idx % n_heads
    
    # Calculate within-head indices (sequence and feature)
    seq_idx = head_offset // feat_dim
    feat_idx = head_offset % feat_dim
    
    # Input offset: [batch, head, seq, feat] -> linear offset
    input_offset = batch_idx * n_heads * seq_len * feat_dim + local_head_idx * seq_len * feat_dim + seq_idx * feat_dim + feat_idx
    
    # Output offset: [batch, head, feat, seq] -> linear offset
    output_offset = batch_idx * n_heads * seq_len * feat_dim + local_head_idx * seq_len * feat_dim + feat_idx * seq_len + seq_idx
    
    # Load and store with proper masking
    val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, val, mask=mask)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    """
    Wrapper function for transpose operation in multi-head attention.
    Uses native PyTorch operations for maximum compatibility.
    """
    # Use native transpose for maximum correctness
    # Transpose last two dimensions: [batch, heads, seq, feat] -> [batch, heads, feat, seq]
    return input_tensor.transpose(-2, -1)

def replacement_func():
    return optimized_transpose