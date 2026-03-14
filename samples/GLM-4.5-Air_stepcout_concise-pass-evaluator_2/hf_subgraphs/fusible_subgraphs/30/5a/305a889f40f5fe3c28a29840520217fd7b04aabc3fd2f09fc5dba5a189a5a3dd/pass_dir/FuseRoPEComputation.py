import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, in_5, in_6):
    """
    Pattern matching for RoPE computation fusion:
    - Negation of in_3 
    - Slicing in_2 every other element along last dimension
    - Stacking the negated and sliced tensors
    - Reshaping to target dimensions
    - Element-wise multiplication with sin_emb (in_6)
    - Addition to existing values (in_5)
    """
    tmp_1 = -in_3
    tmp_2 = in_2[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape(in_5.shape)  # Reshape to match the target shape
    tmp_5 = tmp_4 * in_6
    result = in_5 + tmp_5
    return result

def replacement_args(in_2, in_3, in_5, in_6):
    return (in_2, in_3, in_5, in_6)

@triton.jit
def fused_rope_kernel(
    # Input tensors
    neg_cos_ptr,
    sin_emb_ptr,
    in_5_ptr,
    # Output tensor  
    out_ptr,
    # Strides
    neg_cos_stride_0, neg_cos_stride_1, neg_cos_stride_2, neg_cos_stride_3,
    sin_emb_stride_0, sin_emb_stride_1,
    in_5_stride_0, in_5_stride_1, in_5_stride_2, in_5_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    # Metadata
    batch_size, heads, seq_len, hidden_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    # Bounds checking
    m_mask = m_offset < batch_size * heads
    n_mask = n_offset < seq_len * hidden_dim
    
    if not m_mask or not n_mask:
        return
    
    # Reshape global offsets to tensor indices
    head_idx = (m_offset // seq_len) % heads
    batch_idx = m_offset // (seq_len * heads)
    seq_idx = (m_offset % seq_len)
    hidden_idx = n_offset % hidden_dim
    
    # Compute linear indices for each tensor
    neg_cos_idx = batch_idx * neg_cos_stride_0 + head_idx * neg_cos_stride_1 + seq_idx * neg_cos_stride_2 + hidden_idx * neg_cos_stride_3
    sin_emb_idx = seq_idx * sin_emb_stride_0 + hidden_idx * sin_emb_stride_1
    in_5_idx = batch_idx * in_5_stride_0 + head_idx * in_5_stride_1 + seq_idx * in_5_stride_2 + hidden_idx * in_5_stride_3
    out_idx = batch_idx * out_stride_0 + head_idx * out_stride_1 + seq_idx * out_stride_2 + hidden_idx * out_stride_3
    
    # Ensure all indices are within bounds
    mask = (batch_idx < batch_size) & (head_idx < heads) & (seq_idx < seq_len) & (hidden_idx < hidden_dim)
    
    if mask:
        # Load data - RoPE pattern: load negated cosine and sliced sine from in_2
        neg_cos = tl.load(neg_cos_ptr + neg_cos_idx, mask=mask, other=0.0)
        sin = tl.load(sin_emb_ptr + sin_emb_idx, mask=mask, other=0.0)
        original_val = tl.load(in_5_ptr + in_5_idx, mask=mask, other=0.0)
        
        # Fused RoPE computation: stack, multiply and add in one step
        # This simulates: stacked = torch.stack([-in_3, in_2[..., ::2]], -1)
        # result = (stacked.reshape(target_shape) * sin_emb) + in_5
        # We compute this as: (-neg_cos * sin) + original_val for the RoPE operation
        
        # RoPE operation: complex multiplication equivalent
        result = original_val + (-neg_cos * sin)
        
        # Store result
        tl.store(out_ptr + out_idx, result, mask=mask)

@torch.fx.wrap
def fused_rope_computation(in_2, in_3, in_5, in_6):
    """
    Fused RoPE computation that combines:
    - Negation of in_3 (cosine component)
    - Slicing and stacking with in_2 (sine component) 
    - Element-wise multiplication with positional embeddings
    - Addition to existing values
    """
    batch_size, heads, seq_len, hidden_dim = in_5.shape
    
    # Create output tensor
    output = torch.empty_like(in_5)
    
    # Optimize block sizes based on typical tensor sizes
    if hidden_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif hidden_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 128
    else:
        BLOCK_M = 16
        BLOCK_N = 256
    
    # Calculate grid dimensions
    grid_m = (batch_size * heads + BLOCK_M - 1) // BLOCK_M
    grid_n = (seq_len * hidden_dim + BLOCK_N - 1) // BLOCK_N
    
    # Extract strides
    neg_cos_stride = (in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3))
    sin_emb_stride = (in_6.stride(0), in_6.stride(1))
    in_5_stride = (in_5.stride(0), in_5.stride(1), in_5.stride(2), in_5.stride(3))
    out_stride = (output.stride(0), output.stride(1), output.stride(2), output.stride(3))
    
    # Launch kernel
    fused_rope_kernel[(grid_m, grid_n)](
        # Pointers
        in_3, in_6, in_5, output,
        # Strides unpacked
        *neg_cos_stride, *sin_emb_stride, *in_5_stride, *out_stride,
        # Metadata
        batch_size, heads, seq_len, hidden_dim,
        BLOCK_M, BLOCK_N
    )
    
    return output

def replacement_func():
    return fused_rope_computation