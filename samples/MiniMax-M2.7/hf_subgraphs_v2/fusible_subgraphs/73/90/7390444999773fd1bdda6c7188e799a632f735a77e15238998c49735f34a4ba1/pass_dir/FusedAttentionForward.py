"""
Fused Attention Forward Pass

This pass fuses the transformer attention pattern:
- linear → view → transpose → SDPA → transpose → reshape

Key optimizations:
1. Fused kernel eliminates multiple kernel launches and intermediate tensors
2. Optimized memory access patterns for attention computation
3. Reduced memory bandwidth by avoiding unnecessary copies
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def fused_attention_kernel(
    # Output pointer
    out_ptr,
    # Value tensor (already computed from linear)
    value_ptr,
    # Query tensor
    query_ptr,
    # Key tensor
    key_ptr,
    # Attention mask
    mask_ptr,
    # Strides
    out_stride_batch, out_stride_heads, out_stride_seq, out_stride_dim,
    value_stride_batch, value_stride_heads, value_stride_seq, value_stride_dim,
    query_stride_batch, query_stride_heads, query_stride_seq, query_stride_dim,
    key_stride_batch, key_stride_heads, key_stride_seq, key_stride_dim,
    mask_stride_batch, mask_stride_head, mask_stride_i, mask_stride_j,
    # Shapes
    batch_size, seq_len, num_heads, head_dim, hidden_dim,
    # Mask dimensions
    mask_dim_i, mask_dim_j,
    # Additional params
    is_causal: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused attention kernel that computes SDPA efficiently.
    Input/output format: [batch, num_heads, seq, head_dim]
    """
    # Get program ID
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate output offset for [batch, num_heads, seq, head_dim] format
    out_offset = (
        batch_idx * out_stride_batch +
        head_idx * out_stride_heads +
        seq_idx * out_stride_seq
    )
    
    # Create offset arrays for head dimension
    head_offsets = tl.arange(0, BLOCK_SIZE)
    head_mask = head_offsets < head_dim
    
    # Calculate attention scale
    scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))
    
    # Compute pointer bases for this batch/head
    query_base = (
        batch_idx * query_stride_batch +
        head_idx * query_stride_heads
    )
    
    key_base = (
        batch_idx * key_stride_batch +
        head_idx * key_stride_heads
    )
    
    value_base = (
        batch_idx * value_stride_batch +
        head_idx * value_stride_heads
    )
    
    # Compute attention scores: Q @ K^T for this specific query position
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over sequence to compute attention scores
    for k in range(seq_len):
        q_offset = query_base + seq_idx * query_stride_seq
        q = tl.load(query_ptr + q_offset + head_offsets, mask=head_mask, other=0.0)
        
        k_offset = key_base + k * key_stride_seq
        k_val = tl.load(key_ptr + k_offset + head_offsets, mask=head_mask, other=0.0)
        
        score = tl.sum(q * k_val) * scale
        acc = acc + score
    
    # Apply mask
    if mask_dim_i > 0 and mask_dim_j > 0:
        mask_base = batch_idx * mask_stride_batch
        mask_offset = mask_base + seq_idx * mask_stride_i
        mask_val = tl.load(mask_ptr + mask_offset, mask=False, other=0.0)
        acc = acc + mask_val
    
    # Softmax normalization
    max_val = tl.max(acc)
    acc_minus_max = acc - max_val
    exp_acc = tl.exp(acc_minus_max)
    sum_exp = tl.sum(exp_acc)
    norm_factor = 1.0 / sum_exp
    
    # Multiply with value
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for v in range(seq_len):
        v_offset = value_base + v * value_stride_seq
        v_val = tl.load(value_ptr + v_offset + head_offsets, mask=head_mask, other=0.0)
        result = result + exp_acc[v] * v_val
    
    result = result * norm_factor
    
    # Store output
    tl.store(out_ptr + out_offset + head_offsets, result, mask=head_mask)


@triton.jit
def triton_linear_kernel(
    out_ptr,
    input_ptr, weight_ptr, bias_ptr,
    input_stride_batch, input_stride_seq, input_stride_hidden,
    weight_stride_out, weight_stride_in,
    output_stride_batch, output_stride_seq, output_stride_hidden,
    batch_size, seq_len, hidden_dim, input_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for linear projection."""
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Each thread block handles one output element
    out_offset = (
        batch_idx * output_stride_batch +
        seq_idx * output_stride_seq +
        head_idx * BLOCK_SIZE
    )
    
    head_offsets = tl.arange(0, BLOCK_SIZE)
    head_mask = head_offsets < BLOCK_SIZE
    
    # Load bias
    bias = tl.load(bias_ptr + head_idx * BLOCK_SIZE + head_offsets, mask=head_mask, other=0.0)
    
    # Compute output element
    result = bias
    
    # Accumulate over input dimension
    for k in range(input_dim):
        input_offset = (
            batch_idx * input_stride_batch +
            seq_idx * input_stride_seq +
            k
        )
        weight_offset = (
            head_idx * BLOCK_SIZE +
            k * weight_stride_in
        )
        
        input_val = tl.load(input_ptr + input_offset, mask=False, other=0.0)
        weight_val = tl.load(weight_ptr + weight_offset + head_offsets, mask=head_mask, other=0.0)
        
        result = result + input_val * weight_val
    
    tl.store(out_ptr + out_offset + head_offsets, result, mask=head_mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the transformer attention pattern:
    linear → view → transpose → SDPA → transpose → reshape
    
    This pattern uses the in_4 (key) tensor's shape to infer dimensions.
    The in_4 tensor has shape [batch, num_heads, seq, head_dim].
    """
    # Linear projection for value
    linear_out = torch.nn.functional.linear(in_3, in_1, in_0)
    
    # Get dimensions from in_4 tensor which has shape [batch, num_heads, seq, head_dim]
    batch = in_4.size(0)
    num_heads = in_4.size(1)
    seq = in_4.size(2)
    head_dim = in_4.size(3)
    hidden = in_3.size(2)
    
    # View and transpose
    view_out = linear_out.view(batch, -1, num_heads, head_dim)
    transposed = view_out.transpose(1, 2)
    
    # SDPA with all positional arguments
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        in_5, in_4, transposed, in_2, 0.0, False
    )
    
    # Transpose back and reshape
    transposed_back = attn_out.transpose(1, 2)
    result = transposed_back.reshape(batch, seq, hidden)
    
    return result


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments needed for the replacement kernel.
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@torch.fx.wrap
def fused_attention_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Wrapper function for the fused attention kernel.
    Uses Triton for optimized computation.
    """
    # Get shapes from input tensors (same as pattern)
    batch_size = in_3.shape[0]
    seq_len = in_3.shape[1]
    hidden_dim = in_3.shape[2]
    
    # Infer from in_4 (key tensor) shape: [batch, num_heads, seq, head_dim]
    num_heads = in_4.shape[1]
    head_dim = in_4.shape[3]
    
    # Compute linear projection using Triton
    linear_out = torch.empty(
        (batch_size, seq_len, hidden_dim),
        dtype=in_3.dtype,
        device=in_3.device
    )
    
    # Launch linear kernel
    linear_grid = (batch_size, seq_len, num_heads)
    triton_linear_kernel[linear_grid](
        linear_out,
        in_3, in_1, in_0,
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        in_1.stride(0), in_1.stride(1),
        linear_out.stride(0), linear_out.stride(1), linear_out.stride(2),
        batch_size, seq_len, hidden_dim, in_3.shape[2],
        head_dim,
    )
    
    # Prepare value in (batch, num_heads, seq, head_dim) format
    view_dim0 = linear_out.shape[0]
    value_viewed = linear_out.view(view_dim0, -1, num_heads, head_dim)
    value_transposed = value_viewed.transpose(1, 2)
    
    # Allocate output tensor in (batch, num_heads, seq, head_dim) format
    output_mha = torch.empty(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=in_3.dtype,
        device=in_3.device
    )
    
    # Launch attention kernel
    attn_grid = (batch_size, num_heads, seq_len)
    
    fused_attention_kernel[attn_grid](
        output_mha,
        value_transposed,
        in_5,
        in_4,
        in_2,
        output_mha.stride(0), output_mha.stride(1), output_mha.stride(2), output_mha.stride(3),
        value_transposed.stride(0), value_transposed.stride(1), value_transposed.stride(2), value_transposed.stride(3),
        in_5.stride(0), in_5.stride(1), in_5.stride(2), in_5.stride(3),
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        batch_size, seq_len, num_heads, head_dim, hidden_dim,
        in_2.shape[2], in_2.shape[3],
        False,
        head_dim,
    )
    
    # Transpose back and reshape (same as pattern)
    output_transposed_back = output_mha.transpose(1, 2)
    result = output_transposed_back.reshape(batch_size, seq_len, hidden_dim)
    
    return result


def replacement_func():
    """
    Return the replacement function for the matched pattern.
    """
    return fused_attention_wrapper