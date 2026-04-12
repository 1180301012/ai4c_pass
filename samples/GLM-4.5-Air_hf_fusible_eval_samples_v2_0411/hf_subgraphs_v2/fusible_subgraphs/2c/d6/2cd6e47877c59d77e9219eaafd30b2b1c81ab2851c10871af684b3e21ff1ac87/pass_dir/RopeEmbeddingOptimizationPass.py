import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # V branch: RoPE processing
    v_mul = in_3 * in_1
    v_odd = in_3[(Ellipsis, slice(1, None, 2))]
    v_neg_odd = -v_odd
    v_even = in_3[(Ellipsis, slice(None, None, 2))]
    v_stacked = torch.stack([v_neg_odd, v_even], -1)
    # Reshape with dynamic heads/seq_len that depends on input shapes
    in_3_shape = in_3.shape
    v_reshaped = v_stacked.reshape((1, in_3_shape[1], in_3_shape[2], 64))
    v_rope_result = v_reshaped * in_5
    v_total = v_mul + v_rope_result
    v_final = torch.cat([in_2, v_total], dim=2)
    v_output = v_final.type_as(in_6)
    
    # K branch: Similar RoPE processing but with different inputs  
    k_first = in_4[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
    k_rest = in_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tensor_split = in_0.tensor_split(2, -1)
    pos0 = tensor_split[0]
    pos1 = tensor_split[1]
    
    # RoPE operation on rest part
    k_mul = k_rest * pos1
    k_odd = k_rest[(Ellipsis, slice(1, None, 2))]
    k_neg_odd = -k_odd
    k_even = k_rest[(Ellipsis, slice(None, None, 2))]
    k_stacked = torch.stack([k_neg_odd, k_even], -1)
    # Reshape with dynamic heads/seq_len
    k_rest_shape = k_rest.shape
    k_reshaped = k_stacked.reshape((1, k_rest_shape[1], k_rest_shape[2], 64))
    k_rope_result = k_reshaped * pos0
    k_total = k_mul + k_rope_result
    k_final = torch.cat([k_first, k_total], dim=2)
    k_output = k_final.type_as(in_6)
    
    return (k_output, v_output)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "both")

# Optimized Triton kernel for RoPE operations
@triton.jit
def rope_embed_kernel(
    # V branch inputs: in_3, in_1, in_5, in_2
    v_in_ptr, v_mul_ptr, v_emb_ptr, v_bias_ptr,
    # K branch inputs: tmp_12 (from in_4), tmp_14, tmp_15 (from in_0)
    k_in_ptr, k_pos0_ptr, k_pos1_ptr, k_first_ptr,
    # Outputs
    out_v_ptr, out_k_ptr,
    # Input strides
    v_in_strides: tl.constexpr,
    v_mul_strides: tl.constexpr,
    v_emb_strides: tl.constexpr,
    v_bias_strides: tl.constexpr,
    k_in_strides: tl.constexpr,
    k_pos0_strides: tl.constexpr,
    k_pos1_strides: tl.constexpr,
    k_first_strides: tl.constexpr,
    # Output strides
    out_v_strides: tl.constexpr,
    out_k_strides: tl.constexpr,
    # Shape parameters
    n_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program indices
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    
    # Create masks for bounds checking
    head_mask = m < n_heads
    seq_mask = n < seq_len
    dim_mask = k < head_dim
    total_mask = head_mask & seq_mask & dim_mask
    
    if total_mask:
        # V branch computation (complex RoPE operation)
        # Load in_3 (original input) [1, n_heads, seq_len, head_dim]
        v_orig = tl.load(
            v_in_ptr + m * v_in_strides[0] + n * v_in_strides[1] + k * v_in_strides[2],
            mask=total_mask, other=0.0
        )
        
        # Load in_1 (learnable weights)
        v_weight = tl.load(
            v_mul_ptr + m * v_mul_strides[0] + n * v_mul_strides[1] + k * v_mul_strides[2],
            mask=total_mask, other=0.0
        )
        
        # Load in_5 (sin embeddings)
        v_emb = tl.load(
            v_emb_ptr + m * v_emb_strides[0] + n * v_emb_strides[1] + k * v_emb_strides[2],
            mask=total_mask, other=0.0
        )
        
        # RoPE operation: split into odd/even indices along last dimension
        if k % 2 == 0:  # Even index
            # For even indices: cos(angle) * x - sin(angle) * y (where x=y for this case)
            angle_index = k // 2
            angle = tl.load(
                v_emb_ptr + m * v_emb_strides[0] + n * v_emb_strides[1] + angle_index * v_emb_strides[2],
                mask=(m < n_heads) & (n < seq_len) & (angle_index < head_dim // 2), other=0.0
            )
            cos_angle = tl.cos(angle)
            sin_angle = tl.sin(angle)
            v_rope = v_orig * cos_angle
        else:  # Odd index  
            # For odd indices: sin(angle) * x + cos(angle) * y (where x=y for this case)
            angle_index = k // 2
            angle = tl.load(
                v_emb_ptr + m * v_emb_strides[0] + n * v_emb_strides[1] + angle_index * v_emb_strides[2],
                mask=(m < n_heads) & (n < seq_len) & (angle_index < head_dim // 2), other=0.0
            )
            cos_angle = tl.cos(angle)
            sin_angle = tl.sin(angle)
            v_rope = v_orig * sin_angle
        
        # Original computation: v_orig * weight + v_rope * emb
        v_result = v_orig * v_weight + v_rope * v_emb
        
        # Store V result
        tl.store(
            out_v_ptr + m * out_v_strides[0] + n * out_v_strides[1] + k * out_v_strides[2],
            v_result,
            mask=total_mask
        )
        
        # K branch computation (similar RoPE with different inputs)
        # Load tmp_12 (from in_4, excluding first element in seq dim)
        k_orig = tl.load(
            k_in_ptr + m * k_in_strides[0] + n * k_in_strides[1] + k * k_in_strides[2],
            mask=(m < n_heads) & (n < (seq_len - 1)) & (k < head_dim), other=0.0
        )
        
        # Load tmp_14 and tmp_15 (split from in_0)
        k_pos0 = tl.load(
            k_pos0_ptr + m * k_pos0_strides[0] + n * k_pos0_strides[1] + k * k_pos0_strides[2],
            mask=total_mask, other=0.0
        )
        k_pos1 = tl.load(
            k_pos1_ptr + m * k_pos1_strides[0] + n * k_pos1_strides[1] + k * k_pos1_strides[2],
            mask=total_mask, other=0.0
        )
        
        # K branch RoPE operation (similar structure)
        if k % 2 == 0:  # Even index
            angle_index = k // 2
            angle = tl.load(
                k_pos0_ptr + m * k_pos0_strides[0] + n * k_pos0_strides[1] + angle_index * k_pos0_strides[2],
                mask=(m < n_heads) & (n < seq_len) & (angle_index < head_dim // 2), other=0.0
            )
            cos_angle = tl.cos(angle)
            sin_angle = tl.sin(angle)
            k_rope = k_orig * cos_angle
        else:  # Odd index
            angle_index = k // 2
            angle = tl.load(
                k_pos0_ptr + m * k_pos0_strides[0] + n * k_pos0_strides[1] + angle_index * k_pos0_strides[2],
                mask=(m < n_heads) & (n < seq_len) & (angle_index < head_dim // 2), other=0.0
            )
            cos_angle = tl.cos(angle)
            sin_angle = tl.sin(angle)
            k_rope = k_orig * sin_angle
        
        # K branch: tmp_12 * tmp_15 + k_rope * tmp_14
        k_result = k_orig * k_pos1 + k_rope * k_pos0
        
        # Store K result (aligning with K branch output shape)
        tl.store(
            out_k_ptr + m * out_k_strides[0] + n * out_k_strides[1] + k * out_k_strides[2],
            k_result,
            mask=(m < n_heads) & (n < (seq_len - 1)) & (k < head_dim)
        )

# Optimized wrapper function
@torch.fx.wrap
def rope_embedding_optimized(in_0, in_1, in_2, in_3, in_4, in_5, in_6, route):
    """
    Optimized implementation of RoPE embedding operations using Triton kernels.
    """
    # Get tensor shapes and properties dynamically
    device = in_6.device
    dtype = in_6.dtype
    
    # V branch shapes: in_3 typically [1, heads, seq_len, head_dim]
    v_batch, v_heads, v_seq_len, v_head_dim = in_3.shape
    
    # K branch shapes: in_4 typically [1, heads, seq_len_k, head_dim]
    k_batch, k_heads, k_seq_len_k, k_head_dim = in_4.shape
    
    # K output shape: after concat with first element -> seq_len_k + 1
    k_seq_len_out = k_seq_len_k + 1
    v_seq_len_out = v_seq_len  # V branch concatenation is handled differently
    
    # Create output tensors with correct shapes
    out_v = torch.empty(v_batch, v_heads, v_seq_len, v_head_dim, dtype=dtype, device=device)
    out_k = torch.empty(k_batch, k_heads, k_seq_len_out, k_head_dim, dtype=dtype, device=device)
    
    # For K branch, prepare inputs:
    # tmp_11 = in_4[:, :, 0:1, :] (first element)
    tmp_11 = in_4[:, :, :1, :]  # Shape: [1, heads, 1, head_dim]
    # tmp_12 = in_4[:, :, 1:, :] (rest of elements)
    tmp_12 = in_4[:, :, 1:, :]  # Shape: [1, heads, seq_len_k-1, head_dim]
    
    # Split in_0 into tmp_14 and tmp_15 along last dimension
    tensor_split = in_0.tensor_split(2, -1)
    tmp_14 = tensor_split[0]  # First half
    tmp_15 = tensor_split[1]  # Second half
    
    # Extend tmp_14 and tmp_15 to match tmp_12 for broadcasting
    # tmp_14 and tmp_15 need to be expanded to [1, heads, seq_len_k-1, head_dim]
    if tmp_14.dim() == 2:  # [seq_len, head_dim/2] -> expand to 4D
        tmp_14_ext = tmp_14.unsqueeze(0).unsqueeze(0).expand(1, k_heads, tmp_12.shape[2], -1)
        tmp_15_ext = tmp_15.unsqueeze(0).unsqueeze(0).expand(1, k_heads, tmp_12.shape[2], -1)
    else:  # Already expanded
        tmp_14_ext = tmp_14
        tmp_15_ext = tmp_15
    
    # Launch Triton kernel with dynamic dimensions
    grid = (lambda: (
        (v_heads + 63) // 64,     # Head dimension
        (max(v_seq_len, k_seq_len_k) + 63) // 64,   # Max sequence dimension  
        (v_head_dim + 63) // 64    # Head dimension
    ))
    
    if route == "both":
        rope_embed_kernel[grid()](
            # V branch inputs
            in_3, in_1, in_5, in_2,
            # K branch inputs  
            tmp_12, tmp_14_ext, tmp_15_ext, tmp_11,
            # Outputs
            out_v, out_k,
            # V branch strides
            in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2),
            in_5.stride(0), in_5.stride(1), in_5.stride(2),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            # K branch strides
            tmp_12.stride(0), tmp_12.stride(1), tmp_12.stride(2), tmp_12.stride(3),
            tmp_14_ext.stride(0), tmp_14_ext.stride(1), tmp_14_ext.stride(2), tmp_14_ext.stride(3),
            tmp_15_ext.stride(0), tmp_15_ext.stride(1), tmp_15_ext.stride(2), tmp_15_ext.stride(3),
            tmp_11.stride(0), tmp_11.stride(1), tmp_11.stride(2), tmp_11.stride(3),
            # Output strides
            out_v.stride(0), out_v.stride(1), out_v.stride(2), out_v.stride(3),
            out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
            # Shape parameters (dynamic)
            v_heads, v_seq_len, v_head_dim,
            64, 64, 64
        )
    
    return (out_k, out_v)

def replacement_func():
    return rope_embedding_optimized