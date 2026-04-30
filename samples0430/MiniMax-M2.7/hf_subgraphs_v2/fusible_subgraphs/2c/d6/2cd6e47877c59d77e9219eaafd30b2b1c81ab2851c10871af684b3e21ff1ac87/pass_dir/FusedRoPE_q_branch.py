"""
Fused Rotary Position Embedding (RoPE) optimization.
This pass fuses both query and key RoPE computation branches into a single efficient kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_kernel(
    # Query branch inputs
    q_ptr, cos_q_ptr, sin_q_ptr,
    # Key branch inputs  
    k_ptr, cos_k_ptr, sin_k_ptr,
    # Prepend tensor for key
    k_prepend_ptr,
    # Outputs
    out_q_ptr, out_k_ptr,
    # Sizes
    N_elements, D,
    # Strides for query
    stride_qb, stride_qh, stride_ql, stride_qd,
    # Strides for key
    stride_kb, stride_kh, stride_kl, stride_kd,
    # Strides for cos/sin
    stride_cos_l, stride_cos_d,
    # Strides for outputs
    stride_out_qb, stride_out_qh, stride_out_ql, stride_out_qd,
    stride_out_kb, stride_out_kh, stride_out_kl, stride_out_kd,
    # Strides for k_prepend
    stride_kp_b, stride_kp_h, stride_kp_l, stride_kp_d,
    # Grid params
    B, H, L, L_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RoPE kernel that computes both query and key rotations.
    Grid: one program per element
    """
    # Calculate position in flattened array
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements
    
    # Load all needed data
    # Query: [B, H, L, D] -> flatten to [B*H*L*D]
    # Key: [B, H, L, D] -> flatten to [B*H*L*D]
    
    # Calculate 4D indices for query
    q_idx = offsets
    q_b = q_idx // (H * L * D)
    q_idx = q_idx % (H * L * D)
    q_h = q_idx // (L * D)
    q_idx = q_idx % (L * D)
    q_l = q_idx // D
    q_d = q_idx % D
    
    # Load q value
    q_linear = q_b * (stride_qb) + q_h * (stride_qh) + q_l * (stride_ql) + q_d * (stride_qd)
    q = tl.load(q_ptr + q_linear, mask=mask, other=0.0)
    
    # Load cos/sin for query position
    cos_q = tl.load(cos_q_ptr + q_l * stride_cos_l + q_d * stride_cos_d, mask=mask, other=0.0)
    sin_q = tl.load(sin_q_ptr + q_l * stride_cos_l + q_d * stride_cos_d, mask=mask, other=0.0)
    
    # Apply RoPE to query: q_even * cos + (-q_odd) * sin
    is_even = (q_d % 2 == 0)
    q_even = tl.where(is_even, q, 0.0)
    q_odd = tl.where(is_even, 0.0, q)
    out_q = q_even * cos_q + (-q_odd) * sin_q
    
    # Calculate 4D indices for key output
    k_out_idx = offsets
    k_out_b = k_out_idx // (H * L_out * D)
    k_out_idx = k_out_idx % (H * L_out * D)
    k_out_h = k_out_idx // (L_out * D)
    k_out_idx = k_out_idx % (L_out * D)
    k_out_l = k_out_idx // D
    k_out_d = k_out_idx % D
    
    # Store output_q (for positions 1:L, i.e., skip the prepend element)
    # out_q positions correspond to [B, H, 1:L, D]
    q_out_linear = k_out_b * stride_out_qb + k_out_h * stride_out_qh + (k_out_l + 1) * stride_out_ql + k_out_d * stride_out_qd
    out_q_store_mask = (k_out_l < L)  # Only write for valid positions
    tl.store(out_q_ptr + q_out_linear, out_q, mask=mask & out_q_store_mask)
    
    # Handle prepend element (position 0 of output) with type conversion from k_prepend
    prepend_mask = (k_out_l == 0)
    if tl.sum(prepend_mask) > 0:
        kp_linear = k_out_b * stride_kp_b + k_out_h * stride_kp_h + 0 * stride_kp_l + k_out_d * stride_kp_d
        kp_val = tl.load(k_prepend_ptr + kp_linear, mask=mask, other=0.0)
        out_k_prepend = kp_val  # Just type conversion
        tl.store(out_k_ptr + k_out_b * stride_out_kb + k_out_h * stride_out_kh + 0 * stride_out_kl + k_out_d * stride_out_kd,
                out_k_prepend, mask=mask & prepend_mask)
    
    # Key RoPE computation
    # Calculate original key indices from output position (shift by 1)
    k_l = k_out_l - 1  # Original key position (0 is prepend, so k_l = 0 means original position 0)
    k_d = k_out_d
    
    # Only compute for k_l >= 0 (actual key elements, not prepend)
    k_valid = k_l >= 0
    
    # Load key value
    k_linear = k_out_b * stride_kb + k_out_h * stride_kh + k_l * stride_kl + k_d * stride_kd
    k = tl.load(k_ptr + k_linear, mask=mask & k_valid, other=0.0)
    
    # Load cos/sin for key position
    cos_k = tl.load(cos_k_ptr + k_l * stride_cos_l + k_d * stride_cos_d, mask=mask & k_valid, other=0.0)
    sin_k = tl.load(sin_k_ptr + k_l * stride_cos_l + k_d * stride_cos_d, mask=mask & k_valid, other=0.0)
    
    # Apply RoPE to key: k_even * cos + (-k_odd) * sin
    k_even = tl.where(is_even, k, 0.0)
    k_odd = tl.where(is_even, 0.0, k)
    out_k = k_even * cos_k + (-k_odd) * sin_k
    
    # Store key output (skip prepend position 0)
    k_out_linear = k_out_b * stride_out_kb + k_out_h * stride_out_kh + k_out_l * stride_out_kl + k_out_d * stride_out_kd
    tl.store(out_k_ptr + k_out_linear, out_k, mask=mask & k_valid & ~prepend_mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the complete RoPE computation pattern including both branches.
    
    Query branch:
    - tmp_1 = in_3 * in_1 (q * cos)
    - tmp_2 = in_3[..., 1::2] (slice odd)
    - tmp_3 = -tmp_2 (negate)
    - tmp_4 = in_3[..., ::2] (slice even)
    - tmp_5 = torch.stack([tmp_3, tmp_4], -1) (interleave)
    - tmp_6 = tmp_5.reshape((1, 6, 256, 64)) (reshape)
    - tmp_7 = tmp_6 * in_5 (q_rotated * sin)
    - tmp_8 = tmp_1 + tmp_7 (add)
    - tmp_9 = torch.cat([in_2, tmp_8], dim=2) (concat with prepend)
    - tmp_10 = tmp_9.type_as(in_6) (type conversion)
    
    Key branch:
    - tmp_11 = in_4[..., :1, :] (first element)
    - tmp_12 = in_4[..., 1:, :] (remaining)
    - tensor_split = in_0.tensor_split(2, -1) (split pos_embed)
    - tmp_14 = tensor_split[0], tmp_15 = tensor_split[1]
    - tmp_16 = tmp_12 * tmp_15 (k * cos)
    - tmp_17 = tmp_12[..., 1::2] (slice odd)
    - tmp_18 = -tmp_17 (negate)
    - tmp_19 = tmp_12[..., ::2] (slice even)
    - tmp_20 = torch.stack([tmp_18, tmp_19], -1) (interleave)
    - tmp_21 = tmp_20.reshape((1, 6, 256, 64)) (reshape)
    - tmp_22 = tmp_21 * tmp_14 (k_rotated * sin)
    - tmp_23 = tmp_16 + tmp_22 (add)
    - tmp_24 = torch.cat([tmp_11, tmp_23], dim=2) (concat with prepend)
    - tmp_25 = tmp_24.type_as(in_6) (type conversion)
    
    Returns: (tmp_25, tmp_10)
    """
    # Query branch
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_2 = None
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_3 = tmp_4 = None
    tmp_6 = tmp_5.reshape((1, 6, 256, 64))
    tmp_5 = None
    tmp_7 = tmp_6 * in_5
    tmp_6 = None
    tmp_8 = tmp_1 + tmp_7
    tmp_1 = tmp_7 = None
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_8 = None
    tmp_10 = tmp_9.type_as(in_6)
    tmp_9 = None
    
    # Key branch
    tmp_11 = in_4[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
    tmp_12 = in_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    in_4 = None
    tensor_split = in_0.tensor_split(2, -1)
    in_0 = None
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    tensor_split = None
    tmp_16 = tmp_12 * tmp_15
    tmp_15 = None
    tmp_17 = tmp_12[(Ellipsis, slice(1, None, 2))]
    tmp_18 = -tmp_17
    tmp_17 = None
    tmp_19 = tmp_12[(Ellipsis, slice(None, None, 2))]
    tmp_12 = None
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_18 = tmp_19 = None
    tmp_21 = tmp_20.reshape((1, 6, 256, 64))
    tmp_20 = None
    tmp_22 = tmp_21 * tmp_14
    tmp_21 = tmp_14 = None
    tmp_23 = tmp_16 + tmp_22
    tmp_16 = tmp_22 = None
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_11 = tmp_23 = None
    tmp_25 = tmp_24.type_as(in_6)
    tmp_24 = in_6 = None
    
    return (tmp_25, tmp_10)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Extract arguments needed for the fused RoPE kernel."""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def fused_rope_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Fused RoPE computation for both query and key tensors.
    
    Inputs:
    - in_0: pos_embed [L, 128] - position embeddings (will be split)
    - in_1: cos_emb [L, 64] - cosine embeddings for query
    - in_2: q_prepend [B, H, 1, D] - first query element to prepend
    - in_3: q [B, H, L, D] - query tensor
    - in_4: k [B, H, L+1, D] - key tensor (will be split)
    - in_5: sin_emb [L, 64] - sine embeddings for query
    - in_6: v [B, H, L+1, D] - value tensor (for dtype reference)
    
    Returns: (k_out, q_out) where k_out is [B, H, L+1, D] and q_out is [B, H, L+1, D]
    """
    B, H_k, L_plus_1, D = in_4.shape
    L = L_plus_1 - 1  # Original key length
    
    B_q, H_q, L_q, _ = in_3.shape
    target_dtype = in_6.dtype
    
    # Split in_0 into cos and sin parts for key
    # in_0 has shape [L, 128], split on last dim gives [L, 64] each
    cos_k, sin_k = in_0.split(64, dim=-1)
    
    # Output shapes
    out_k_shape = (B, H_k, L_plus_1, D)
    out_q_shape = (B_q, H_q, L_plus_1, D)
    
    # Allocate outputs
    out_k = torch.empty(out_k_shape, dtype=target_dtype, device=in_4.device)
    out_q = torch.empty(out_q_shape, dtype=target_dtype, device=in_3.device)
    
    # Extract key prepend (first element)
    k_prepend = in_4[:, :, :1, :]
    # Extract key content (remaining elements)
    k_content = in_4[:, :, 1:, :]
    
    N_elements = B * H_q * L_plus_1 * D
    
    # Grid: one program per BLOCK_SIZE elements
    BLOCK_SIZE = 128
    grid = ((N_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_rope_kernel[grid](
        # Query branch
        in_3, in_1, in_5,
        # Key branch
        k_content, cos_k, sin_k,
        # Prepend tensor
        k_prepend,
        # Outputs
        out_q, out_k,
        # Sizes
        N_elements, D,
        # Query strides
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        # Key strides
        k_content.stride(0), k_content.stride(1), k_content.stride(2), k_content.stride(3),
        # Cos/sin strides
        in_1.stride(0), in_1.stride(1),
        # Output strides
        out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
        out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
        # Prepend strides
        k_prepend.stride(0), k_prepend.stride(1), k_prepend.stride(2), k_prepend.stride(3),
        # Grid params
        B, H_q, L_q, L_plus_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_k, out_q)


def replacement_func():
    """Return the fused RoPE wrapper function."""
    return fused_rope_wrapper