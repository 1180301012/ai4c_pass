"""
Fused Attention Kernel optimization for multi-head attention mechanism.

This pass replaces the attention computation (matmul + scale + softmax + matmul)
with a single optimized Triton kernel.
"""

import torch
import triton
import triton.language as tl


# Autotuning configuration for best performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_D': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_D': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
    ],
    key=['B', 'H', 'N', 'D'],
)
@triton.jit
def fused_attention_kernel(
    # Pointers
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Scales
    scale: tl.constexpr,
    # Dimensions
    B: tl.int32, H: tl.int32, N: tl.int32, D: tl.int32,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Fully fused attention kernel:
    Q @ K -> scale -> softmax -> softmax @ V
    
    This fuses the entire attention computation into a single GPU kernel.
    """
    # Get program indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Each program handles a query row (seq_len dimension)
    # Offsets for this row
    row_idx = tl.program_id(2)
    
    q_offset = (batch_idx * H * N * D + head_idx * N * D + 
                row_idx * D + tl.arange(0, BLOCK_D))
    
    # Load query for this row
    q_mask = tl.arange(0, BLOCK_D) < D
    q = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0, eviction_policy='evict_last')
    q = q.to(tl.float32)
    
    # Compute attention scores: q @ k for this row vs all keys
    # Initialize accumulator
    attn = tl.zeros((N,), dtype=tl.float32)
    
    # Iterate over key blocks
    for block_start in range(0, N, BLOCK_N):
        block_n_offsets = block_start + tl.arange(0, BLOCK_N)
        
        # K matrix offsets for this block
        k_offset = (batch_idx * H * N * D + head_idx * N * D +
                    block_n_offsets * D + tl.arange(0, BLOCK_D)[:, None] * 0)
        
        # Actually need to compute k properly - k is [B, H, D, N]
        # So k[head, :, key_idx] = k_ptr[stride_kb*batch + stride_kh*head + stride_kn*D + stride_kd*key_idx : ...]
        k_base = batch_idx * stride_kb + head_idx * stride_kh
        
        # Load key - shape is [D, N] for this batch/head
        k = tl.load(
            k_ptr + k_base + stride_kn * block_start + stride_kd * 0 + 
                  tl.arange(0, BLOCK_N)[None, :] * stride_kn +
                  tl.arange(0, BLOCK_D)[:, None] * stride_kd,
            mask=(block_start + tl.arange(0, BLOCK_N)[None, :] < N) & 
                 (tl.arange(0, BLOCK_D)[:, None] < D),
            other=0.0
        )
        k = k.to(tl.float32)
        
        # Compute dot product: q (D,) @ k (D, BLOCK_N) -> (BLOCK_N,)
        attn_block = tl.dot(q, k)
        
        # Store to accumulator
        attn_start = block_start
        tl.store(attn + attn_start, attn_block, mask=block_start + tl.arange(0, BLOCK_N) < N)
    
    # Apply scaling
    inv_scale = 1.0 / scale
    scaled_attn = attn * inv_scale
    
    # Softmax with numerical stability
    max_attn = tl.max(scaled_attn)
    exp_attn = tl.exp(scaled_attn - max_attn)
    sum_exp = tl.sum(exp_attn)
    softmax_attn = exp_attn / sum_exp
    
    # Compute output: softmax_attn (N,) @ v (N,D) -> (D,)
    output = tl.zeros((D,), dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_N):
        block_n_offsets = block_start + tl.arange(0, BLOCK_N)
        
        # V matrix - v is [B, H, N, D], so v[batch, head, n, :] is the row
        v_base = batch_idx * stride_vb + head_idx * stride_vh
        
        # Load value block: shape [BLOCK_N, D]
        v = tl.load(
            v_ptr + v_base + stride_vn * block_start + 
                  block_n_offsets[:, None] * stride_vn +
                  tl.arange(0, D)[None, :] * stride_vd,
            mask=(block_start + block_n_offsets[:, None] < N) & 
                 (tl.arange(0, D)[None, :] < D),
            other=0.0
        )
        v = v.to(tl.float32)
        
        # Load softmax values for this block
        attn_block = tl.load(
            softmax_attn + block_start,
            mask=block_start + tl.arange(0, BLOCK_N) < N,
            other=0.0
        )
        
        # Multiply: attn_block (BLOCK_N,) @ v (BLOCK_N, D) -> (D,)
        output += tl.dot(attn_block, v)
    
    # Write output
    o_offset = (batch_idx * stride_ob + head_idx * stride_oh + 
                row_idx * stride_on + tl.arange(0, D) * stride_od)
    o_mask = tl.arange(0, D) < D
    
    tl.store(o_ptr + o_offset, output, mask=o_mask)


@triton.jit  
def fused_scale_softmax_bf16_kernel(
    input_ptr,
    output_ptr,
    scale: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    BFloat16 version of the fused scale + softmax kernel.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data and convert to float for computation
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply scaled softmax
    inv_scale = 1.0 / scale
    scaled_x = x * inv_scale
    exp_x = tl.exp(scaled_x)
    
    # Store result
    tl.store(output_ptr + offsets, exp_x.to(tl.bfloat16), mask=mask)


@triton.jit
def fused_scale_softmax_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fully fused attention kernel: matmul(query, key) -> scale -> softmax -> matmul(attn, value)
    
    This kernel fuses the entire attention computation into a single GPU kernel.
    """
    # Get program indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    # Compute row offset in output
    row_offset = (batch_idx * H + head_idx) * N + row_idx
    
    # Offsets for query, key, value, output
    q_offset = (batch_idx * H + head_idx) * N * D + row_idx * D + tl.arange(0, D)
    k_offset_base = (batch_idx * H + head_idx) * N * D + tl.arange(0, D)[:, None] * N + tl.arange(0, N)[None, :]
    v_offset_base = (batch_idx * H + head_idx) * N * D + tl.arange(0, N)[:, None] * D + tl.arange(0, D)[None, :]
    
    # Bounds mask
    q_mask = q_offset < (batch_idx * H + head_idx + 1) * N * D
    k_mask = k_offset_base < N * N
    v_mask = v_offset_base < N * D
    
    # Load query
    q = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0)
    
    # Compute attention scores: q @ k
    # Each thread block handles a subset of N
    attn = tl.zeros((D, N), dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_N):
        block_offsets = block_start + tl.arange(0, BLOCK_N)
        
        # Load key block
        k_block = tl.load(
            k_ptr + (batch_idx * H + head_idx) * N * D + block_offsets[None, :] * N + tl.arange(0, D)[:, None],
            mask=(block_offsets[None, :] < N) & (tl.arange(0, D)[:, None] < D),
            other=0.0
        )
        
        # Multiply q with key block
        attn += tl.dot(q[:, None], k_block[None, :])
    
    # Apply scaling
    scaled_attn = attn * (1.0 / scale)
    
    # Softmax
    max_attn = tl.max(scaled_attn, axis=0)
    exp_attn = tl.exp(scaled_attn - max_attn[None, :])
    sum_exp = tl.sum(exp_attn, axis=0)
    softmax_attn = exp_attn / sum_exp[None, :]
    
    # Compute output: softmax_attn @ v
    # Load value
    v = tl.load(
        v_ptr + (batch_idx * H + head_idx) * N * D + tl.arange(0, N)[:, None] * D + tl.arange(0, D)[None, :],
        mask=(tl.arange(0, N)[:, None] < N) & (tl.arange(0, D)[None, :] < D),
        other=0.0
    )
    
    output = tl.dot(softmax_attn, v)
    
    # Store output
    o_offset = row_offset * D + tl.arange(0, D)
    o_mask = o_offset < B * H * N * D
    tl.store(o_ptr + o_offset, output, mask=o_mask)


@torch.fx.wrap
def _triton_fused_attention(input_tensor, output_tensor, scale_val, B, H, N, K):
    """
    Triton kernel wrapper for fused scale + softmax computation.
    """
    # Launch kernel with grid: (B, H, N) for batch, heads, seq_len
    grid = (B, H, N)
    
    fused_attention_kernel[grid](
        input_tensor,  # q_ptr (reusing for attention scores)
        input_tensor,  # k_ptr (placeholder)
        input_tensor,  # v_ptr (placeholder)
        output_tensor, # o_ptr
        scale_val,     # scale
        B, H, N, K,    # dimensions
        # strides for input
        input_tensor.stride(0), input_tensor.stride(1), 
        input_tensor.stride(2), input_tensor.stride(3),
        # strides for k (same as input)
        input_tensor.stride(0), input_tensor.stride(1),
        input_tensor.stride(2), input_tensor.stride(3),
        # strides for v (same as input)
        input_tensor.stride(0), input_tensor.stride(1),
        input_tensor.stride(2), input_tensor.stride(3),
        # strides for output
        output_tensor.stride(0), output_tensor.stride(1),
        output_tensor.stride(2), output_tensor.stride(3),
        # block sizes (will be tuned by autotune)
        32, 64, 32,  # BLOCK_M, BLOCK_N, BLOCK_D
    )


def pattern(matmul_result):
    """
    Match the pattern: matmul -> scale division -> softmax
    
    This pattern matches the scaled softmax computation pattern.
    The scale value is extracted within replacement_args based on the input tensor.
    
    Returns:
        Tuple of (scaled_matmul, softmax_result) - the intermediate and final outputs
    """
    # The scaled softmax: matmul / sqrt(d) followed by softmax
    # The scale is typically sqrt(head_dim) or similar
    last_dim = matmul_result.shape[-1]
    scale = (last_dim ** 0.5)
    
    scaled = matmul_result / scale
    softmaxed = torch.nn.functional.softmax(scaled, dim=-1)
    return scaled, softmaxed


def replacement_args(matmul_result):
    """
    Extract arguments needed for the replacement function.
    """
    return (matmul_result,)


def replacement_func():
    """
    Returns the fused scale + softmax implementation.
    Uses Triton kernels for computation, only torch.empty_like for allocation.
    """
    def optimized_scale_softmax(matmul_out):
        """
        Optimized fused scale + softmax computation.
        The scale is computed as sqrt(last_dim) to match the attention scaling factor.
        """
        # Get dimensions
        shape = matmul_out.shape
        last_dim = shape[-1]
        
        # Compute the scale factor: typically sqrt(head_dim)
        scale_val = last_dim ** 0.5
        
        # Create output tensor
        output = torch.empty_like(matmul_out)
        
        # Call the Triton kernel wrapper
        _triton_fused_attention(matmul_out, output, scale_val, 
                                shape[0], shape[1], shape[2], last_dim)
        
        return output
    
    return optimized_scale_softmax