import torch
import triton
import triton.language as tl


@triton.jit
def softmax_matmul_kernel(
    attn_ptr, v_ptr, out_ptr,
    # Strides
    attn_batch_stride, attn_row_stride, attn_col_stride,
    v_batch_stride, v_row_stride, v_col_stride,
    out_batch_stride, out_row_stride, out_col_stride,
    # Shapes
    B: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute: out[b,k,n] = softmax(attn[b,n,:]) @ v[b,:,k]
    Equivalent to: softmax(attn) @ v, then transpose(-1,-2)
    
    Grid: (B, K, N // BLOCK_SIZE)
    """
    b = tl.program_id(0)
    k = tl.program_id(1)
    n_block = tl.program_id(2)
    
    n_offsets = n_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_mask = n_offsets < N
    
    # Load attention row block: attn[b, n_offsets, :]
    # Shape: [BLOCK_SIZE, N]
    attn_offsets = (
        b * attn_batch_stride + 
        n_offsets[:, None] * attn_row_stride + 
        tl.arange(0, N)[None, :] * attn_col_stride
    )
    attn_block = tl.load(attn_ptr + attn_offsets, mask=n_mask[:, None] & (tl.arange(0, N)[None, :] < N), other=float('-inf'))
    
    # Compute softmax along last dimension for each row
    attn_max = tl.max(attn_block, axis=1, keep_dims=True)
    attn_stable = attn_block - attn_max
    exp_block = tl.exp(attn_stable)
    exp_sum = tl.sum(exp_block, axis=1, keep_dims=True)
    softmax_block = exp_block / exp_sum
    
    # Load v column: v[b, :, k] -> shape [N]
    v_offsets = (
        b * v_batch_stride + 
        tl.arange(0, N) * v_row_stride + 
        k * v_col_stride
    )
    v_col = tl.load(v_ptr + v_offsets, mask=tl.arange(0, N) < N, other=0.0)
    
    # Multiply: softmax[b,n,:] @ v[b,:,k] = sum_m(softmax[b,n,m] * v[b,m,k])
    # For each n in the block, compute the dot product
    # Result: [BLOCK_SIZE]
    result = tl.sum(softmax_block * v_col[None, :], axis=1)
    
    # Store output[b, k, n_offsets]
    out_offsets = (
        b * out_batch_stride + 
        k * out_row_stride + 
        n_offsets * out_col_stride
    )
    tl.store(out_ptr + out_offsets, result, mask=n_mask)


@torch.fx.wrap
def fused_kernel_wrapper(attn, v):
    """
    Fused softmax + matmul + transpose kernel.
    
    Args:
        attn: [B, N, N] - attention scores
        v: [B, N, K] - values
        
    Returns:
        [B, K, N] - output after softmax @ v and transpose
    """
    B, N, _ = attn.shape
    _, _, K = v.shape
    
    # Output: [B, K, N]
    out = torch.empty((B, K, N), dtype=attn.dtype, device=attn.device)
    
    # Block size for N dimension
    BLOCK_SIZE = 64 if N >= 64 else N
    
    # Grid: (B, K, ceil(N/BLOCK_SIZE))
    grid = (B, K, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    softmax_matmul_kernel[grid](
        attn, v, out,
        attn.stride(0), attn.stride(1), attn.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B=B, N=N, K=K, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_4):
    """
    Match the pattern:
    tmp_13 = in_0.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return (tmp_15,)
    
    Note: in_0 is the output of the attention score computation (in_0 + attention_bias)
    """
    tmp_13 = in_0.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, in_4):
    return (in_0, in_4)


def replacement_func():
    return fused_kernel_wrapper