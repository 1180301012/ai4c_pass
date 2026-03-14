import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match just linear + reshape + permute, skip the unbind complication
    """
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_qkv_matmul_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized matmul kernel for QKV projection
    Input: [M, K] where M = batch * seq_len
    Weight: [N, K] where N = 3 * num_heads * head_dim
    Output: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute matmul in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load input block
        input_ptrs = input_ptr + (offs_m[:, None] * stride_im + (k + offs_k[None, :]) * stride_ik)
        input_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight block
        weight_ptrs = weight_ptr + (offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk)
        weight_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        weight_block = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_block, weight_block)
    
    # Store output
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)

@torch.fx.wrap
def fused_qkv_optimized(weight, input):
    """
    Optimized QKV projection - replaces linear+reshape+permute
    """
    batch_size, seq_len, in_features = input.shape
    out_features, _ = weight.shape
    
    # For convit_small: 9 heads, 48 head_dim
    head_dim = 48
    num_heads = out_features // (3 * head_dim)
    
    # Flatten input for matmul
    input_2d = input.reshape(-1, in_features)
    M, K = input_2d.shape
    N = out_features
    
    # Allocate output for matmul
    output_2d = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    # Launch matmul kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    fused_qkv_matmul_kernel[grid](
        input_2d, weight, output_2d,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(0), weight.stride(1),
        output_2d.stride(0), output_2d.stride(1),
    )
    
    # Reshape and permute to get final output
    output_reshaped = output_2d.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    output_permuted = output_reshaped.permute(2, 0, 3, 1, 4)
    
    return output_permuted

def replacement_func():
    return fused_qkv_optimized