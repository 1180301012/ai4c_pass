import torch
import triton
import triton.language as tl


# Tuned block sizes for this specific shape [1, 16, 257, 80]
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
):
    """Fused attention kernel: Q @ K^T -> softmax -> @ V"""
    
    # Program ID for block
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for Q [B, H, M, K]
    q_ptrs = (
        q_ptr + pid_b * stride_qb + pid_h * stride_qh +
        offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    )
    
    # Load Q block
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    
    # Compute Q @ K^T for this block
    scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K blocks
    for k_start in range(0, K, BLOCK_K):
        # Load K block [B, H, N, BLOCK_K]
        k_ptrs = (
            k_ptr + pid_b * stride_kb + pid_h * stride_kh +
            offs_n[None, :] * stride_kn + (k_start + offs_k)[:, None] * stride_kk
        )
        k = tl.load(
            k_ptrs, 
            mask=(offs_k[:, None] < K - k_start) & (offs_n[None, :] < N), 
            other=0.0
        )
        
        # Compute partial score: Q @ K^T
        score_block = tl.dot(tl.trans(q), k)
        scores += score_block.to(tl.float32)
    
    # Softmax (row-wise over N dimension)
    # Subtract max for numerical stability
    max_scores = tl.max(scores, axis=1, keepdim=True)
    scores_minus_max = scores - max_scores
    exp_scores = tl.exp(scores_minus_max)
    sum_exp = tl.sum(exp_scores, axis=1, keepdim=True)
    softmax_scores = exp_scores / sum_exp
    
    # Convert to output dtype (bfloat16)
    softmax_scores_bf16 = softmax_scores.to(tl.bfloat16)
    
    # Compute softmax @ V
    # V is [B, H, N, K]
    outputs = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    
    for n_start in range(0, N, BLOCK_N):
        # Load V block [B, H, BLOCK_N, K]
        v_ptrs = (
            v_ptr + pid_b * stride_vb + pid_h * stride_vh +
            (n_start + offs_n)[None, :] * stride_vm + offs_k[:, None] * stride_vk
        )
        v = tl.load(
            v_ptrs,
            mask=(offs_n[None, :] < N - n_start) & (offs_k[:, None] < K),
            other=0.0
        )
        
        # Partial matmul: softmax_scores[:, BLOCK_N] @ V[BLOCK_N, K]
        output_block = tl.dot(softmax_scores_bf16[:, n_start:n_start+BLOCK_N].to(tl.float32), v)
        outputs += output_block
    
    # Store output [B, H, M, K]
    out_ptrs = (
        out_ptr + pid_b * stride_ob + pid_h * stride_oh +
        offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    )
    tl.store(out_ptrs, outputs.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


def pattern(in_0, in_1, in_2):
    """
    Match the fused attention pattern:
    matmul(in_0, in_1) * 1.0 -> softmax -> dropout(p=0) -> type conversion -> matmul with in_2
    """
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul * 1.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3 = tmp_2.to(torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    to = tmp_4.to(torch.bfloat16)
    matmul_1 = torch.matmul(to, in_2)
    return to, matmul_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_attention_wrapper(q, k, v):
    """Wrapper function for the fused attention kernel"""
    B, H, M, K = q.shape  # [1, 16, 257, 80]
    _, _, N, _ = k.shape  # [1, 16, 80, 257] -> N=257
    # V has shape [1, 16, 257, 80] -> [B, H, N, K]
    
    # Output: [B, H, M, K]
    out = torch.empty((B, H, M, K), dtype=torch.bfloat16, device=q.device)
    
    # Grid with 3D launch - (B, H, num_M_blocks)
    grid = (B, H, (M + 64 - 1) // 64)  # Use 64 as default BLOCK_M for grid calculation
    
    fused_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, M, N, K,
    )
    
    return out


def replacement_func():
    return fused_attention_wrapper