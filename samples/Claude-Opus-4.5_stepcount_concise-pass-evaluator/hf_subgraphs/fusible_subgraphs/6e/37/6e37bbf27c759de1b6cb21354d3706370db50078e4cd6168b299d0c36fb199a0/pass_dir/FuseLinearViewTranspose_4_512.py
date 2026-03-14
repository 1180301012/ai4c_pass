import torch
import triton
import triton.language as tl

# Pattern matching just linear + view + transpose for Graph 5
def pattern(in_0, in_2):
    tmp_1 = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = tmp_1.view((4, 512, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3

def replacement_args(in_0, in_2):
    return (in_0, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_transpose_kernel(
    hidden_ptr, weight_ptr, output_ptr,
    M, N, K,
    seq, num_heads, head_dim,
    hidden_stride_b, hidden_stride_s, hidden_stride_f,
    output_stride_b, output_stride_h, output_stride_s, output_stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    batch_idx = offs_m // seq
    seq_idx = offs_m % seq
    head_idx = offs_n // head_dim
    d_idx = offs_n % head_dim
    
    hidden_ptrs = hidden_ptr + batch_idx[:, None] * hidden_stride_b + seq_idx[:, None] * hidden_stride_s + offs_k[None, :] * hidden_stride_f
    weight_ptrs = weight_ptr + offs_n[:, None] * K + offs_k[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        h_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        
        h = tl.load(hidden_ptrs, mask=h_mask, other=0.0)
        w = tl.load(weight_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(h, tl.trans(w))
        
        hidden_ptrs += BLOCK_K * hidden_stride_f
        weight_ptrs += BLOCK_K
    
    acc = acc.to(tl.bfloat16)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    mask = m_mask[:, None] & n_mask[None, :]
    
    output_ptrs = output_ptr + batch_idx[:, None] * output_stride_b + head_idx[None, :] * output_stride_h + seq_idx[:, None] * output_stride_s + d_idx[None, :] * output_stride_d
    tl.store(output_ptrs, acc, mask=mask)


@torch.fx.wrap
def fused_linear_transpose_wrapper(in_0, in_2):
    batch, seq, in_features = in_2.shape
    out_features = in_0.shape[0]
    head_dim = 128
    num_heads = out_features // head_dim
    
    M = batch * seq
    N = out_features
    K = in_features
    
    output = torch.empty((batch, num_heads, seq, head_dim), dtype=in_2.dtype, device=in_2.device)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    fused_linear_transpose_kernel[grid](
        in_2, in_0, output,
        M, N, K,
        seq, num_heads, head_dim,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


def replacement_func():
    return fused_linear_transpose_wrapper