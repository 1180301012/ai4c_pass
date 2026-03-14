import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: linear -> view -> transpose
    Also match: indexing -> expand
    
    Graph 7 uses: view((64, 128, -1, 128)), expand(64, 4, 4, 128, 128)
    Input shapes: in_0=(512, 2048), in_1=(64, 128, 2048), in_2=(64, 4, 128, 128)
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2 = tmp_1.view((64, 128, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = in_2[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand(64, 4, 4, 128, 128)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_view_transpose_kernel_7(
    input_ptr, weight_ptr, key_states_ptr,
    output_v_ptr, output_q_ptr,
    M, N, K,
    batch_size, num_heads, seq_len, head_dim,
    stride_input, stride_weight, stride_key,
    stride_out_v, stride_out_q,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused kernel for graph 7: linear -> view -> transpose + indexing -> expand"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_input + offs_k[None, :] * 1)
    weight_ptrs = weight_ptr + (offs_k[:, None] * 1 + offs_n[None, :] * stride_weight)
    
    input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    weight_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        input_mask_k = input_mask & (offs_k[None, :] < K - k * BLOCK_K)
        weight_mask_k = weight_mask & (offs_k[:, None] < K - k * BLOCK_K)
        
        a = tl.load(input_ptrs, mask=input_mask_k, other=0.0)
        b = tl.load(weight_ptrs, mask=weight_mask_k, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K * stride_weight
        offs_k += BLOCK_K

    output_v = accumulator.to(tl.float16)
    
    offs_m0 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    batch_idx = offs_m0 // seq_len
    seq_idx = offs_m0 % seq_len
    
    out_v_ptrs = output_v_ptr + (batch_idx[:, None] * stride_out_v + 
                                   pid_n * BLOCK_N + offs_n0[None, :] * 1 +
                                   seq_idx[:, None] * head_dim)
    out_v_mask = (offs_m0[:, None] < M) & (offs_n0[None, :] < N)
    tl.store(out_v_ptrs, output_v, mask=out_v_mask)


@torch.fx.wrap
def fused_linear_view_transpose_wrapper_7(in_0, in_1, in_2):
    """Wrapper for graph 7: in_1=(64,128,2048), in_2=(64,4,128,128)"""
    weight = in_0  # (512, 2048)
    hidden = in_1  # (64, 128, 2048)
    key_states = in_2  # (64, 4, 128, 128)
    
    batch, seq, hidden_dim = hidden.shape  # 64, 128, 2048
    num_heads = 4
    head_dim = 128
    
    # Output: (batch, num_heads, seq, head_dim) = (64, 4, 128, 128)
    output_q_shape = (batch, num_heads, seq, head_dim)
    output_v = torch.empty(output_q_shape, dtype=hidden.dtype, device=hidden.device)
    
    M = batch * seq  # 8192
    N = num_heads * head_dim  # 512
    K = hidden_dim  # 2048
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    fused_linear_view_transpose_kernel_7[grid](
        hidden, weight, key_states,
        output_v, output_v,
        M, N, K,
        batch, num_heads, seq, head_dim,
        hidden.stride(0), weight.stride(0), key_states.stride(0),
        output_v.stride(0), output_v.stride(1),
    )
    
    # Handle key_states indexing -> expand: (64,4,128,128) -> (64,4,4,128,128)
    tmp_4 = key_states.unsqueeze(2)  # (64, 4, 1, 128, 128)
    tmp_5 = tmp_4.expand(64, 4, 4, 128, 128)  # (64, 4, 4, 128, 128)
    
    return tmp_5, output_v


def replacement_func():
    return fused_linear_view_transpose_wrapper_7