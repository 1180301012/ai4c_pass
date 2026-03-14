import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match: linear -> view -> transpose
    Graph 5 uses: view((4, 512, -1, 128))
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2 = tmp_1.view((4, 512, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_view_transpose_kernel_5(
    input_ptr, weight_ptr,
    output_ptr,
    M, N, K,
    stride_input, stride_weight, stride_output,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for graph 5: linear -> view -> transpose
    
    Compute: output = input @ weight.T
    - input shape: (M, K) = (batch*seq, hidden_dim)
    - weight shape: (N, K) = (num_heads*head_dim, hidden_dim)
    - output shape: (M, N) = (batch*seq, num_heads*head_dim)
    
    Since weight is stored as (N, K) in PyTorch row-major:
    - weight[n, k] = weight_ptr[n * K + k]
    - For weight.T[k, n] = weight[n, k], we need: weight_ptr[n * K + k]
    """
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

    # input: (M, K) matrix, row-major
    # Load: input[offs_m, offs_k] = input_ptr + offs_m * stride_input + offs_k
    input_ptrs = input_ptr + (offs_m[:, None] * stride_input + offs_k[None, :])
    
    # weight: stored as (N, K) in PyTorch row-major
    # For weight.T[k, n], we need weight[n, k] = weight_ptr + n * K + k
    # So we access: weight_ptr + offs_n * K + offs_k
    weight_ptrs = weight_ptr + (offs_n[:, None] * K + offs_k[None, :])
    
    input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        # a: (BLOCK_M, BLOCK_K), b: (BLOCK_N, BLOCK_K)
        # Need to transpose b to (BLOCK_K, BLOCK_N) for matmul
        accumulator += tl.dot(a, tl.trans(b))
        
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K
        offs_k += BLOCK_K
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

    output = accumulator.to(tl.float16)
    
    offs_m0 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_ptrs = output_ptr + (offs_m0[:, None] * stride_output + offs_n0[None, :])
    out_mask = (offs_m0[:, None] < M) & (offs_n0[None, :] < N)
    tl.store(out_ptrs, output, mask=out_mask)


@torch.fx.wrap
def fused_linear_view_transpose_wrapper_5(in_0, in_1):
    """Wrapper for graph 5: in_1=(4,512,2048)
    
    Returns tmp_3: result of linear->view->transpose, shape (4, 4, 512, 128)
    """
    weight = in_0  # (512, 2048)
    hidden = in_1  # (4, 512, 2048)
    
    batch, seq, hidden_dim = hidden.shape  # 4, 512, 2048
    num_heads = 4
    head_dim = 128
    
    # Flatten input for the kernel: (4, 512, 2048) -> (2048, 2048)
    hidden_flat = hidden.view(batch * seq, hidden_dim)  # (2048, 2048)
    
    M = batch * seq  # 2048
    N = num_heads * head_dim  # 512
    K = hidden_dim  # 2048
    
    output_shape = (M, N)  # (2048, 512)
    output = torch.empty(output_shape, dtype=hidden.dtype, device=hidden.device)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    fused_linear_view_transpose_kernel_5[grid](
        hidden_flat, weight,
        output,
        M, N, K,
        hidden_flat.stride(0), weight.stride(0), output.stride(0),
    )
    
    # Reshape output to (batch, num_heads, seq, head_dim) = (4, 4, 512, 128)
    tmp_3 = output.view(batch, num_heads, seq, head_dim)
    
    return tmp_3


def replacement_func():
    return fused_linear_view_transpose_wrapper_5