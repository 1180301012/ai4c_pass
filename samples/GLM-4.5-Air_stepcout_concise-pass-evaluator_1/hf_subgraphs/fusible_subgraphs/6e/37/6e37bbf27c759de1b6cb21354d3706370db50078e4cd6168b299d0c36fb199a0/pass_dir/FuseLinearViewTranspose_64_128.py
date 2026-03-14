import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_view_transpose_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wm, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused kernel for linear + view + transpose."""
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

    # For input: [M, K] -> load [BLOCK_M, BLOCK_K] blocks
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    
    # For weight: [N, K] in row-major. Need to load as [K, N] for input @ weight.T
    # Load BLOCK_K rows and BLOCK_N columns: [BLOCK_K, BLOCK_N]
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wm)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(input_ptrs, mask=mask, other=0.0)
        
        mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(weight_ptrs, mask=mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        input_ptrs += BLOCK_K * stride_ik
        weight_ptrs += BLOCK_K * stride_wm
        offs_k += BLOCK_K

    output = accumulator.to(tl.bfloat16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output, mask=mask)


def fused_linear_view_transpose(x, weight, view_shape, transpose_dim1, transpose_dim2):
    """Fused linear + view + transpose operation."""
    batch, seq, hidden = x.shape
    out_features = weight.shape[0]
    
    M = batch * seq
    K = hidden
    N = out_features
    
    x_2d = x.view(M, K)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
    
    fused_linear_view_transpose_kernel[grid](
        x_2d, weight, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    
    out = out.view(view_shape)
    out = out.transpose(transpose_dim1, transpose_dim2)
    
    return out


@torch.fx.wrap
def fused_linear_view_transpose_wrapper(x, weight, view_shape, transpose_dim1, transpose_dim2):
    return fused_linear_view_transpose(x, weight, view_shape, transpose_dim1, transpose_dim2)


def pattern(in_0, in_2):
    """Match linear -> view((64, 128, -1, 128)) -> transpose"""
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_2, tmp_0, None)
    tmp_2 = tmp_1.view((64, 128, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3


def replacement_args(in_0, in_2):
    # View shape (64, 128, -1, 128) for hidden_states [64, 128, 2048]
    batch, seq, hidden = in_2.shape
    out_features = in_0.shape[0]
    head_dim = 128
    num_heads = out_features // head_dim
    view_shape = (batch, seq, num_heads, head_dim)
    return (in_2, in_0, view_shape, 1, 2)


def replacement_func():
    return fused_linear_view_transpose_wrapper