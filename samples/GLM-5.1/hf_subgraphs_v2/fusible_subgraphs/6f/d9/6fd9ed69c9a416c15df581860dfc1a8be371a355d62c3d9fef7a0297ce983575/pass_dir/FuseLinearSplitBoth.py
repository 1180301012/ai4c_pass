import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5 = linear[:, :256]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = linear[:, -256:]
    tmp_8 = tmp_7.view(-1, 256)
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = linear_1[..., :256]
    tmp_12 = linear_1[..., -256:]
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_11, tmp_12, tmp_8, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_split_kernel(
    x_ptr, W_ptr, b_ptr,
    out_first_ptr, out_second_ptr,
    M, K, N, N_half,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_of_m, stride_of_n,
    stride_os_m, stride_os_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Determine which half this tile belongs to
    n_start = pid_n * BLOCK_N
    is_first_half = n_start < N_half

    # Accumulator in fp32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main matmul loop
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W^T: [BLOCK_K, BLOCK_N]
        # W has shape [N, K], W[n, k] at W_ptr + n*stride_wn + k*stride_wk
        # We load W^T[k, n] = W[n, k] as [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        W_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        W = tl.load(W_ptrs, mask=W_mask, other=0.0)

        accumulator += tl.dot(x, W)

    # Add bias
    b_ptrs = b_ptr + offs_n
    b_mask = offs_n < N
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0)
    accumulator += bias[None, :]

    # Store results to appropriate output buffer
    mask_m = offs_m < M

    if is_first_half:
        # First half columns [0, N_half)
        mask = mask_m[:, None] & (offs_n[None, :] < N_half)
        out_ptrs = out_first_ptr + offs_m[:, None] * stride_of_m + offs_n[None, :] * stride_of_n
        tl.store(out_ptrs, accumulator, mask=mask)
    else:
        # Second half columns [N_half, N), stored with local column offset
        local_offs_n = offs_n - N_half
        mask = mask_m[:, None] & (local_offs_n[None, :] < N_half) & (offs_n[None, :] < N)
        out_ptrs = out_second_ptr + offs_m[:, None] * stride_os_m + local_offs_n[None, :] * stride_os_n
        tl.store(out_ptrs, accumulator, mask=mask)


@torch.fx.wrap
def fused_linear_split_both(in_0, in_1, in_2, in_3, in_4, in_5):
    M = 300
    K = 256
    N = 512
    N_half = 256

    # First linear outputs
    # out_13: first half of linear(in_5, in_1, in_0), shape [300, 1, 256] (unsqueezed)
    # out_8: second half of linear(in_5, in_1, in_0), shape [300, 256]
    out_13 = torch.empty(M, 1, N_half, dtype=in_5.dtype, device=in_5.device)
    out_8 = torch.empty(M, N_half, dtype=in_5.dtype, device=in_5.device)

    # Grid for autotune - lambda that uses META block sizes
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    # Strides for first linear (all contiguous tensors)
    # in_5 [300, 256]: stride(0)=256, stride(1)=1
    # in_1 [512, 256]: stride(0)=256, stride(1)=1
    # out_13 [300, 1, 256]: stride(0)=256, stride(2)=1
    # out_8 [300, 256]: stride(0)=256, stride(1)=1

    fused_linear_split_kernel[grid](
        x_ptr=in_5, W_ptr=in_1, b_ptr=in_0,
        out_first_ptr=out_13, out_second_ptr=out_8,
        M=M, K=K, N=N, N_half=N_half,
        stride_xm=K, stride_xk=1,
        stride_wn=K, stride_wk=1,
        stride_of_m=N_half, stride_of_n=1,
        stride_os_m=N_half, stride_os_n=1,
    )

    # Second linear outputs
    # out_11: first half of linear(reshaped_in_4, in_3, in_2), shape [300, 1, 256]
    # out_12: second half, shape [300, 1, 256]
    out_11 = torch.empty(M, 1, N_half, dtype=in_4.dtype, device=in_4.device)
    out_12 = torch.empty(M, 1, N_half, dtype=in_4.dtype, device=in_4.device)

    # For in_4 [1, 150, 1, 512] reshaped to [300, 1, 256]
    # Contiguous reshape: stride(0)=256, stride(2)=1 for the reshaped view
    stride_xm_2 = N_half  # 256
    stride_xk_2 = 1

    # in_3 [512, 256]: stride(0)=256, stride(1)=1
    # out_11 [300, 1, 256]: stride(0)=256, stride(2)=1
    # out_12 [300, 1, 256]: stride(0)=256, stride(2)=1

    fused_linear_split_kernel[grid](
        x_ptr=in_4, W_ptr=in_3, b_ptr=in_2,
        out_first_ptr=out_11, out_second_ptr=out_12,
        M=M, K=K, N=N, N_half=N_half,
        stride_xm=stride_xm_2, stride_xk=stride_xk_2,
        stride_wn=K, stride_wk=1,
        stride_of_m=N_half, stride_of_n=1,
        stride_os_m=N_half, stride_os_n=1,
    )

    return (out_11, out_12, out_8, out_13)


def replacement_func():
    return fused_linear_split_both