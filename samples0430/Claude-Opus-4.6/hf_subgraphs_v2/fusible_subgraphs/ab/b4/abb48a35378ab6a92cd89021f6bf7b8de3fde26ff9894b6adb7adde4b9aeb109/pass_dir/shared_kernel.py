import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    acc_out = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, acc_out, mask=c_mask)


@torch.fx.wrap
def fused_linear_qkv_dispatch(weight, x):
    M = x.shape[1]  # 197
    K = x.shape[2]  # hidden_dim
    N = weight.shape[0]  # 3 * num_heads * 48
    HEAD_DIM = 48
    num_heads = N // (3 * HEAD_DIM)
    dtype = x.dtype

    # Allocate output for matmul result: [1, M, N]
    c = torch.empty((1, M, N), dtype=dtype, device=x.device)

    # Launch matmul kernel
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    grid_matmul = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    matmul_kernel[grid_matmul](
        x, weight, c,
        M, N, K,
        x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        c.stride(1), c.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Use view operations (zero-cost) to produce the same output as
    # reshape(1, 197, 3, H, 48) -> permute(2, 0, 3, 1, 4) -> unbind(0)
    reshaped = c.reshape(1, M, 3, num_heads, HEAD_DIM)
    permuted = reshaped.permute(2, 0, 3, 1, 4)
    q = permuted[0]
    k = permuted[1]
    v = permuted[2]

    return (q, k, v)