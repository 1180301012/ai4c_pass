import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Configs optimized for YOLO-type: M=64, K=400, N=400
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        # Configs for large M, N with small K (e.g., S-ViPNAS K=48)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_stages=4, num_warps=4),
        # Configs for GCNet: M=512, K=4096, N=1
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 64},  num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 64},  num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 256}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Batched matmul: C[b] = A[b] @ B[b]
    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), BATCH)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers into A and B for this batch
    A_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Write output
    C_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def triton_batched_matmul(in_0, in_1):
    """
    Compute in_1 @ in_0 using Triton.
    in_1: [..., M, K]  (left operand)
    in_0: [..., K, N]  (right operand)
    Returns: [..., M, N]
    """
    ndim = in_1.ndim

    if ndim == 4:
        B, H, M, K = in_1.shape
        _B, _H, K2, N = in_0.shape
        BATCH = B * H
        # Flatten batch dimensions for the kernel
        a = in_1.reshape(BATCH, M, K).contiguous()
        b = in_0.reshape(BATCH, K2, N).contiguous()
    elif ndim == 3:
        BATCH, M, K = in_1.shape
        _BATCH, K2, N = in_0.shape
        a = in_1.contiguous()
        b = in_0.contiguous()
        B, H = BATCH, 1
    else:
        # Treat as 3D by inserting batch dim of 1
        BATCH = 1
        a = in_1.unsqueeze(0).contiguous()
        b = in_0.unsqueeze(0).contiguous()
        _B, M, K = a.shape
        _B2, K2, N = b.shape
        B, H = 1, 1

    c = torch.empty((BATCH, M, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        BATCH,
    )

    batched_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )

    # Restore original batch shape
    if ndim == 4:
        return c.reshape(B, H, M, N)
    return c


def pattern(in_0, in_1):
    """
    Match torch.matmul(in_1, in_0) graphs (GCNet, S-ViPNAS).
    Try using @ operator in case getattr resolves to same node.
    """
    return in_1 @ in_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_batched_matmul