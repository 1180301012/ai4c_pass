import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return (tmp_1,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Ultra-minimal kernel: no stride args, computes strides from K/N
# Assumes contiguous tensor layout (valid for this specific problem)
# Fewer arguments = faster kernel launch
@triton.jit
def fused_matmul_squeeze_minimal_kernel(
    a_ptr, b_ptr, out_ptr,
    K, N,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Compute strides from shapes (contiguous layout)
    # A is [B, 1, K]: stride_ab = K, stride_ak = 1
    # B is [B, K, N]: stride_bb = K*N, stride_bk = N, stride_bn = 1
    # Out is [B, N]: stride_ob = N, stride_on = 1
    stride_ab = K
    stride_ak = 1
    stride_bb = K * N
    stride_bk = N
    stride_bn = 1
    stride_ob = N
    stride_on = 1

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(K):
        a_val = tl.load(a_ptr + pid_b * stride_ab + k * stride_ak).to(tl.float32)
        b_row = tl.load(
            b_ptr + pid_b * stride_bb + k * stride_bk + n_offsets * stride_bn,
            mask=n_mask, other=0.0
        ).to(tl.float32)
        acc += a_val * b_row

    tl.store(out_ptr + pid_b * stride_ob + n_offsets * stride_on, acc, mask=n_mask)


# Simple kernel with explicit stride args (for non-contiguous tensors)
@triton.jit
def fused_matmul_squeeze_simple_kernel(
    a_ptr, b_ptr, out_ptr,
    stride_ab, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_on,
    K, N,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(K):
        a_val = tl.load(a_ptr + pid_b * stride_ab + k * stride_ak).to(tl.float32)
        b_row = tl.load(
            b_ptr + pid_b * stride_bb + k * stride_bk + n_offsets * stride_bn,
            mask=n_mask, other=0.0
        ).to(tl.float32)
        acc += a_val * b_row

    tl.store(out_ptr + pid_b * stride_ob + n_offsets * stride_on, acc, mask=n_mask)


@torch.fx.wrap
def fused_matmul_squeeze_fn(a, b):
    B = a.shape[0]
    K = a.shape[2]
    N = b.shape[2]

    out = torch.empty((B, N), dtype=a.dtype, device=a.device)

    BLOCK_N = 16
    grid = (B, triton.cdiv(N, BLOCK_N))

    # Use minimal kernel (no stride args) for contiguous tensors
    # Check if tensors are contiguous
    a_contiguous = a.is_contiguous()
    b_contiguous = b.is_contiguous()
    
    if a_contiguous and b_contiguous:
        fused_matmul_squeeze_minimal_kernel[grid](
            a_ptr=a, b_ptr=b, out_ptr=out,
            K=K, N=N,
            BLOCK_N=BLOCK_N,
        )
    else:
        # Fallback to kernel with stride args for non-contiguous tensors
        fused_matmul_squeeze_simple_kernel[grid](
            a_ptr=a, b_ptr=b, out_ptr=out,
            stride_ab=a.stride(0), stride_ak=a.stride(2),
            stride_bb=b.stride(0), stride_bk=b.stride(1), stride_bn=b.stride(2),
            stride_ob=out.stride(0), stride_on=out.stride(1),
            K=K, N=N,
            BLOCK_N=BLOCK_N,
        )

    return out


def replacement_func():
    return fused_matmul_squeeze_fn