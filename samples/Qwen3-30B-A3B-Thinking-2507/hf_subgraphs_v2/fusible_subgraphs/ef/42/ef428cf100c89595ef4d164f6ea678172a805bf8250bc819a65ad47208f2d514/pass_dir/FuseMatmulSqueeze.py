import torch
import triton
import triton.language as tl

def pattern(a, b):
    matmul = torch.matmul(a, b)
    return matmul.squeeze(1)

def replacement_args(a, b):
    return (a, b)

@triton.jit
def matmul_squeeze_kernel(a_ptr, b_ptr, c_ptr, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N):
    pid = tl.program_id(0)
    n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = n < N

    a = tl.load(a_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = k + tl.arange(0, BLOCK_SIZE_K) < K
        a_seg = tl.load(a_ptr + k + tl.arange(0, BLOCK_SIZE_K), mask=k_mask, other=0.0)
        b_seg = tl.load(
            b_ptr + k * N + n,
            mask=(k_mask[:, None] & mask_n[None, :]),
            other=0.0
        )
        acc += a_seg * b_seg

    tl.store(c_ptr + n, acc, mask=mask_n)

@torch.fx.wrap
def triton_matmul_squeeze(a, b):
    batch = a.shape[0]
    m = a.shape[1]
    k = a.shape[2]
    n = b.shape[2]

    output = torch.empty((batch, n), dtype=a.dtype, device=a.device)
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 64
    grid = (n // BLOCK_SIZE_N,)

    matmul_squeeze_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=output.data_ptr(),
        K=k,
        N=n,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return output

def replacement_func():
    return triton_matmul_squeeze