import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, n, k, BLOCK_N: tl.constexpr = 32, BLOCK_K: tl.constexpr = 32):
    row = tl.program_id(0)
    acc = tl.zeros((1,), dtype=tl.float32)
    for start_k in range(0, k, BLOCK_K):
        end_k = min(start_k + BLOCK_K, k)
        a = tl.load(A + row * k + start_k, 
                   mask=start_k + tl.arange(0, BLOCK_K) < end_k,
                   other=0.0)
        b = tl.load(B + start_k, 
                   mask=start_k + tl.arange(0, BLOCK_K) < end_k,
                   other=0.0)
        acc += a * b
    tl.store(C + row, acc)

@torch.fx.wrap
def optimized_matmul(a, b):
    n = a.shape[0]
    k = a.shape[1]
    c = torch.empty((n, 1), dtype=a.dtype)
    num_programs = n
    BLOCK_N = 32
    BLOCK_K = 32
    matmul_kernel[(num_programs,)](
        a, b, c, n, k, BLOCK_N, BLOCK_K
    )
    return c

def pattern(a, b):
    return torch.matmul(a, b)

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    return optimized_matmul