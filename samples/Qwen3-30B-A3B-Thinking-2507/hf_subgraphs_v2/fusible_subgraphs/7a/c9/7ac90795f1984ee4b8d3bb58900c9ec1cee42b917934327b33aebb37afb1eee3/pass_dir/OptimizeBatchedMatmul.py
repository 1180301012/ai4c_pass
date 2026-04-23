import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    matmul = torch.matmul(in_1, in_0)
    return matmul

def replacement_args(in_1, in_0):
    return (in_1, in_0)

@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, k, n,
    stride_A_batch, stride_A_row, stride_A_col,
    stride_B_batch, stride_B_row, stride_B_col,
    stride_C_batch, stride_C_row, stride_C_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    block_A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    block_B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    
    pid = tl.program_id(0)
    batch_id = pid // (m // BLOCK_SIZE_M)
    m_block_id = pid % (m // BLOCK_SIZE_M)
    
    row_start = m_block_id * BLOCK_SIZE_M
    col_start = 0
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_start in range(0, k, BLOCK_SIZE_K):
        A = tl.load(
            A_ptr + batch_id * stride_A_batch + row_start * stride_A_row + k_start * stride_A_col,
            shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            mask=(row_start + tl.arange(0, BLOCK_SIZE_M) < m)[:, None] & (k_start + tl.arange(0, BLOCK_SIZE_K) < k)[None, :],
            other=0.0
        )
        tl.store(block_A, A)
        
        B = tl.load(
            B_ptr + batch_id * stride_B_batch + k_start * stride_B_row + col_start * stride_B_col,
            shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            mask=(k_start + tl.arange(0, BLOCK_SIZE_K) < k)[:, None] & (col_start + tl.arange(0, BLOCK_SIZE_N) < n)[None, :],
            other=0.0
        )
        tl.store(block_B, B)
        
        tl.sync()
        
        block_A = tl.load(block_A)
        block_B = tl.load(block_B)
        acc += tl.dot(block_A, block_B)
    
    C = C_ptr + batch_id * stride_C_batch + row_start * stride_C_row + col_start * stride_C_col
    tl.store(
        C,
        acc,
        mask=(row_start + tl.arange(0, BLOCK_SIZE_M) < m)[:, None] & (col_start + tl.arange(0, BLOCK_SIZE_N) < n)[None, :]
    )

@torch.fx.wrap
def batched_matmul(in_1, in_0):
    stride_A_batch = in_1.stride(0)
    stride_A_row = in_1.stride(1)
    stride_A_col = in_1.stride(2)
    
    stride_B_batch = in_0.stride(0)
    stride_B_row = in_0.stride(1)
    stride_B_col = in_0.stride(2)
    
    batch_size = in_1.shape[0]
    m = in_1.shape[1]
    k = in_1.shape[2]
    n = in_0.shape[2]
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 8
    BLOCK_SIZE_N = 1
    
    m_blocks = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (batch_size * m_blocks,)
    
    out = torch.empty((batch_size, m, n), dtype=in_1.dtype, device=in_1.device)
    
    stride_C_batch = out.stride(0)
    stride_C_row = out.stride(1)
    stride_C_col = out.stride(2)
    
    batched_matmul_kernel[grid](
        in_1,
        in_0,
        out,
        batch_size,
        m,
        k,
        n,
        stride_A_batch, stride_A_row, stride_A_col,
        stride_B_batch, stride_B_row, stride_B_col,
        stride_C_batch, stride_C_row, stride_C_col,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return batched_matmul