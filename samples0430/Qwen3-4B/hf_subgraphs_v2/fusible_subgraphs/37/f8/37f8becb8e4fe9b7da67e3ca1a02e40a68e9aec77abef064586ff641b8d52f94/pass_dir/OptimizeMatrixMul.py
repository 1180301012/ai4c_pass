import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    tmp_3 = in_1.to(device='cuda')
    tmp_4 = in_0.to(device='cuda')
    return (tmp_4, tmp_3, matmul)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

class MatrixMulConfig:
    BLOCK_SIZE = 128

@triton.jit
def kernel(in_2_ptr, in_3_ptr, out_ptr, n_elements, m_elements, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offset = row * BLOCK_SIZE
    row_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float16)
    
    # Load row from in_2 (shape [n, m])
    for j in range(BLOCK_SIZE):
        idx = offset + j
        if idx < n_elements * m_elements:
            row_vec[j] = tl.load(in_2_ptr + idx, mask=idx < n_elements * m_elements)
    
    # Process with in_3 (column vector)
    result = tl.zeros(1, dtype=tl.float16)
    for j in range(BLOCK_SIZE):
        if j < m_elements:
            val = tl.load(in_3_ptr + j)
            result += row_vec[j] * val
    
    tl.store(out_ptr + row, result)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    n = in_2.shape[0]
    m = in_2.shape[1]
    out = torch.empty((n, 1), device='cuda', dtype=in_2.dtype)
    
    # Handle the matrix multiplication
    kernel[
        (tl.cdiv(n, 128),)
    ](
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=n,
        m_elements=m,
        BLOCK_SIZE=128,
    )
    
    # Handle device transfers
    tmp_0 = in_0.to(device='cuda')
    tmp_1 = in_1.to(device='cuda')
    return (tmp_0, tmp_1, out)

def replacement_func():
    return kernel_wrapper
    def kernel(in_2_ptr, in_3_ptr, out_ptr, n_elements, m_elements, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offset = row * BLOCK_SIZE
        row_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float16)
        
        # Load row from in_2 (shape [n, m])
        for j in range(BLOCK_SIZE):
            idx = offset + j
            if idx < n_elements * m_elements:
                row_vec[j] = tl.load(in_2_ptr + idx, mask=idx < n_elements * m_elements)
        
        # Process with in_3 (column vector)
        result = tl.zeros(1, dtype=tl.float16)
        for j in range(BLOCK_SIZE):
            if j < m_elements:
                val = tl.load(in_3_ptr + j)
                result += row_vec[j] * val
        
        tl.store(out_ptr + row, result)

    @torch.fx.wrap
    def wrapper(in_0, in_1, in_2, in_3):
        n = in_2.shape[0]
        m = in_2.shape[1]
        out = torch.empty((n, 1), device='cuda', dtype=in_2.dtype)
        
        # Handle the matrix multiplication
        MatrixMulKernel.kernel[
            (tl.cdiv(n, MatrixMulConfig.BLOCK_SIZE),)
        ](
            in_2_ptr=in_2,
            in_3_ptr=in_3,
            out_ptr=out,
            n_elements=n,
            m_elements=m,
            BLOCK_SIZE=MatrixMulConfig.BLOCK_SIZE,
        )
        
        # Handle device transfers
        tmp_0 = in_0.to(device='cuda')
        tmp_1 = in_1.to(device='cuda')
        return (tmp_0, tmp_1, out)

def replacement_func():
    return kernel_wrapper