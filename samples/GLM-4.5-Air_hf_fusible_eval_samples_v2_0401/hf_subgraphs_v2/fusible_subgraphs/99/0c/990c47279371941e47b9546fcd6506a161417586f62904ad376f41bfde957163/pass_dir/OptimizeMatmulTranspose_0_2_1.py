import torch
import triton
import triton.language as tl

def pattern(tmp_1, in_1):
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3

def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1)

@triton.jit
def optimized_matmul_transpose_kernel(a_ptr, b_ptr, output_ptr, 
                                    a_stride_0, a_stride_1, a_stride_2,
                                    b_stride_0, b_stride_1, b_stride_2,
                                    output_stride_0, output_stride_1, output_stride_2,
                                    batch_size, m, n, k,
                                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute memory offsets
    batch_offset = pid_batch * a_stride_0
    
    # Each program processes a tile of size BLOCK_SIZE_M x BLOCK_SIZE_N
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    m_mask = m_offsets < m
    k_mask = k_offsets < k
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Loop over k dimension (the common dimension)
    for k_pos in range(0, n, BLOCK_SIZE_N):
        k_end = min(k_pos + BLOCK_SIZE_N, n)
        
        # Load block from A: [batch, m, k_pos:k_end] → need [m, blocked_k]
        a_ptrs = (a_ptr + batch_offset + 
                 m_offsets[:, None] * a_stride_1 + 
                 k_pos * a_stride_2)
        a_block = tl.load(a_ptrs, 
                         mask=m_mask[:, None] & (tl.arange(k_pos, k_end) < n)[:, None], 
                         other=0.0)
        
        # Load block from B: [batch, n, k] → but we need B_transposed, so load [blocked_k, n]
        # We transpose B on the fly: original shape [batch, n, k] → we want [batch, k, n] for this operation
        b_ptrs = (b_ptr + batch_offset + 
                 k_offsets[:, None] * b_stride_1 + 
                 k_pos * b_stride_2)
        b_block = tl.load(b_ptrs, 
                         mask=k_mask[:, None] & (tl.arange(k_pos, k_end) < n)[:, None], 
                         other=0.0)
        b_block = b_block.T  # Transpose to [blocked_n, blocked_k]
        
        # Matrix multiply: a_block [m, blocked_k] @ b_block [blocked_k, n] → accumulator [m, n]
        accumulator += tl.dot(a_block, b_block, out_type=tl.float32)
    
    # Store the result in transposed position: output has shape [batch, k, m]
    # So we need to store with indices [batch, k_offsets, m_offsets]
    output_ptrs = (output_ptr + 
                  pid_batch * output_stride_0 +
                  k_offsets[:, None] * output_stride_1 +
                  m_offsets[None, :] * output_stride_2)
    tl.store(output_ptrs, accumulator, mask=m_mask[None, :] & k_mask[:, None])

@torch.fx.wrap
def optimized_matmul_transpose(a, b):
    # Input shapes: a [batch, m, n], b [batch, n, k]
    # Output shape: [batch, k, m] (transposed result)
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("Both inputs must be 3D tensors")
    
    batch_size, m, n = a.shape
    _, _, k = b.shape
    
    # Check if batch dimensions match
    if batch_size != b.shape[0]:
        raise ValueError("Batch dimensions must match")
    
    if a.shape[2] != b.shape[1]:
        raise ValueError("Inner dimensions must match for matrix multiplication")
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32  # Actually the inner dimension n
    BLOCK_SIZE_K = 32  # The k dimension of output
    
    output = torch.empty((batch_size, k, m), dtype=a.dtype, device=a.device)
    
    # Calculate grid dimensions
    grid_batch = batch_size
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    optimized_matmul_transpose_kernel[(grid_batch, grid_m, grid_k)](
        a_ptr=a,
        b_ptr=b,
        output_ptr=output,
        a_stride_0=a.stride(0),
        a_stride_1=a.stride(1),
        a_stride_2=a.stride(2),
        b_stride_0=b.stride(0),
        b_stride_1=b.stride(1),
        b_stride_2=b.stride(2),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        output_stride_2=output.stride(2),
        batch_size=batch_size,
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_matmul_transpose