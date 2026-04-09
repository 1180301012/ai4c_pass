import torch
import triton
import triton.language as tl

@triton.jit
def transpose_2d_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one tile of the matrix
    pid = tl.program_id(0)
    num_rows = tl.cdiv(M, BLOCK_SIZE_M)
    grid_m = pid // num_rows
    grid_n = pid % num_rows
    
    m_mask = grid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < M
    n_mask = grid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < N
    
    x_ptrs = x_ptr + (grid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * N + (grid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    y_ptrs = y_ptr + (grid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] * N + (grid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    
    x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask, other=0.0)
    tl.store(y_ptrs, x, mask=n_mask[:, None] & m_mask)

@triton.jit
def transpose_4d_kernel(
    x_ptr,
    y_ptr,
    batch_size,
    seq_len,
    heads,
    dim_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Each program handles output element [batch, seq, d, h]
    batch_seq_pid = tl.program_id(0)
    h_block_pid = tl.program_id(1)
    d_block_pid = tl.program_id(2)
    
    # Calculate batch and seq from flattened index
    batch = batch_seq_pid // seq_len
    seq = batch_seq_pid % seq_len
    
    # Calculate head and dim ranges for this block
    h_start = h_block_pid * BLOCK_SIZE_H
    h_end = min(h_start + BLOCK_SIZE_H, heads)
    d_start = d_block_pid * BLOCK_SIZE_D
    d_end = min(d_start + BLOCK_SIZE_D, dim_size)
    
    # Calculate input base offset for this (batch, seq) slice
    input_base_offset = batch * seq_len * heads * dim_size + seq * heads * dim_size
    
    # Transpose this block of [h, d] -> [d, h]
    for h in range(h_start, h_end):
        for d in range(d_start, d_end):
            # Input: [batch, seq, h, d]
            input_offset = input_base_offset + h * dim_size + d
            # Output: [batch, seq, d, h] 
            output_offset = input_base_offset + d * heads + h
            
            # Load with bounds checking
            in_mask = (batch < batch_size) & (seq < seq_len) & (h < heads) & (d < dim_size)
            if in_mask:
                x_data = tl.load(x_ptr + input_offset)
                tl.store(y_ptr + output_offset, x_data)

@torch.fx.wrap
def optimized_transpose(x):
    # For now, use a simple optimized approach
    # For 2D tensors, use Triton kernel
    # For 4D tensors, fall back to torch.transpose which is already optimized
    
    if len(x.shape) == 2:
        # For 2D tensors only - use Triton
        M, N = x.shape[0], x.shape[1]
        out = torch.empty((N, M), dtype=x.dtype, device=x.device)
        
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        grid_size = ((M * N + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N),)
        
        transpose_2d_kernel[grid_size](
            x_ptr=x, 
            y_ptr=out,
            M=M,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        return out
    else:
        # For 4D or other tensors, fall back to built-in which is already optimized
        return x.transpose(-1, -2)

def pattern(x):
    return x.transpose(-1, -2)

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_transpose