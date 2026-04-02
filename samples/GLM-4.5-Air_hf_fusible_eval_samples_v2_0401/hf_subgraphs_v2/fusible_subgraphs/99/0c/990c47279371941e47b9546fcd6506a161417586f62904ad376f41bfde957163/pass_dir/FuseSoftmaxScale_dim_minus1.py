import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_full_kernel(x_ptr, y_ptr, output_ptr, 
                         x_stride_0, x_stride_1, x_stride_2,
                         y_stride_0, y_stride_1, y_stride_2,
                         output_stride_0, output_stride_1, output_stride_2,
                         batch_size, m, n, k,
                         scale_factor: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute memory offsets
    batch_offset = pid_batch * x_stride_0
    
    # Each program processes a tile of size BLOCK_SIZE_M x BLOCK_SIZE_K
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    m_mask = m_offsets < m
    k_mask = k_offsets < k
    
    # Load data from x (for softmax part)
    x_ptrs = x_ptr + batch_offset + m_offsets[:, None] * x_stride_1
    x_data = tl.load(x_ptrs, mask=m_mask[:, None], other=-float('inf'))
    
    # First scale and compute softmax along last dimension
    x_scaled = x_data * scale_factor
    max_val = tl.max(x_scaled, axis=1)
    max_val_bcast = max_val[:, None]
    exp_x = tl.exp(x_scaled - max_val_bcast)
    sum_exp = tl.sum(exp_x, axis=1)
    sum_exp_bcast = sum_exp[:, None]
    softmax_out = exp_x / (sum_exp_bcast + 1e-20)
    
    # Accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Matrix multiplication: softmax_out @ y, where y is transposed on the fly
    for n_pos in range(0, n, BLOCK_SIZE_N):
        n_end = min(n_pos + BLOCK_SIZE_N, n)
        
        # Load from x (softmax output) for this n block
        x_ptrs_n = x_ptr + batch_offset + m_offsets[:, None] * x_stride_1 + n_pos * x_stride_2
        x_block = tl.load(x_ptrs_n, mask=m_mask[:, None] & (tl.arange(n_pos, n_end) < n)[:, None], other=0.0)
        
        # Load from y and transpose: original [batch, n, k] → we want [batch, k, n] 
        y_ptrs = y_ptr + batch_offset + k_offsets[:, None] * y_stride_1 + n_pos * y_stride_2
        y_block = tl.load(y_ptrs, mask=k_mask[:, None] & (tl.arange(n_pos, n_end) < n)[:, None], other=0.0)
        y_block = y_block.T  # Transpose to [blocked_n, blocked_k]
        
        # Matrix multiply: x_block [m, blocked_n] @ y_block [blocked_n, k] → accumulator [m, k]
        accumulator += tl.dot(x_block, y_block, out_type=tl.float32)
    
    # Store the result directly in transposed position: output [batch, k, m]
    output_ptrs = output_ptr + pid_batch * output_stride_0 + k_offsets[:, None] * output_stride_1 + m_offsets[None, :] * output_stride_2
    tl.store(output_ptrs, accumulator, mask=m_mask[None, :] & k_mask[:, None])

@torch.fx.wrap
def optimized_full_forward(x, y, scale_factor=0.0625):
    # Full computation: x -> scale -> softmax -> matmul with y -> transpose
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("Both inputs must be 3D tensors")
    
    batch_size, m, n = x.shape
    _, n_inner, k = y.shape
    
    if n != n_inner:
        raise ValueError("Inner dimensions must match for matrix multiplication")
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Allocate output tensor: [batch_size, k, m] (transposed result)
    output = torch.empty((batch_size, k, m), dtype=x.dtype, device=x.device)
    
    # Step 1: Scale, softmax, and matmul with transpose in one kernel
    optimized_full_kernel[(batch_size, (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        x_stride_0=x.stride(0),
        x_stride_1=x.stride(1),
        x_stride_2=x.stride(2),
        y_stride_0=y.stride(0),
        y_stride_1=y.stride(1),
        y_stride_2=y.stride(2),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        output_stride_2=output.stride(2),
        batch_size=batch_size,
        m=m,
        n=n,
        k=k,
        scale_factor=scale_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_full_forward