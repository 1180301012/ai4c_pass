import torch
import triton
import triton.language as tl
import math

def pattern(a, b, scale, c):
    """
    Match the exact computation sequence:
    tmp_0 = torch.matmul(a, b)
    tmp_1 = tmp_0 / scale  
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, c)
    Returns tmp_4, tmp_2 to maintain observability
    """
    tmp_0 = torch.matmul(a, b)
    tmp_1 = tmp_0 / scale
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, c)
    return tmp_4, tmp_2  # Return both results to maintain observability outside the pattern

def replacement_args(x, y, scale, z):
    return (x, y, scale, z)

@triton.jit
def fused_attention_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr, softmax_ptr,
    batch_size, num_heads, seq_len, head_dim,
    scale_val,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Fused attention computation: matmul(x,y)/scale → softmax → matmul(softmax,z)
    """
    # Program IDs for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory addresses
    x_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    x_ptrs = x_ptr + (x_offset[:, None] * seq_len + tl.arange(0, seq_len)[None, :]) * head_dim
    y_ptrs = y_ptr + (tl.arange(0, seq_len)[None, :] * seq_len + y_offset[:, None]) * head_dim
    z_ptrs = z_ptr + (y_offset[:, None] * seq_len + tl.arange(0, seq_len)[None, :]) * head_dim
    
    out_ptrs = out_ptr + (x_offset[:, None] * seq_len + y_offset[:, None]) * head_dim
    softmax_ptrs = softmax_ptr + (x_offset[:, None] * seq_len + y_offset[:, None]) * head_dim
    
    # Initialize accumulator and load z
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    z_block = tl.load(z_ptrs, mask=(y_offset[:, None] < seq_len) & (tl.arange(0, seq_len)[None, :] < seq_len), other=0.0)
    
    # Main loop for matmul(x,y)/scale
    for k in range(0, seq_len, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, seq_len)
        x_block = tl.load(x_ptrs, mask=(x_offset[:, None] < seq_len) & (tl.arange(k, k_end)[None, :] < seq_len), other=0.0)
        y_block = tl.load(y_ptrs, mask=(tl.arange(k, k_end)[None, :] < seq_len) & (y_offset[:, None] < seq_len), other=0.0)
        
        # Perform matrix multiplication with scaling
        accumulator += tl.dot(x_block, y_block.to(tl.float32), trans_b=True) * scale_val
    
    # Apply softmax
    max_val = tl.max(accumulator, axis=1)
    softmax = tl.exp(accumulator - max_val[:, None])
    sum_exp = tl.sum(softmax, axis=1)
    softmax = softmax / sum_exp[:, None]
    
    # Store softmax result
    tl.store(softmax_ptrs, softmax.to(tl.float32), mask=(x_offset[:, None] < seq_len) & (y_offset[:, None] < seq_len))
    
    # Second matmul: matmul(softmax,z)
    for k in range(0, seq_len, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, seq_len)
        softmax_block = tl.load(softmax_ptrs + (tl.arange(0, seq_len)[None, :] * seq_len + tl.arange(k, k_end)[:, None]) * head_dim,
                               mask=(x_offset[:, None] < seq_len) & (tl.arange(k, k_end)[None, :] < seq_len), other=0.0)
        z_block_slice = tl.load(z_ptrs, mask=(y_offset[:, None] < seq_len) & (tl.arange(k, k_end)[None, :] < seq_len), other=0.0)
        
        accumulator = tl.dot(softmax_block, z_block_slice.to(tl.float32))
    
    # Store final result
    tl.store(out_ptrs, accumulator.to(tl.float32), mask=(x_offset[:, None] < seq_len) & (y_offset[:, None] < seq_len))

@torch.fx.wrap
def fused_attention_operation(x, y, scale, z):
    """
    Fused attention computation wrapper
    """
    # Get input shapes
    batch_size, num_heads, seq_len_x, head_dim = x.shape
    _, _, seq_len_y, _ = y.shape
    _, _, seq_len_z, _ = z.shape
    
    # Ensure shapes are compatible
    assert seq_len_x == seq_len_z, "Sequence length mismatch"
    assert head_dim == y.shape[-1], "Head dimension mismatch"
    assert seq_len_y == z.shape[-2], "Key sequence length mismatch"
    
    # Calculate optimal block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    
    # Allocate output tensors
    out_shape = (batch_size, num_heads, seq_len_x, seq_len_y)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    softmax_out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Calculate grid dimensions
    m_grid = (seq_len_x + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_grid = (seq_len_y + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_attention_kernel[(m_grid, n_grid)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        out_ptr=out,
        softmax_ptr=softmax_out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len_x,
        head_dim=head_dim,
        scale_val=scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out, softmax_out

def replacement_func():
    return fused_attention_operation