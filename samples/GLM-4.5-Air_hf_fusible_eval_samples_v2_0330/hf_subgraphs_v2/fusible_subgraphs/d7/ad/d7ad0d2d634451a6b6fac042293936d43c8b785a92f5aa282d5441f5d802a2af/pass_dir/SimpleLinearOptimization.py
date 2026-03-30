import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    return torch.nn.functional.linear(z, y, x)

def replacement_args(x, y, z):
    return (x, y, z)

@triton.jit
def simple_linear_kernel(
    bias_ptr,      # [hidden_out]
    weight_ptr,    # [hidden_out, hidden_in] 
    x_ptr,         # [batch, seq_len, hidden_in]
    out_ptr,       # [batch, seq_len, hidden_out]
    batch,
    seq_len,
    hidden_in,
    hidden_out,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a portion of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Combine batch and seq_len dimensions
    total_batch_seq = batch * seq_len
    
    # Compute ranges 
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Mask for valid dimensions
    mask_m = m_range < total_batch_seq
    mask_n = n_range < hidden_out
    
    # Initialize accumulator with the correct data type
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)
    
    # Loop over K dimension (hidden_in)
    for k_start in range(0, hidden_in, BLOCK_K):
        k_end = min(k_start + BLOCK_K, hidden_in)
        k_offset = tl.arange(0, BLOCK_K)
        # Use only valid elements
        k_mask = k_offset < (k_end - k_start)
        
        # Compute indices for current K chunk
        input_idx = m_range[:, None] * hidden_in + (k_offset[None, :] + k_start)
        weight_idx = n_range[:, None] * hidden_in + (k_offset[None, :] + k_start)
        
        # Load tiles
        input_tile = tl.load(
            x_ptr + input_idx,
            mask=(mask_m[:, None] & k_mask[None, :]),
            other=0.0
        )
        
        weight_tile = tl.load(
            weight_ptr + weight_idx,
            mask=(mask_n[:, None] & k_mask[None, :]),
            other=0.0
        )
        
        # Transpose weight tile for correct matmul
        weight_tile = weight_tile.T
        
        # Matrix multiplication: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        # Ensure output type matches accumulator type
        dot_result = tl.dot(input_tile, weight_tile)
        accumulator += dot_result.to(dtype)
    
    # Load bias and add
    bias = tl.load(bias_ptr + n_range, mask=mask_n, other=0.0)
    accumulator += bias[None, :]
    
    # Store result
    output_idx = m_range[:, None] * hidden_out + n_range[None, :]
    tl.store(
        out_ptr + output_idx,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :]
    )

@torch.fx.wrap
def optimized_linear(bias, weight, x):
    batch, seq_len, hidden_in = x.shape
    hidden_out = bias.shape[0]
    
    # Smaller block sizes to reduce shared memory usage
    BLOCK_M = 64      # Combined batch*seq_len tile  
    BLOCK_N = 128     # hidden_out tile  
    BLOCK_K = 32      # hidden_in tile
    
    # Calculate grid size - 2D grid for the output matrix
    total_batch_seq = batch * seq_len
    grid_m = (total_batch_seq + BLOCK_M - 1) // BLOCK_M
    grid_n = (hidden_out + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Preserve input data type for better performance
    out = torch.empty((batch, seq_len, hidden_out), dtype=x.dtype, device=x.device)
    
    # Map torch dtype to triton dtype
    if x.dtype == torch.float16:
        triton_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    elif x.dtype == torch.float32:
        triton_dtype = tl.float32
    else:
        triton_dtype = tl.float32  # fallback
    
    # Launch kernel
    simple_linear_kernel[grid](
        bias_ptr=bias,
        weight_ptr=weight,
        x_ptr=x,
        out_ptr=out,
        batch=batch,
        seq_len=seq_len,
        hidden_in=hidden_in,
        hidden_out=hidden_out,
        dtype=triton_dtype,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return out

def replacement_func():
    return optimized_linear