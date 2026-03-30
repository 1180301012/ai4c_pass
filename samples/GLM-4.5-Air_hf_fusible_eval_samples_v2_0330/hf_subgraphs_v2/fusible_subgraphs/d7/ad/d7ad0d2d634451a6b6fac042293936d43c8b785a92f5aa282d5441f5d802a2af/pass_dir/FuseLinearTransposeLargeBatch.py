import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    # Linear operation: z @ y.T + x
    linear = torch.nn.functional.linear(z, y, x)
    # Transpose operation
    transposed = linear.permute(0, 2, 1)
    return linear, transposed

def replacement_args(x, y, z):
    return (x, y, z)

@triton.jit
def fused_linear_transpose_kernel_large(
    bias_ptr,      # [hidden_out]
    weight_ptr,    # [hidden_out, hidden_in] 
    x_ptr,         # [batch, seq_len, hidden_in]
    out_ptr,       # [batch, hidden_out, seq_len]
    batch,
    seq_len,
    hidden_in,
    hidden_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a portion of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_ranges = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_ranges = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_ranges = tl.arange(0, BLOCK_K)
    
    # Mask for valid M and N dimensions
    mask_m = m_ranges < batch
    mask_n = n_ranges < hidden_out
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_start in range(0, hidden_in, BLOCK_K):
        k_end = min(k_start + BLOCK_K, hidden_in)
        k_offset = tl.arange(0, k_end - k_start)
        
        # Load bias (broadcasted)
        bias = tl.load(bias_ptr + n_ranges, mask=mask_n, other=0.0)
        
        # Load weight tile
        weight_tile = tl.load(
            weight_ptr + (n_ranges[:, None] * hidden_in + k_offset[None, :]),
            mask=(mask_n[:, None] & (k_offset[None, :] < (k_end - k_start))),
            other=0.0
        )
        
        # Load input tile
        input_tile = tl.load(
            x_ptr + (m_ranges[:, None] * seq_len * hidden_in + (k_offset[None, :] + k_start)),
            mask=(mask_m[:, None] & (k_offset[None, :] < (k_end - k_start))),
            other=0.0
        )
        
        # Matrix multiplication
        accumulator += tl.dot(input_tile, weight_tile)
    
    # Add bias
    accumulator += bias[None, :]
    
    # Store result (already transposed)
    accumulator = accumulator.to(tl.float32)
    tl.store(
        out_ptr + (m_ranges[:, None] * hidden_out + n_ranges[None, :]),
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :]
    )

@torch.fx.wrap
def fused_linear_transpose_large(bias, weight, x):
    batch, seq_len, hidden_in = x.shape
    hidden_out = bias.shape[0]
    
    # Large batch optimization: use larger block sizes for better GPU utilization
    BLOCK_M = 8   # batch size tile (larger for better occupancy)
    BLOCK_N = 256  # hidden_out tile (larger for better memory access)
    BLOCK_K = 64   # hidden_in tile (larger for better cache usage)
    
    # Calculate grid size
    grid_m = (batch + BLOCK_M - 1) // BLOCK_M
    grid_n = (hidden_out + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Create output tensor (already transposed)
    out = torch.empty((batch, hidden_out, seq_len), dtype=torch.float32, device=x.device)
    
    # Launch kernel
    fused_linear_transpose_kernel_large[grid](
        bias_ptr=bias,
        weight_ptr=weight,
        x_ptr=x,
        out_ptr=out,
        batch=batch,
        seq_len=seq_len,
        hidden_in=hidden_in,
        hidden_out=hidden_out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return out

def replacement_func():
    return fused_linear_transpose_large