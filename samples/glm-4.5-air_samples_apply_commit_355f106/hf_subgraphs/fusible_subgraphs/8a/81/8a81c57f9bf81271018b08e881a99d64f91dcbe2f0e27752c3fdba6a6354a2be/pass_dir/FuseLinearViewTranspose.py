import torch
import triton
import triton.language as tl

@triton.jit
def linear_view_transpose_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_hidden,
    n_heads,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)
    
    # Create masks
    m_mask = m_offset < n_batch
    n_mask = n_offset < n_hidden
    k_mask = k_offset < n_seq
    
    # Load bias if exists
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    else:
        bias = 0.0
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator += bias[None, :]
    
    # Loop over K dimension
    for k in range(0, n_seq, BLOCK_K):
        k_start = k + k_offset
        
        # Load x and weight
        x_ptrs = x_ptr + m_offset[:, None] * n_seq * n_hidden + k_start[None, :] * n_hidden + n_offset[None, :]
        w_ptrs = weight_ptr + n_offset[:, None] * n_seq + k_start[None, :]
        
        x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(x, w, out_type=tl.float32)
    
    # Transpose to get [batch, n_heads, seq_len, head_dim] format
    # First reshape to [batch, n_hidden, seq_len] then transpose last two dims
    # Since n_hidden = n_heads * head_dim, we reshape and transpose
    output = accumulator
    
    # Store result directly in the transposed format
    out_base = out_ptr + m_offset[:, None] * n_heads * head_dim * n_seq + n_offset[None, :] * n_seq
    tl.store(out_base, output, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def linear_view_transpose_fusion(x, weight, bias, view_shape):
    batch_size, seq_len, hidden_dim = x.shape
    n_heads, head_dim = view_shape[-2:]
    
    assert hidden_dim == n_heads * head_dim, f"Hidden dim {hidden_dim} doesn't match {n_heads} * {head_dim}"
    
    # Allocate output in transposed format: [batch, n_heads, seq_len, head_dim]
    output = torch.empty((batch_size, n_heads, seq_len, head_dim), dtype=x.dtype, device=x.device)
    
    # Set block sizes based on typical GPU architecture
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Calculate grid sizes
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (hidden_dim + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    linear_view_transpose_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        n_batch=batch_size,
        n_seq=seq_len,
        n_hidden=hidden_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output

def pattern(x, weight, bias):
    # Simple pattern without shape access during symbolic tracing
    tmp_2 = torch.addmm(bias, x, weight)
    # Use standard transformer reshaping: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
    tmp_3 = tmp_2.view(x.shape[0], x.shape[1], 12, 64)  # Common transformer pattern
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4

def replacement_args(x, weight, bias):
    # Extract shapes from inputs
    return (x, weight, bias)

def replacement_func():
    return linear_view_transpose_fusion