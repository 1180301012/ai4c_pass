import torch
import triton
import triton.language as tl

# Pattern matching function for unified linear layer
def pattern(x, weight, bias):
    """
    Pattern: torch.nn.functional.linear(x, weight, bias)
    Handles both 2D and 3D+ inputs
    """
    return torch.nn.functional.linear(x, weight, bias)

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for 2D linear layer - optimized version
@triton.jit
def linear_kernel_2d_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M, K, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized 2D Linear kernel with tiling for better GPU utilization
    """
    # Program identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine the range of rows and cols this program handles
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Initialize output accumulator for this block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load bias for all output features in this block
    bias_base = bias_ptr + n_offsets
    bias_vals = tl.load(bias_base, mask=n_mask, other=0.0)
    acc += bias_vals[None, :]
    
    # Loop over K dimension in tiles for better memory locality
    for k in range(0, K, 1):
        # Load input elements for this batch row
        x_base = x_ptr + m_offsets[:, None] * K + k
        x_vals = tl.load(x_base, mask=m_mask[:, None], other=0.0)
        
        # Load weights for all output features
        w_base = weight_ptr + n_offsets[None, :] * K + k
        w_vals = tl.load(w_base, mask=n_mask[None, :], other=0.0)
        
        # Multiply and accumulate
        acc += x_vals * w_vals
    
    # Store results
    out_base = out_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(out_base, acc, mask=m_mask[:, None] & n_mask[None, :])

# Triton kernel for 3D linear layer - optimized version
@triton.jit
def linear_kernel_3d_optimized(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size, inner_dim, K, N,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_INNER: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """
    Optimized 3D Linear kernel with tiling
    """
    # Program identifiers
    batch_pid = tl.program_id(0)
    inner_pid = tl.program_id(1)
    feat_pid = tl.program_id(2)
    
    # Calculate ranges for this program
    batch_start = batch_pid * BLOCK_SIZE_BATCH
    inner_start = inner_pid * BLOCK_SIZE_INNER
    feat_start = feat_pid * BLOCK_SIZE_FEAT
    
    batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    inner_offsets = inner_start + tl.arange(0, BLOCK_SIZE_INNER)
    feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    
    # Bounds checking
    batch_mask = batch_offsets < batch_size
    inner_mask = inner_offsets < inner_dim
    feat_mask = feat_offsets < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER, BLOCK_SIZE_FEAT), dtype=tl.float32)
    
    # Load bias
    bias_base = bias_ptr + feat_offsets
    bias_vals = tl.load(bias_base, mask=feat_mask, other=0.0)
    acc += bias_vals[None, None, :]
    
    # Compute matrix multiplication
    for k in range(0, K, 1):
        # Load input elements - tensor of shape [BLOCK_SIZE_BATCH, BLOCK_SIZE_INNER]
        x_base = input_ptr + (batch_offsets[:, None] * inner_dim + inner_offsets[None, :]) * K + k
        x_vals = tl.load(x_base, mask=batch_mask[:, None] & inner_mask[None, :], other=0.0)
        
        # Load weights - tensor of shape [BLOCK_SIZE_FEAT]
        w_base = weight_ptr + feat_offsets * K + k
        w_vals = tl.load(w_base, mask=feat_mask, other=0.0)
        
        # Multiply and accumulate using broadcasting
        acc += x_vals[:, :, None] * w_vals[None, None, :]
    
    # Store results
    out_base = out_ptr + (batch_offsets[:, None, None] * inner_dim + inner_offsets[None, :, None]) * N + feat_offsets[None, None, :]
    mask = batch_mask[:, None, None] & inner_mask[None, :, None] & feat_mask[None, None, :]
    tl.store(out_base, acc, mask=mask)

@torch.fx.wrap
def optimized_linear_unified(x, weight, bias):
    N = bias.shape[0]
    
    # For linear layer: input @ weight.t() + bias
    # Weight should have shape [N, K]
    assert weight.shape[0] == N, f"Weight first dim {weight.shape[0]} must match bias dim {N}"
    
    if len(x.shape) == 2:
        # 2D case: [M, K] -> [M, N]
        M, K = x.shape
        assert weight.shape[1] == K, f"Weight second dim {weight.shape[1]} must match input dim {K}"
        
        output_shape = (M, N)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        
        # Use block sizes for better GPU utilization
        BLOCK_SIZE_M = 16  # Process 16 batch rows per program
        BLOCK_SIZE_N = 16  # Process 16 output features per program
        
        # Calculate grid size
        grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid = (grid_m, grid_n)
        
        linear_kernel_2d_optimized[grid](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=output,
            M=M, K=K, N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
    elif len(x.shape) == 3:
        # 3D case: [batch, inner, K] -> [batch, inner, N]
        batch_size, inner_dim, K = x.shape
        assert weight.shape[1] == K, f"Weight second dim {weight.shape[1]} must match input dim {K}"
        
        output_shape = (batch_size, inner_dim, N)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        
        # Use block sizes for better GPU utilization
        BLOCK_SIZE_BATCH = 8   # Process 8 batch items per program
        BLOCK_SIZE_INNER = 1   # Process 1 inner dim per program (since it's usually 1)
        BLOCK_SIZE_FEAT = 16   # Process 16 output features per program
        
        # Calculate grid size
        grid_batch = (batch_size + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
        grid_inner = (inner_dim + BLOCK_SIZE_INNER - 1) // BLOCK_SIZE_INNER
        grid_feat = (N + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT
        grid = (grid_batch, grid_inner, grid_feat)
        
        linear_kernel_3d_optimized[grid](
            input_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=output,
            batch_size=batch_size,
            inner_dim=inner_dim,
            K=K,
            N=N,
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_INNER=BLOCK_SIZE_INNER,
            BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT
        )
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear_unified