import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_1, in_0):
    """
    Pattern to match the LayerNorm computation after the initial addition:
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5  # Redundant - same as tmp_6
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    """
    tmp_4 = tmp_3.float()
    
    # Calculate mean and variance
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    
    # Eliminate redundant operation - use tmp_6 instead of recomputing tmp_9
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_6 / tmp_11  # Use tmp_6 instead of recomputed tmp_4 - tmp_5
    tmp_13 = tmp_12.to(torch.float32)
    
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    
    return tmp_15

def replacement_args(tmp_3, in_1, in_0):
    return (tmp_3, in_1, in_0)

@triton.jit
def optimized_layernorm_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    N,           # batch_size * seq_len
    D,           # hidden_dim (768)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a row (tokens in sequence)
    pid = tl.program_id(0)
    batch_seq_idx = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = batch_seq_idx < N
    
    # Each thread handles a dimension for the given token
    pid_d = tl.program_id(1)
    dim_idx = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    mask_d = dim_idx < D
    
    # Load input for this token
    x_ptr_token = x_ptr + batch_seq_idx[:, None] * D + dim_idx[None, :]
    x = tl.load(x_ptr_token, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    
    # Convert to float32 for numerical precision
    x_float = x.to(tl.float32)
    
    # Calculate mean for this token
    mean_sum = tl.sum(x_float, axis=1)
    mean = mean_sum / D
    
    # Compute centered values and sum of squares for variance
    x_centered = x_float - mean[:, None]
    sum_squares = tl.sum(x_centered * x_centered, axis=1)
    var = sum_squares / D
    
    # Add epsilon and compute std
    var_eps = var + 1e-07
    std = tl.sqrt(var_eps)
    
    # Normalize
    x_norm = x_centered / std[:, None]
    
    # Load weight and bias
    weight = tl.load(weight_ptr + dim_idx, mask=mask_d, other=0.0)
    bias = tl.load(bias_ptr + dim_idx, mask=mask_d, other=0.0)
    
    # Apply weight and bias
    out = x_norm * weight[None, :] + bias[None, :]
    
    # Store result
    out_ptr_token = out_ptr + batch_seq_idx[:, None] * D + dim_idx[None, :]
    tl.store(out_ptr_token, out, mask=mask_n[:, None] & mask_d[None, :])

@torch.fx.wrap
def optimized_layernorm(x, weight, bias):
    # Get input dimensions
    batch_size, seq_len, hidden_dim = x.shape
    N = batch_size * seq_len
    
    # Optimal block sizes based on hidden dimension (768)
    BLOCK_SIZE_D = 64   # Process 64 dimensions per thread
    BLOCK_SIZE_N = 128   # Process 128 tokens per program
    
    # Calculate grid size
    num_programs_N = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs_D = (hidden_dim + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch kernel
    optimized_layernorm_kernel[(num_programs_N, num_programs_D)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N,
        D=hidden_dim,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_layernorm