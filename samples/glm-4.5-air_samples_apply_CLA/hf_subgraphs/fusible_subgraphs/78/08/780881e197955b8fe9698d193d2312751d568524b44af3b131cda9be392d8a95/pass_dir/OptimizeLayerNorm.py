import torch
import triton
import triton.language as tl

def pattern(tmp_8, normalized_shape, tmp_1, tmp_0, eps=1e-05):
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, normalized_shape, tmp_1, tmp_0, eps)
    return tmp_8, tmp_9

def replacement_args(tmp_8, normalized_shape, tmp_1, tmp_0, eps=1e-05):
    return (tmp_8, normalized_shape, tmp_1, tmp_0, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    n_elements,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program id
    pid = tl.program_id(0)
    
    # Compute row offsets
    row_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = row_offsets < n_elements
    
    # Load input data
    x_ptrs = x_ptr + row_offsets.reshape(-1, 1) + tl.arange(0, BLOCK_SIZE_N).reshape(1, -1)
    x = tl.load(x_ptrs, mask=mask[:, None], other=0.0)
    
    # Compute mean
    row_sum = tl.sum(x, axis=1)
    row_mean = row_sum / n_cols
    
    # Compute variance  
    x_centered = x - row_mean.reshape(-1, 1)
    x_squared = x_centered * x_centered
    row_sum_sq = tl.sum(x_squared, axis=1)
    row_var = row_sum_sq / n_cols + eps
    
    # Compute standard deviation
    row_std = tl.sqrt(row_var)
    
    # Load gamma and beta
    gamma_ptrs = gamma_ptr + tl.arange(0, BLOCK_SIZE_N).reshape(1, -1)
    beta_ptrs = beta_ptr + tl.arange(0, BLOCK_SIZE_N).reshape(1, -1)
    
    gamma = tl.load(gamma_ptrs, mask=gamma_ptrs < n_cols, other=1.0)
    beta = tl.load(beta_ptrs, mask=beta_ptrs < n_cols, other=0.0)
    
    # Normalize
    x_normalized = (x_centered / row_std.reshape(-1, 1)) * gamma + beta
    
    # Store output
    output_ptrs = output_ptr + row_offsets.reshape(-1, 1) + tl.arange(0, BLOCK_SIZE_N).reshape(1, -1)
    tl.store(output_ptrs, x_normalized, mask=mask[:, None])

@torch.fx.wrap
def triton_layer_norm(x, normalized_shape, weight, bias, eps=1e-05):
    # Extract the last dimension size from the shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = min(128, n_cols)
    num_programs = (n_rows * n_cols + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        gamma_ptr=weight,
        beta_ptr=bias,
        output_ptr=output,
        n_elements=n_rows * n_cols,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return x, output  # Return both input and output as required

def replacement_func():
    return triton_layer_norm