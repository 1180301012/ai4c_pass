import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    n_rows,
    n_cols,
    n_features_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized Triton kernel for linear operation"""
    # Program identifiers for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this program
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for bounds checking
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_features_out
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over columns of input rows (K dimension)
    for k in range(0, n_cols, BLOCK_SIZE_M):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_M)
        k_mask = k_offsets < n_cols
        
        # Load input data
        x_local = tl.load(x_ptr + row_offsets[:, None] * n_cols + k_offsets[None, :], 
                         mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load weight data
        weight_local = tl.load(weight_ptr + k_offsets[:, None] * n_features_out + col_offsets[None, :], 
                              mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(x_local, weight_local.to(tl.float32))
    
    # Load bias and add
    bias_local = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    accumulator += bias_local[None, :]
    
    # Store the result
    out_ptrs = out_ptr + row_offsets[:, None] * n_features_out + col_offsets[None, :]
    tl.store(out_ptrs, accumulator.to(tl.float32), mask=row_mask[:, None] & col_mask[None, :])

@triton.jit
def batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for batch normalization"""
    # Program identifier for 1D grid
    pid = tl.program_id(0)
    element_start = pid * BLOCK_SIZE
    element_offsets = element_start + tl.arange(0, BLOCK_SIZE)
    element_mask = element_offsets < (n_rows * n_features)
    
    # Convert element offsets to row/feature indices
    feature_idx = element_offsets % n_features
    row_idx = element_offsets // n_features
    
    # Load parameters for the specific features we're processing
    mean_vals = tl.load(mean_ptr + feature_idx, mask=element_mask, other=0.0)
    var_vals = tl.load(var_ptr + feature_idx, mask=element_mask, other=1.0)
    weight_vals = tl.load(weight_ptr + feature_idx, mask=element_mask, other=1.0)
    bias_vals = tl.load(bias_ptr + feature_idx, mask=element_mask, other=0.0)
    
    # Load input values
    x_vals = tl.load(x_ptr + element_offsets, mask=element_mask, other=0.0)
    
    # Apply batch normalization formula
    x_float = x_vals.to(tl.float32)
    denominator = tl.sqrt(var_vals + eps)
    normalized = (x_float - mean_vals) / denominator * weight_vals + bias_vals
    
    # Store result
    tl.store(out_ptr + element_offsets, normalized, mask=element_mask)

@torch.fx.wrap
def parallel_linear_batch_norm(x_linear, weight_linear, bias_linear, 
                             x_bn, mean_bn, var_bn, weight_bn, bias_bn):
    """Wrapper function that launches both kernels in parallel"""
    
    # Linear operation setup
    n_rows, n_cols = x_linear.shape
    n_features_out = weight_linear.shape[0]
    
    # Linear kernel launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_features_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Output tensors
    out_linear = torch.empty((n_rows, n_features_out), device=x_linear.device, dtype=x_linear.dtype)
    out_bn = torch.empty_like(x_bn)
    
    # Launch both kernels (they run independently on different data)
    linear_kernel[(grid_m, grid_n)](
        x_linear, weight_linear, bias_linear, out_linear,
        n_rows, n_cols, n_features_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    batch_norm_kernel[((x_bn.shape[0] * x_bn.shape[1]) + 63) // 64](
        x_bn, mean_bn, var_bn, weight_bn, bias_bn, out_bn,
        x_bn.shape[0], x_bn.shape[1], 1e-05, 64
    )
    
    return out_linear, out_bn

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Pattern matching the computation:
    - Linear: torch.nn.functional.linear(in_6, in_5, in_4)
    - BatchNorm: torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    """
    linear_out = torch.nn.functional.linear(in_6, in_5, in_4)
    batch_norm_out = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return linear_out, batch_norm_out

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Extract all arguments needed for the replacement"""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

def replacement_func():
    """Return the replacement function"""
    return parallel_linear_batch_norm