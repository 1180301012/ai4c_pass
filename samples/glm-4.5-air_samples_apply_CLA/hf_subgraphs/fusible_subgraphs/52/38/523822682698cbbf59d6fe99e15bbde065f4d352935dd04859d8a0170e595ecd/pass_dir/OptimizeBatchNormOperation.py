import torch
import triton
import triton.language as tl

# Pattern matching function - matches batch norm operation
def pattern(in_7, in_0, in_1, in_3, in_2):
    # Batch norm operation with training=False, momentum=0.1, eps=1e-05
    # This corresponds to torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)

# Argument extraction function
def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

# Optimized batch norm kernel using Triton
@triton.jit
def batch_norm_kernel(
    x_ptr,           # Pointer to input tensor [batch, features]
    running_mean_ptr, # Pointer to running mean [features]
    running_var_ptr,  # Pointer to running variance [features] 
    weight_ptr,       # Pointer to weight [features]
    bias_ptr,         # Pointer to bias [features]
    out_ptr,          # Pointer to output tensor [batch, features]
    batch,            # Batch size
    features,         # Features dimension
    eps,              # Epsilon for numerical stability
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for features dimension
):
    # Program identifiers
    m = tl.program_id(0)  # Batch dimension
    n = tl.program_id(1)  # Features dimension
    
    # Compute ranges for this program
    m_offset = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    m_mask = m_offset < batch
    n_mask = n_offset < features
    
    # Load parameters for this feature slice
    running_mean = tl.load(running_mean_ptr + n_offset, mask=n_mask, other=0.0)
    running_var = tl.load(running_var_ptr + n_offset, mask=n_mask, other=0.0)
    weight = tl.load(weight_ptr + n_offset, mask=n_mask, other=1.0)
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    
    # Precompute normalization terms for better performance
    # Avoid repeated computation in the batch dimension
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    
    # Load input slice
    x = tl.load(x_ptr + m_offset[:, None] * features + n_offset[None, :], 
               mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Batch norm computation: (x - running_mean) * inv_std * weight + bias
    # Group operations for better numerical stability and performance
    normalized = (x - running_mean[None, :]) * inv_std[None, :] * weight[None, :] + bias[None, :]
    
    # Store result
    tl.store(out_ptr + m_offset[:, None] * features + n_offset[None, :], 
             normalized, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def triton_batch_norm(in_7, in_0, in_1, in_3, in_2):
    # Get tensor shapes
    batch, features = in_7.shape
    
    # Output tensor
    out = torch.empty((batch, features), dtype=in_7.dtype, device=in_7.device)
    
    # Set tile sizes based on problem characteristics
    # For smaller batch sizes, use smaller M block size
    if batch == 1:
        BLOCK_SIZE_M = 1
    elif batch <= 32:
        BLOCK_SIZE_M = 32
    else:
        BLOCK_SIZE_M = 64  # For larger batches
    
    # For features, use powers of 2 since features=384 in our case
    BLOCK_SIZE_N = 128  # 384/128 = 3, good distribution
    
    # Calculate grid dimensions
    grid_m = (batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with epsilon=1e-05
    batch_norm_kernel[(grid_m, grid_n)](
        x_ptr=in_7,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        out_ptr=out,
        batch=batch,
        features=features,
        eps=1e-05,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_batch_norm