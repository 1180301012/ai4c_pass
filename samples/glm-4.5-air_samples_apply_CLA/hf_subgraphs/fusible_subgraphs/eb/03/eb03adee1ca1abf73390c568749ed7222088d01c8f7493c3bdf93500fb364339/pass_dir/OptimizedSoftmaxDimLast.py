import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern matches softmax on last dimension
    return x.softmax(dim=-1)

def replacement_args(x):
    return (x,)

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr, 
    row_stride,
    col_stride,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one row of the softmax dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k_group = tl.program_id(2)
    
    # Check bounds
    if pid_m >= M or pid_n >= N:
        return
        
    # Calculate k indices for this workgroup
    k_start = pid_k_group * BLOCK_SIZE_K
    k_end = min(k_start + BLOCK_SIZE_K, K)
    k_range = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_range < (k_end - k_start)
    k_indices = k_range + k_start
    
    # Base offset for this row
    base_offset = pid_m * row_stride + pid_n * col_stride
    
    # Load the entire row for softmax computation
    # We need to handle the remaining spatial dimensions efficiently
    spatial_size = row_stride // (M * N * K)
    k_offset = base_offset + k_indices * spatial_size
    
    # Load the row with masking        
    x = tl.load(x_ptr + k_offset, mask=k_mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(x)
    x_exp = tl.exp(x - row_max)
    
    # Compute sum of exponentials  
    sum_exp = tl.sum(x_exp)
    
    # Compute softmax
    softmax_vals = x_exp / sum_exp
    
    # Store results
    tl.store(out_ptr + k_offset, softmax_vals, mask=k_mask)

# Define the decorated function at module level
@torch.fx.wrap
def optimized_softmax(x):
    # Get shape information: [1, 361, 3, 49, 49]
    shape = x.shape
    batch_size = shape[0]
    heads = shape[1] 
    dim_n = shape[2]
    spatial_size1 = shape[3]
    spatial_size2 = shape[4]  # This is the softmax dimension
    
    # For Triton, we process without batch dimension first
    x_no_batch = x.squeeze(0)  # [361, 3, 49, 49]
    
    # Calculate strides
    # We need to flatten the spatial dimensions before the softmax dim
    # New shape: [361, 3, 49*49] with softmax on the last dimension [49*49]

    # Calculate stride for the pre-softmax dimensions
    row_stride = dim_n * spatial_size1 * spatial_size2  # stride from one head to next
    col_stride = spatial_size1 * spatial_size2         # stride from one n to next
    
    # Flatten the spatial dimensions before softmax
    x_flat = x_no_batch.reshape(heads, dim_n, spatial_size1 * spatial_size2)
    
    # Create output tensor
    out_flat = torch.empty_like(x_flat)
    
    # Configure kernel launch
    M = heads
    N = dim_n  
    K = spatial_size1 * spatial_size2  # This is our softmax dimension size
    
    # Block sizes
    BLOCK_SIZE_K = 64  # Process multiple softmax elements per thread
    
    # Calculate grid size - each program handles one (head, n) pair
    grid_m = M
    grid_n = N
    num_k_groups = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid_k = num_k_groups
    
    # Launch kernel
    softmax_kernel[(grid_m, grid_n, grid_k)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        row_stride=row_stride,
        col_stride=col_stride,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape back to original format
    out = out_flat.reshape(heads, dim_n, spatial_size1, spatial_size2)
    
    # Add batch dimension back
    return out.unsqueeze(0)

def replacement_func():
    return optimized_softmax