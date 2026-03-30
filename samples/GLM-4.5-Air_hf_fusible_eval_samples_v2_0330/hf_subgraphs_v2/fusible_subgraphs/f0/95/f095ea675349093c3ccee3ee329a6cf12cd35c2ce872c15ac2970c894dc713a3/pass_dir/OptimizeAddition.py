import torch
import triton
import triton.language as tl

# Pattern matching for addition operation (skip connection)
def pattern(in_3, tmp_6):
    # Addition operation with skip connection
    tmp_7 = in_3 + tmp_6
    return tmp_7

# Extract arguments for replacement
def replacement_args(in_3, tmp_6):
    return (in_3, tmp_6)

@triton.jit
def optimized_add_kernel(
    x_ptr,        # First tensor [1, 249, 1024] or similar
    y_ptr,        # Second tensor [1, 185, 1024] or similar
    out_ptr,      # Output tensor
    n_batch: tl.constexpr,
    n_seq_x: tl.constexpr,
    n_seq_y: tl.constexpr, 
    n_features: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Calculate program IDs
    pid_m = tl.program_id(0)  # Batch dimension
    
    # Bounds checking for batch
    if pid_m >= n_batch:
        return
        
    # Each program handles a block of sequence and feature dimensions
    seq_start = tl.program_id(1) * BLOCK_SIZE_M
    feat_start = tl.program_id(2) * BLOCK_SIZE_N
    
    seq_end = min(seq_start + BLOCK_SIZE_M, max(n_seq_x, n_seq_y))
    feat_end = min(feat_start + BLOCK_SIZE_N, n_features)
    
    if seq_start >= max(n_seq_x, n_seq_y) or feat_start >= n_features:
        return
    
    # Process each sequence and feature position
    for seq_idx in range(seq_start, seq_end):
        for feat_idx in range(feat_start, feat_end):
            offset_x = pid_m * (n_seq_x * n_features) + seq_idx * n_features + feat_idx
            offset_y = pid_m * (n_seq_y * n_features) + min(seq_idx, n_seq_y - 1) * n_features + feat_idx
            offset_out = pid_m * (max(n_seq_x, n_seq_y) * n_features) + seq_idx * n_features + feat_idx
            
            # Load values with bounds checking
            x_val = tl.load(x_ptr + offset_x) if seq_idx < n_seq_x else 0.0
            y_val = tl.load(y_ptr + offset_y) if seq_idx < n_seq_y else 0.0
            
            # Add elements
            result = x_val + y_val
            
            # Store result
            tl.store(out_ptr + offset_out, result)

@torch.fx.wrap
def optimized_addition(in_3, tmp_6):
    # Get input shapes
    in_3_shape = in_3.shape
    tmp_6_shape = tmp_6.shape
    
    # Handle different sequence lengths (skip connection)
    n_batch = max(in_3_shape[0], tmp_6_shape[0])
    n_seq_x = in_3_shape[1]  # 249 for in_3
    n_seq_y = tmp_6_shape[1]  # 185 for tmp_6  
    n_features = max(in_3_shape[2], tmp_6_shape[2])  # Should be 1024
    
    # Result shape takes the maximum sequence length
    n_output_seq = max(n_seq_x, n_seq_y)
    out_shape = (n_batch, n_output_seq, n_features)
    out = torch.empty(out_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Configure block sizes for optimal GPU utilization  
    BLOCK_SIZE_M = 64   # Sequence dimension block size
    BLOCK_SIZE_N = 256  # Feature dimension block size
    
    # Calculate grid size
    grid_m = n_batch
    grid_seq = (n_output_seq + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_feat = (n_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    optimized_add_kernel[(grid_m, grid_seq, grid_feat)](
        x_ptr=in_3,
        y_ptr=tmp_6,
        out_ptr=out,
        n_batch=n_batch,
        n_seq_x=n_seq_x,
        n_seq_y=n_seq_y,
        n_features=n_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_addition