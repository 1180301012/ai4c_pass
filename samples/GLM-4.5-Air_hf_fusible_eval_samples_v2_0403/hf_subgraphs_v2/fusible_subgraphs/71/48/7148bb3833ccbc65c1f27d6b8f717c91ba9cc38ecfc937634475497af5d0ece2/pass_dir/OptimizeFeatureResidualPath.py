import torch
import triton
import triton.language as tl
import math

def pattern(in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def feature_residual_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    pid_batch = tl.program_id(2)
    
    # Compute ranges
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    m_mask = m_offset < C
    n_mask = n_offset + tl.arange(0, BLOCK_SIZE_N) < W
    batch_mask = pid_batch < N
    
    if not (m_mask and n_mask and batch_mask):
        return
    
    column_offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
    scale_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
    batch_offset = pid_batch
    
    # Load input [C, H, W]
    base_offset = batch_offset * C * H * W + m_offset * H * W
    feature_offsets = (column_offsets[:, None] * H + tl.arange(0, H)[None, :]).to(tl.int64)
    
    # Load features
    features = tl.load(input_ptr + base_offset + feature_offsets, 
                       mask=(m_mask[:, None] & n_mask[None, :] & batch_mask), 
                       other=0.0)
    
    # Compute ReLU
    features_relu = tl.maximum(features, 0.0)
    
    # Compute avg_pool2d with 3x3 kernel, stride 1, padding 1
    # Simplified pooling - average over 3x3 neighborhood
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, H), dtype=tl.float32)
    
    for k_col in range(-1, 2):
        for k_row in range(-1, 2):
            # Compute padded coordinates
            padded_col = column_offsets[:, None] + k_col
            padded_row = tl.arange(0, H)[None, :] + k_row
            
            # Create mask for valid positions
            col_mask = (padded_col >= 0) & (padded_col < W)
            row_mask = (padded_row >= 0) & (padded_row < H)
            valid_mask = col_mask & row_mask
            
            # Load pooled data with boundary handling
            pooled_data = tl.where(valid_mask,
                features[:, :, padded_col % W, padded_row],
                0.0)
            acc += pooled_data
    
    relu_pooled = acc / 9.0
    
    # Compute subtraction (residual)
    diff_features = relu_pooled - features_relu
    
    # Store result
    tl.store(output_ptr + base_offset + feature_offsets, diff_features,
             mask=(m_mask[:, None] & n_mask[None, :] & batch_mask))

@torch.fx.wrap
def feature_residual_optimized_torch(in_2):
    # Get tensor shapes
    N, C, H, W = in_2.shape
    output = torch.empty_like(in_2)
    
    # Kernel configuration
    BLOCK_SIZE_M = 64  # Process multiple channels at once
    BLOCK_SIZE_N = 64  # Process multiple width positions at once
    
    # Calculate grid dimensions
    grid_m = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_batch = N
    
    # Launch kernel
    feature_residual_kernel[(grid_m, grid_n, grid_batch)](
        input_ptr=in_2,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return feature_residual_optimized_torch