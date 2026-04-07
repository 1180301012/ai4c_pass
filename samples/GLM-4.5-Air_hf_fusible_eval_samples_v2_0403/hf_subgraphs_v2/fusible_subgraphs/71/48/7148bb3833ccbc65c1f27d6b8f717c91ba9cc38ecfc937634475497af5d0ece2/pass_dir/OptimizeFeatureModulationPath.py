import torch
import triton
import triton.language as tl
import math

def pattern(in_2, in_0, in_1):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_3 = None
    tmp_5 = in_0.unsqueeze(-1)
    tmp_0 = None
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_5 = None
    tmp_7 = tmp_6 * tmp_4
    tmp_6 = tmp_4 = None
    tmp_8 = tmp_2 + tmp_7
    tmp_2 = tmp_7 = None
    tmp_9 = in_1.unsqueeze(-1)
    tmp_1 = None
    tmp_10 = tmp_9.unsqueeze(-1)
    tmp_9 = None
    return (tmp_8, tmp_10)

def replacement_args(in_2, in_0, in_1):
    return (in_2, in_0, in_1)

@triton.jit
def feature_modulation_kernel(
    features_ptr,
    scale_ptr,
    output_ptr,
    C,      # channels
    H,      # height
    W,      # width
    N,      # batch size
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
    
    # Load features [C, H, W]
    base_offset = batch_offset * C * H * W + m_offset * H * W
    feature_offsets = (column_offsets[:, None] * H + tl.arange(0, H)[None, :]).to(tl.int64)
    
    # Load scale [C] -> [C, 1, 1]
    scale = tl.load(scale_ptr + scale_offsets, mask=m_mask, other=0.0)
    scale = scale[:, None, None]
    
    # Load features
    features = tl.load(features_ptr + base_offset + feature_offsets, mask=(m_mask[:, None] & n_mask[None, :] & batch_mask), other=0.0)
    
    # Compute ReLU
    features_relu = tl.maximum(features, 0.0)
    
    # Compute avg_pool2d with 3x3 kernel, stride 1, padding 1
    # Create padded input implicitly by handling boundaries
    relu_col = tl.arange(0, BLOCK_SIZE_N)
    relu_row = tl.arange(0, H)
    relu_col_padded = relu_col + 1
    relu_row_padded = relu_row + 1
    
    pad_mask_col = (relu_col_padded >= 1) & (relu_col_padded < W)
    pad_mask_row = (relu_row_padded >= 1) & (relu_row_padded < H)
    
    # Compute pooling with proper boundary handling
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_col in range(3):
        for k_row in range(3):
            col_idx = relu_col_padded[:, None] - k_col
            row_idx = relu_row_padded[None, :] - k_row
            
            valid_col = col_idx >= 0
            valid_row = row_idx >= 0
            valid = valid_col & valid_row
            
            pooled_features = tl.where(valid, 
                features_relu[:, :, col_idx % W, row_idx % H], 
                0.0)
            acc += pooled_features
    relu_pooled = acc / 9.0
    
    # Compute subtraction
    diff_features = relu_pooled - features_relu
    
    # Apply scale modulation
    modulated_features = features_relu + diff_features * scale
    
    # Store result
    output_base_offset = batch_offset * C * H * W + m_offset * H * W
    tl.store(output_ptr + output_base_offset + feature_offsets, modulated_features, 
             mask=(m_mask[:, None] & n_mask[None, :] & batch_mask))

@torch.fx.wrap
def feature_modulation_optimized_torch(in_2, in_0, in_1):
    # Get tensor shapes
    N, C, H, W = in_2.shape
    scale_size_1 = in_0.shape[0]
    scale_size_2 = in_1.shape[0]
    
    # Create output tensors
    output_features = torch.empty_like(in_2)  # tmp_8
    output_expand = torch.empty((scale_size_2, 1, 1), dtype=in_1.dtype, device=in_1.device)  # tmp_10
    
    # Optimized kernel for feature modulation (first path)
    BLOCK_SIZE_M = 64  # Process multiple channels at once
    BLOCK_SIZE_N = 64  # Process multiple width positions at once
    
    # Calculate grid dimensions
    grid_m = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_batch = N
    
    # Launch feature modulation kernel
    feature_modulation_kernel[(grid_m, grid_n, grid_batch)](
        features_ptr=in_2,
        scale_ptr=in_0,
        output_ptr=output_features,
        C=C,
        H=H,
        W=W,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Simple expansion for in_1 (second path)
    output_expand = in_1.unsqueeze(-1).unsqueeze(-1)
    
    return (output_features, output_expand)

def replacement_func():
    return feature_modulation_optimized_torch