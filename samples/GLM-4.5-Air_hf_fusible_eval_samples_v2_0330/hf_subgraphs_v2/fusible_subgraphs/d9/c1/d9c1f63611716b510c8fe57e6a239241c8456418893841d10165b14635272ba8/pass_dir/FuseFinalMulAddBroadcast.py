import torch
import triton
import triton.language as tl

def pattern(tmp_10, tmp_11, tmp_12, tmp_13):
    """
    Pattern matches the final computation sequence:
    - tmp_12.unsqueeze(-2) for broadcasting
    - tmp_11 * tmp_14 (where tmp_14 = tmp_12.unsqueeze(-2))
    - tmp_10 * tmp_13
    - tmp_15 + tmp_16 (where tmp_15 = tmp_11 * tmp_14, tmp_16 = tmp_10 * tmp_13)
    """
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17

def replacement_args(tmp_10, tmp_11, tmp_12, tmp_13):
    return (tmp_10, tmp_11, tmp_12, tmp_13)

@triton.jit
def fused_mul_add_broadcast_kernel(
    tmp_10_ptr, tmp_11_ptr, tmp_12_ptr, tmp_13_ptr,
    out_ptr,
    n_batch, n_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance kernel with vectorized memory access for better memory coalescing.
    Uses 2D grid where each program handles multiple elements for better GPU occupancy.
    """
    # Get 2D program IDs for better parallelism
    batch_id = tl.program_id(0)
    feat_block_id = tl.program_id(1)
    
    # Calculate feature range for this program
    feat_block_start = feat_block_id * BLOCK_SIZE
    feat_block_end = min(feat_block_start + BLOCK_SIZE, n_features)
    
    # Calculate batch range (broadcasting means we process all features for one batch)
    batch_start = batch_id
    batch_end = batch_id + 1
    
    # Process features with optimized memory access
    for feat_idx in range(feat_block_start, feat_block_end):
        # Calculate global memory offset
        offset = batch_start * n_features + feat_idx
        
        # Load all inputs with optimized bounds checking
        mask = batch_start < n_batch
        tmp_10_val = tl.load(tmp_10_ptr + offset, mask=mask, other=0.0)
        tmp_11_val = tl.load(tmp_11_ptr + offset, mask=mask, other=0.0)
        tmp_13_val = tl.load(tmp_13_ptr + offset, mask=mask, other=0.0)
        tmp_12_val = tl.load(tmp_12_ptr + offset, mask=mask, other=0.0)
        
        # Optimized computation order for better register usage
        mul1 = tmp_11_val * tmp_12_val
        mul2 = tmp_10_val * tmp_13_val
        result = mul1 + mul2
        
        # Store result
        tl.store(out_ptr + offset, result, mask=mask)

@torch.fx.wrap
def fused_mul_add_broadcast(tmp_10, tmp_11, tmp_12, tmp_13):
    """
    Wrapper function for the fused kernel with optimized 2D grid
    """
    # Handle different input shapes and extract dimensions
    if len(tmp_10.shape) == 3:  # [300, 1, 256]
        n_batch, _, n_features = tmp_10.shape
    else:  # [300, 256] or [300, n_features]
        n_batch, n_features = tmp_10.shape
    
    # Create output tensor with same shape as input
    out = torch.empty_like(tmp_10)
    
    # Set up 2D grid configuration
    BLOCK_SIZE = 128  # Intermediate block size for better balance
    
    # Calculate grid dimensions:
    # - First dimension: batches (each program processes one batch)
    # - Second dimension: feature blocks (each program processes BLOCK_SIZE features)
    grid_y = n_batch
    grid_x = (n_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with 2D grid
    fused_mul_add_broadcast_kernel[(grid_y, grid_x)](
        tmp_10_ptr=tmp_10,
        tmp_11_ptr=tmp_11,
        tmp_12_ptr=tmp_12,
        tmp_13_ptr=tmp_13,
        out_ptr=out,
        n_batch=n_batch,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_mul_add_broadcast