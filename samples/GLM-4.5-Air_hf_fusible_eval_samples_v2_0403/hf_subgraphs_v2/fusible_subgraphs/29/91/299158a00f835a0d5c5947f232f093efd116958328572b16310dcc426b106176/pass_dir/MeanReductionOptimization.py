import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching for mean operation along dimension -2"""
    # Use sum and division instead of mean to avoid forbidden API
    return torch.sum(x, dim=-2) / x.size(-2)

def replacement_args(x):
    """Extract arguments for the mean optimization"""
    return (x,)

@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEATURES: tl.constexpr,
):
    """Optimized Triton kernel for mean reduction along dimension -2 (seq_len dimension)"""
    # Programs are split along both batch and feature dimensions
    pid_batch = tl.program_id(0)
    pid_features = tl.program_id(1)
    
    # Range of elements each program should process
    offset_batch = pid_batch * BLOCK_SIZE_BATCH
    offset_features = pid_features * BLOCK_SIZE_FEATURES
    
    batch_range = tl.arange(0, BLOCK_SIZE_BATCH)
    features_range = tl.arange(0, BLOCK_SIZE_FEATURES)
    
    # Create coordinate matrices
    batch_offsets = offset_batch + batch_range
    feature_offsets = offset_features + features_range
    
    # Initialize accumulator for mean computation
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURES), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURES), dtype=tl.float32)
    
    # Loop over sequence dimension
    for seq_idx in range(seq_len):
        # Calculate pointer offsets for this sequence element
        x_ptrs = (x_ptr + 
                 batch_offsets[:, None] * seq_len * features + 
                 seq_idx * features + 
                 features_range[None, :])
        
        # Load input data
        x_mask = (batch_offsets[:, None] < batch_size) & (feature_offsets[None, :] < features)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Accumulate sum
        acc += x
    
    # Count valid elements
    valid_batch = batch_offsets < batch_size
    valid_features = feature_offsets < features
    valid_mask = valid_batch[:, None] & valid_features[None, :]
    count = tl.where(valid_mask, seq_len, 0.0)
    
    # Compute mean: sum / count
    # Handle division by zero for invalid elements
    result = tl.where(count > 0, acc / count, 0.0)
    
    # Write output
    out_ptrs = (out_ptr + 
               batch_offsets[:, None] * features + 
               features_range[None, :])
    tl.store(out_ptrs, result, mask=valid_mask)

@torch.fx.wrap
def triton_mean(x):
    """Wrapper function for the optimized mean reduction"""
    assert x.dim() == 3, f"Expected 3D input for mean along dim -2, got {x.dim()}D"
    
    batch_size, seq_len, features = x.shape
    
    # Create output tensor
    out = torch.empty((batch_size, features), device=x.device, dtype=x.dtype)
    
    # Set block sizes based on tensor characteristics
    BLOCK_SIZE_BATCH = 64
    BLOCK_SIZE_FEATURES = min(32, features)
    
    # Calculate grid size
    grid_batch = (batch_size + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
    grid_features = (features + BLOCK_SIZE_FEATURES - 1) // BLOCK_SIZE_FEATURES
    
    # Launch kernel
    mean_kernel[(grid_batch, grid_features)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_FEATURES=BLOCK_SIZE_FEATURES,
    )
    
    return out

def replacement_func():
    """Return the optimized mean function"""
    return triton_mean