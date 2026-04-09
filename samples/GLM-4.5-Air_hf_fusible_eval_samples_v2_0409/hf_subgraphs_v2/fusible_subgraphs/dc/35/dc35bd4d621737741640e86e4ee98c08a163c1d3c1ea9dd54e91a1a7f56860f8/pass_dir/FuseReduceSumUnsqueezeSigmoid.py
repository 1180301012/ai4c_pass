import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern for element-wise multiply + reduce-sum(dim=1) + unsqueeze(1) + sigmoid"""
    tmp_0 = x * y
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1) 
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

def replacement_args(x, y):
    """Extract arguments for the replacement function"""
    return x, y

@triton.jit
def fused_kernel(
    x_ptr, y_ptr, out_ptr,
    n_batch, n_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    """Optimized fused kernel: vectorized memory access"""
    # Program ids
    batch_pid = tl.program_id(0)
    spatial_pid = tl.program_id(1)
    
    # Each block processes multiple batch elements
    batch_start = batch_pid * BLOCK_SIZE_M
    batch_idx = batch_start + tl.arange(0, BLOCK_SIZE_M)
    batch_mask = batch_idx < n_batch
    
    # Process spatial locations in chunks
    spatial_start = spatial_pid * BLOCK_SIZE_C
    spatial_idx = spatial_start + tl.arange(0, BLOCK_SIZE_C)
    spatial_mask = spatial_idx < (height * width)
    
    # Convert spatial indices to 2D coordinates
    w_ids = spatial_idx % width
    h_ids = spatial_idx // width
    
    # Initialize accumulator for reduction
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_C], dtype=tl.float32)
    
    # Process channels sequentially for all combinations
    for c in range(n_channels):
        # Compute pointers for current channel
        # offset = batch * n_channels * H * W + c * H * W + h * W + w
        x_ptrs = x_ptr[:, None] + batch_idx[:, None] * n_channels * height * width + \
                 c * height * width + \
                 h_ids[None, :] * width + w_ids[None, :]
        
        y_ptrs = y_ptr[:, None] + batch_idx[:, None] * n_channels * height * width + \
                 c * height * width + \
                 h_ids[None, :] * width + w_ids[None, :]
        
        # Load x and y values with mask
        x_loaded = tl.load(x_ptrs, mask=batch_mask[:, None] & spatial_mask[None, :], other=0.0)
        y_loaded = tl.load(y_ptrs, mask=batch_mask[:, None] & spatial_mask[None, :], other=0.0)
        
        # Accumulate element-wise product
        acc += x_loaded * y_loaded
    
    # Apply sigmoid activation
    out = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store results: output shape [batch, 1, height, width]
    out_ptrs = out_ptr[:, None] + batch_idx[:, None] * height * width + \
               h_ids[None, :] * width + w_ids[None, :]
    tl.store(out_ptrs, out, mask=batch_mask[:, None] & spatial_mask[None, :])

@torch.fx.wrap
def fused_operation(x, y):
    """Wrapper function for the fused kernel"""
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
    
    # Get input shapes
    batch_size, channels, height, width = x.shape
    
    # Optimized block sizes based on tensor dimensions for best performance
    spatial_total = height * width
    if spatial_total <= 256:
        BLOCK_SIZE_M = 32      # Small tensors: moderate batch processing
        BLOCK_SIZE_C = 16      # Small tensors: fewer spatial locations per thread
    elif spatial_total <= 1024:
        BLOCK_SIZE_M = 16      # Medium tensors: less batch parallelism
        BLOCK_SIZE_C = 32      # Medium tensors: more spatial parallelism
    elif spatial_total <= 4096:
        BLOCK_SIZE_M = 8       # Larger tensors: minimal batch parallelism
        BLOCK_SIZE_C = 64      # Larger tensors: maximum spatial parallelism
    else:
        BLOCK_SIZE_M = 4       # Very large tensors: single batch elements
        BLOCK_SIZE_C = 128     # Very large tensors: maximal spatial parallelism
    
    # Calculate grid dimensions:
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (height * width + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Create output tensor with correct shape [batch_size, 1, height, width]
    if x.dtype == torch.bfloat16:
        out = torch.empty(batch_size, 1, height, width, dtype=torch.bfloat16, device=x.device)
    else:
        out = torch.empty(batch_size, 1, height, width, dtype=x.dtype, device=x.device)
    
    # Launch kernel with 2D grid
    fused_kernel[(grid_m, grid_n)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_batch=batch_size,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out

def replacement_func():
    """Return the fused operation function"""
    return fused_operation