import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0, in_1):
    # The exact operations from the model
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel that fuses softmax + multiplication + sum along dim=1
@triton.jit
def optimized_softmax_mul_sum_kernel(
    in_0_ptr,  # [1, 2, 256, H, W]
    in_1_ptr,  # [1, 2, 256, 1, 1]  
    out_ptr,   # [1, 256, H, W]
    n_channels,
    h_dim,
    w_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, 
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles a block of output elements
    m_block = tl.program_id(0) * BLOCK_SIZE_M  # channel block start
    h_block = tl.program_id(1) * BLOCK_SIZE_H  # height block start  
    w_block = tl.program_id(2) * BLOCK_SIZE_W  # width block start
    
    # Process within the block bounds
    for m in range(m_block, min(m_block + BLOCK_SIZE_M, n_channels)):
        for h in range(h_block, min(h_block + BLOCK_SIZE_H, h_dim)):
            for w in range(w_block, min(w_block + BLOCK_SIZE_W, w_dim)):
                # Load two softmax inputs along dim=1
                val1_0 = tl.load(in_1_ptr + (0 * n_channels + m) * 1 * 1)  # [0, channel, 0, 0]
                val1_1 = tl.load(in_1_ptr + (1 * n_channels + m) * 1 * 1)  # [1, channel, 0, 0]
                
                # Find max for numerical stability
                max_val = tl.maximum(val1_0, val1_1)
                
                # Compute softmax weights
                exp_val1_0 = tl.exp(val1_0 - max_val)
                exp_val1_1 = tl.exp(val1_1 - max_val)
                total_weight = exp_val1_0 + exp_val1_1
                
                # Load corresponding in_0 values and compute weighted sum
                val0_0 = tl.load(in_0_ptr + (0 * n_channels + m) * h_dim * w_dim + h * w_dim)
                val0_1 = tl.load(in_0_ptr + (1 * n_channels + m) * h_dim * w_dim + h * w_dim)
                
                weighted_sum = exp_val1_0 * val0_0 + exp_val1_1 * val0_1
                
                # Normalize and store result
                result = weighted_sum / total_weight
                tl.store(out_ptr + (m * h_dim * w_dim + h * w_dim + w), result)



@torch.fx.wrap
def fused_softmax_mul_sum_triton(in_0, in_1):
    # Get input shapes
    shape_0 = in_0.shape
    shape_1 = in_1.shape
    
    n_channels = shape_0[2]  # 256
    h_dim = shape_0[3]       # Height (32 or 8)
    w_dim = shape_0[4]       # Width (32 or 8)
    
    # Output shape: [1, 256, H, W]
    out_shape = [1, n_channels, h_dim, w_dim]
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Choose optimal block sizes based on problem size
    if h_dim >= 32 and w_dim >= 32:
        # Large feature maps: use larger blocks for better occupancy
        BLOCK_SIZE_M = 16  # Process 16 channels per program
        BLOCK_SIZE_H = 8   # Process 8x8 spatial block per program
        BLOCK_SIZE_W = 8
    else:
        # Small feature maps: use smaller blocks to avoid over-subscription
        BLOCK_SIZE_M = 8   # Process 8 channels per program  
        BLOCK_SIZE_H = 4   # Process 4x4 spatial block per program
        BLOCK_SIZE_W = 4
    
    # Calculate grid dimensions based on block sizes
    grid_m = (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_h = (h_dim + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (w_dim + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (grid_m, grid_h, grid_w)
    
    # Launch kernel
    optimized_softmax_mul_sum_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_channels=n_channels,
        h_dim=h_dim,
        w_dim=w_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_H=BLOCK_SIZE_H, 
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_softmax_mul_sum_triton