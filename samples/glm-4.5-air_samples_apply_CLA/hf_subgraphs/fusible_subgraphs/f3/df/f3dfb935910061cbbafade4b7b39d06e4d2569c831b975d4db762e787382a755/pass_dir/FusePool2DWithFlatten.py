import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_3 = tmp_2.flatten(1, -1)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_pool_flatten_kernel(
    x_ptr, 
    out_ptr, 
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute spatial mean directly: for each sample and channel, compute mean over H x W
    n_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Output is [N, C], one value per sample per channel
    out_idx = n_idx * C + c_idx
    mask = (out_idx < N * C)
    
    if mask:
        spatial_sum = 0.0
        spatial_count = 0.0
        
        # Simple approach: iterate spatial locations (for small spatial dimensions this works well)
        # For larger spatial dimensions, we might want a more sophisticated approach,
        # but given our inputs are 7x7 or 8x8, this is reasonable
        for h in range(H):
            for w in range(W):
                spatial_idx = (n_idx * C * H * W + 
                              c_idx * H * W + 
                              h * W + w)
                
                # Always in bounds since we're iterating within the valid tensor dimensions
                value = tl.load(x_ptr + spatial_idx)
                spatial_sum += value
                spatial_count += 1.0
        
        # Compute mean
        spatial_mean = spatial_sum / spatial_count if spatial_count > 0 else 0.0
        
        # Store result
        tl.store(out_ptr + out_idx, spatial_mean)

@torch.fx.wrap  
def fused_pool_flatten(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    # For small spatial dimensions, use grid launch with N x C
    # Block sizes based on typical GPU grid dimensions
    BLOCK_SIZE_N = 512  # Number of samples per block
    BLOCK_SIZE_C = 256  # Number of channels per block
    
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(C, BLOCK_SIZE_C))
    
    fused_pool_flatten_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out

def replacement_func():
    return fused_pool_flatten