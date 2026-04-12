import torch
import triton
import triton.language as tl

# Pattern matching function for mean reduction only
def pattern(input_tensor):
    # Match the mean reduction operation that follows GELU
    return input_tensor.mean((2, 3), keepdim=True)

def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized Triton kernel for spatial mean reduction
@triton.jit
def mean_spatial_kernel(
    x_ptr,           # Input tensor pointer
    out_ptr,         # Output tensor pointer  
    N,               # Batch size
    C,               # Channels  
    H,               # Height
    W,               # Width
    TILE_SIZE: tl.constexpr,
):
    # Program IDs: one program per (batch, channel) pair
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Check bounds
    if pid_n >= N or pid_c >= C:
        return
    
    # Initialize sum accumulator
    spatial_sum = 0.0
    spatial_elements = H * W
    
    # Process spatial tiles efficiently
    for h_offset in range(0, H, TILE_SIZE):
        for w_offset in range(0, W, TILE_SIZE):
            # Compute tile bounds
            h_start = h_offset
            h_end = tl.minimum(h_start + TILE_SIZE, H)
            w_start = w_offset  
            w_end = tl.minimum(w_start + TILE_SIZE, W)
            
            # Process current tile
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    # Compute global index for this (n, c, h, w) location
                    idx = pid_n * C * H * W + pid_c * H * W + h * W + w
                    
                    # Load input value
                    x_val = tl.load(x_ptr + idx)
                    
                    # Cast to float32 for precise summation
                    x_float = tl.cast(x_val, tl.float32)
                    
                    # Accumulate sum
                    spatial_sum += x_float
    
    # Compute mean
    mean_val = spatial_sum / spatial_elements
    
    # Store result - output should have shape [N, C, 1, 1]
    out_idx = pid_n * C + pid_c
    tl.store(out_ptr + out_idx, mean_val)

@torch.fx.wrap  
def mean_spatial_wrapper(x):
    N, C, H, W = x.shape
    
    # Allocate output tensor
    output = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    
    # Configuration for optimal GPU utilization
    TILE_SIZE = 32  # Tile size for spatial dimensions
    
    # Calculate grid dimensions - one program per (batch, channel) pair
    grid_n = N
    grid_c = C
    
    # Launch the optimized kernel
    mean_spatial_kernel[(grid_n, grid_c)](
        x_ptr=x,
        out_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        TILE_SIZE=TILE_SIZE,
    )
    
    return output

def replacement_func():
    return mean_spatial_wrapper