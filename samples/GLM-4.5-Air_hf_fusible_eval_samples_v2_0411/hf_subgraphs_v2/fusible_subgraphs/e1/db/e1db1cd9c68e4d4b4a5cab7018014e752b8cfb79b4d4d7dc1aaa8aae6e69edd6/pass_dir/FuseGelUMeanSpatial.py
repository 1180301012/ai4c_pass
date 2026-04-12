import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the model computation
def pattern(in_0 : torch.Tensor):
    tmp_0 = torch.nn.functional.gelu(in_0);  in_0 = None
    tmp_1 = tmp_0.mean((2, 3), keepdim = True)
    return (tmp_0, tmp_1)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel for GELU + Mean reduction
@triton.jit
def fused_gelu_mean_kernel(
    x_ptr,           # Input tensor pointer
    gelu_out_ptr,    # GELU output tensor pointer  
    mean_out_ptr,    # Mean output tensor pointer
    N,               # Batch size (first dimension)
    C,               # Channels (second dimension)
    H,               # Height (third dimension)
    W,               # Width (fourth dimension)
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
):
    # Compute program IDs - simpler approach: one program per (batch, channel) pair
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Check bounds for this program
    if pid_n >= N or pid_c >= C:
        return
    
    # Initialize sum for mean calculation - use register memory for simplicity and accuracy
    spatial_sum = 0.0
    spatial_elements = H * W
    
    # Process spatial tiles efficiently
    for h_offset in range(0, H, TILE_SIZE_H):
        for w_offset in range(0, W, TILE_SIZE_W):
            # Compute tile bounds
            h_start = h_offset
            h_end = tl.minimum(h_start + TILE_SIZE_H, H)
            w_start = w_offset
            w_end = tl.minimum(w_start + TILE_SIZE_W, W)
            
            # Process the current tile for this batch/channel pair
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    # Compute global indices  
                    idx = pid_n * C * H * W + pid_c * H * W + h * W + w
                    
                    # Load input value
                    x_val = tl.load(x_ptr + idx)
                    
                    # Cast to float32 for GELU computation for precision
                    x_val_float = tl.cast(x_val, tl.float32)
                    
                    # Apply GELU operation using stable sigmoid approximation
                    # GELU(x) ≈ x * sigmoid(1.702 * x) for better numerical stability
                    sigmoid_val = 1.0 / (1.0 + tl.exp(-1.702 * x_val_float))
                    gelu_val_float = x_val_float * sigmoid_val
                    
                    # Cast back to original dtype for storage
                    original_dtype = tl.float32 if x_val.dtype == tl.float32 else tl.bfloat16 if x_val.dtype == tl.bfloat16 else tl.float16
                    gelu_val = tl.cast(gelu_val_float, original_dtype)
                    
                    # Store GELU output
                    tl.store(gelu_out_ptr + idx, gelu_val)
                    
                    # Accumulate for mean calculation
                    spatial_sum += gelu_val_float
    
    # Compute mean for this batch/channel pair
    mean_val = spatial_sum / spatial_elements
    
    # Store mean output - result has shape [N, C, 1, 1]
    mean_idx = pid_n * C + pid_c
    tl.store(mean_out_ptr + mean_idx, mean_val)

# Kernel wrapper decorated with @torch.fx.wrap as required
@torch.fx.wrap
def fused_gelu_mean_wrapper(in_0):
    N, C, H, W = in_0.shape
    
    # Allocate output tensors
    gelu_out = torch.empty_like(in_0)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Configuration for optimal GPU occupancy
    TILE_SIZE_H = 32  # Tile size for height dimension
    TILE_SIZE_W = 32  # Tile size for width dimension
    
    # Calculate grid dimensions - one program per (batch, channel) pair
    grid_n = N
    grid_c = C
    
    # Launch the fused kernel
    fused_gelu_mean_kernel[(grid_n, grid_c)](
        x_ptr=in_0,
        gelu_out_ptr=gelu_out,
        mean_out_ptr=mean_out,
        N=N,
        C=C,
        H=H,
        W=W,
        TILE_SIZE_H=TILE_SIZE_H,
        TILE_SIZE_W=TILE_SIZE_W,
    )
    
    return (gelu_out, mean_out)

# Replacement function - returns the kernel wrapper function
def replacement_func():
    return fused_gelu_mean_wrapper