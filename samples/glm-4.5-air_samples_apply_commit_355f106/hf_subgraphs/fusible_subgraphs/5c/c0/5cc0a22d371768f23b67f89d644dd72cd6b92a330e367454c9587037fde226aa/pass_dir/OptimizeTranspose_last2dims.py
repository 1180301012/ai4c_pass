import torch
import triton
import triton.language as tl

# Pattern matching function: transpose last two dimensions
def pattern(x):
    return x.transpose(-2, -1)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel for transpose operation with tiling
@triton.jit
def transpose_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)  # Batch and channel: N * C
    pid_h = tl.program_id(1)  # Tile row
    pid_w = tl.program_id(2)  # Tile column
    
    # Calculate block
    block_h = pid_h * BLOCK_SIZE_HW
    block_w = pid_w * BLOCK_SIZE_HW
    
    # Process within this tile
    for i in range(block_h, min(block_h + BLOCK_SIZE_HW, H)):
        for j in range(block_w, min(block_w + BLOCK_SIZE_HW, W)):
            # Calculate thread index within tile
            local_idx = (i - block_h) * BLOCK_SIZE_HW + (j - block_w)
            
            # Calculate offsets for this element
            offset_n = pid_n // C
            offset_c = pid_n % C
            
            # Transpose coordinates: swap H and W dimensions
            offset_h_trans = j
            offset_w_trans = i
            
            # Calculate input and output offsets
            in_offset = offset_n * (C * H * W) + offset_c * (H * W) + i * W + j
            out_offset = offset_n * (C * W * H) + offset_c * (W * H) + offset_h_trans * H + offset_w_trans
            
            # Check bounds (should be safe due to tile bounds, but just in case)
            mask = (in_offset < N * C * H * W) & (out_offset < N * C * W * H)
            
            if mask:
                # Load input value and store transposed
                val = tl.load(x_ptr + in_offset, mask=mask)
                tl.store(out_ptr + out_offset, val, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_transpose(x):
    # Input x shape: [N, C, H, W] -> [1, 16, 196, 48]
    # Expected output shape: [N, C, W, H] -> [1, 16, 48, 196]
    
    N, C, H, W = x.shape
    
    # Output shape after transpose: [N, C, W, H]
    output = torch.empty((N, C, W, H), device=x.device, dtype=x.dtype)
    
    # Triton kernel launch parameters
    BLOCK_SIZE_C = 32   # Block size for channels dimension
    BLOCK_SIZE_HW = 128 # Block size for height/width dimensions
    
    # Use 2D grid for better GPU occupancy
    # Grid: (N * C, (H + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW, (W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)
    grid = (N * C, (H + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW, (W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)
    
    # Launch kernel
    transpose_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_transpose