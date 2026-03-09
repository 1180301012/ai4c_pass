import torch
import triton
import triton.language as tl

# Pattern matching function for hardtanh operation
def pattern(arg1, arg2, arg3, hardtanh_input):
    # Match hardtanh operation with specific parameters
    tmp_3 = torch.nn.functional.hardtanh(hardtanh_input, 0.0, 6.0, False)
    return tmp_3

# Argument extraction function
def replacement_args(arg1, arg2, arg3, hardtanh_input):
    return (arg1, arg2, arg3, hardtanh_input)

# Optimized hardtanh kernel using Triton
@triton.jit
def hardtanh_kernel(
    x_ptr,               # Input tensor pointer
    out_ptr,             # Output tensor pointer  
    N, C, H, W,          # Tensor dimensions
    BLOCK_SIZE_C: tl.constexpr,  # Channels per program
    BLOCK_SIZE_H: tl.constexpr,  # Height per program
    BLOCK_SIZE_W: tl.constexpr,  # Width per program
):
    # Get program indices
    c_idx = tl.program_id(0)  # Channel tile
    h_idx = tl.program_id(1)  # Height tile
    w_idx = tl.program_id(2)  # Width tile
    b_idx = tl.program_id(3)  # Batch tile
    
    # Calculate coordinate ranges
    c_off = c_idx * BLOCK_SIZE_C
    h_off = h_idx * BLOCK_SIZE_H
    w_off = w_idx * BLOCK_SIZE_W
    b_off = b_idx * 1  # Single batch program
    
    # Create masks for validity
    mask_c = c_off + tl.arange(0, BLOCK_SIZE_C) < C
    mask_h = h_off + tl.arange(0, BLOCK_SIZE_H) < H
    mask_w = w_off + tl.arange(0, BLOCK_SIZE_W) < W
    mask_b = b_off < N
    
    # Load input data [BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W]
    x_data = tl.load(
        x_ptr + (b_off * C * H * W + 
                (c_off + tl.arange(0, BLOCK_SIZE_C))[:, None, None] * H * W +
                (h_off + tl.arange(0, BLOCK_SIZE_H))[None, :, None] * W +
                (w_off + tl.arange(0, BLOCK_SIZE_W))[None, None, :]),
        mask=(mask_b & mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]),
        other=0.0
    ).to(tl.float32)
    
    # Apply hardtanh: f(x) = max(0, min(6, x))
    # Equivalent to: 0.0 if x < 0.0, x if 0.0 <= x <= 6.0, 6.0 if x > 6.0
    out_data = tl.where(x_data < 0.0, 0.0, tl.where(x_data > 6.0, 6.0, x_data))
    
    # Store result
    out_idx = (b_off * C * H * W + 
              (c_off + tl.arange(0, BLOCK_SIZE_C))[:, None, None] * H * W +
              (h_off + tl.arange(0, BLOCK_SIZE_H))[None, :, None] * W +
              (w_off + tl.arange(0, BLOCK_SIZE_W))[None, None, :])
    
    tl.store(
        out_ptr + out_idx,
        out_data,
        mask=(mask_b & mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :])
    )

# Kernel wrapper
@torch.fx.wrap
def optimized_hardtanh(input_tensor):
    # Get input dimensions
    N, C, H, W = input_tensor.shape
    
    # Output tensor shape (same as input)
    output = torch.empty_like(input_tensor)
    
    # Block sizes - optimized for typical GPU architectures
    BLOCK_SIZE_C = 64   # Channels per program
    BLOCK_SIZE_H = 8    # Height per program
    BLOCK_SIZE_W = 8    # Width per program
    
    # Calculate grid dimensions
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_b = N  # Each batch is processed by separate programs
    
    # Launch kernel
    hardtanh_kernel[(grid_c, grid_h, grid_w, grid_b)](
        input_tensor,
        output,
        N, C, H, W,
        BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_hardtanh