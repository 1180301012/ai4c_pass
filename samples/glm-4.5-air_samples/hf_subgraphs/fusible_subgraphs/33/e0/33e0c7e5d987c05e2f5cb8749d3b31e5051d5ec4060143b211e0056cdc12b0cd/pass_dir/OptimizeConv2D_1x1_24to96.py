import torch
import triton
import triton.language as tl

# Pattern matching function for conv2d operation - direct match
def pattern(bias, weight, input_tensor, _):
    # Direct match of conv2d without intermediate assignments
    result = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), groups=1)
    return result

# Argument extraction function
def replacement_args(bias, weight, input_tensor, _):
    return (bias, weight, input_tensor, None)

# Optimized 1x1 convolution kernel using Triton
@triton.jit
def conv2d_1x1_kernel(
    x_ptr,           # Input tensor [N, C_in, H, W]
    w_ptr,           # Weight tensor [C_out, C_in, 1, 1]
    b_ptr,           # Bias tensor [C_out]
    out_ptr,         # Output tensor [N, C_out, H, W]
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr,  # Channels per program
    BLOCK_SIZE_N: tl.constexpr,  # Output channels per program
    BLOCK_SIZE_H: tl.constexpr,  # Height per program
    BLOCK_SIZE_W: tl.constexpr,  # Width per program
):
    # Get program indices
    m_idx = tl.program_id(0)  # Batch and input channel tile
    n_idx = tl.program_id(1)  # Output channel tile
    h_idx = tl.program_id(2)  # Height tile
    w_idx = tl.program_id(3)  # Width tile
    
    # Calculate coordinate ranges
    m_off = m_idx * BLOCK_SIZE_M
    n_off = n_idx * BLOCK_SIZE_N
    h_off = h_idx * BLOCK_SIZE_H
    w_off = w_idx * BLOCK_SIZE_W
    
    # Create masks for validity
    mask_m = m_off + tl.arange(0, BLOCK_SIZE_M) < N * C_in
    mask_n = n_off + tl.arange(0, BLOCK_SIZE_N) < C_out
    mask_h = h_off + tl.arange(0, BLOCK_SIZE_H) < H
    mask_w = w_off + tl.arange(0, BLOCK_SIZE_W) < W
    
    # Reshape for efficient access
    x_ptr = x_ptr + m_off * H * W + n_off // C_in * H * W  # Simplified for 1x1 conv
    
    # For 1x1 convolution, we can treat it as a matrix multiplication
    # Compute dot products: output[n, m, h, w] = sum_c (input[n, c, h, w] * weight[n, c, m])
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over input channels
    for c_offset in range(0, C_in, BLOCK_SIZE_M):
        c_end = min(c_offset + BLOCK_SIZE_M, C_in)
        
        # Load input slice [BLOCK_SIZE_M, H, W]
        x_slice = tl.load(
            x_ptr + (c_offset // C_in) * BLOCK_SIZE_M * H * W + 
            (tl.arange(0, BLOCK_SIZE_M)[:, None, None] * H * W + 
             tl.arange(0, BLOCK_SIZE_H)[None, :, None] * W + 
             tl.arange(0, BLOCK_SIZE_W)[None, None, :]),
            mask=mask_m[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :],
            other=0.0
        ).to(tl.float32)
        
        # Load weight slice [C_out, BLOCK_SIZE_M]
        w_slice = tl.load(
            w_ptr + (tl.arange(0, BLOCK_SIZE_N)[:, None] * C_in + 
                     tl.arange(0, min(c_end - c_offset, BLOCK_SIZE_M))[None, :]),
            mask=mask_n[:, None] & (tl.arange(0, min(c_end - c_offset, BLOCK_SIZE_M))[None, :] < C_in),
            other=0.0
        ).to(tl.float32)
        
        # Compute matrix multiplication: output = x_slice @ w_slice.T
        # For spatial positions, we need to accumulate
        for h in range(BLOCK_SIZE_H):
            for w in range(BLOCK_SIZE_W):
                # Get spatial element
                x_spatial = x_slice[:, h, w]
                # Multiply with weights and accumulate
                acc += tl.dot(x_spatial, w_slice[n_off % C_out, :c_end - c_offset])
    
    # Load bias
    bias_val = tl.load(
        b_ptr + tl.arange(0, BLOCK_SIZE_N),
        mask=mask_n,
        other=0.0
    ).to(tl.float32)
    
    # Add bias
    acc += bias_val
    
    # Store output
    out_idx = n_off * H * W + h_off * W + w_off + \
              tl.arange(0, BLOCK_SIZE_N)[:, None] * H * W + \
              tl.arange(0, BLOCK_SIZE_H)[None, :, None] * W + \
              tl.arange(0, BLOCK_SIZE_W)[None, None, :]
    
    tl.store(
        out_ptr + out_idx,
        acc[:, None, None],
        mask=mask_n[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    )

# Kernel wrapper
@torch.fx.wrap
def optimized_conv2d_1x1(bias, weight, input_tensor):
    # Get input dimensions
    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    
    # Output tensor shape
    output_shape = (N, C_out, H, W)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block sizes - optimized for typical GPU architectures
    BLOCK_SIZE_M = 16  # Input channels/tile
    BLOCK_SIZE_N = 32  # Output channels/tile  
    BLOCK_SIZE_H = 8   # Height/tile
    BLOCK_SIZE_W = 8   # Width/tile
    
    # Calculate grid dimensions
    grid_m = (N * C_in + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    conv2d_1x1_kernel[(grid_m, grid_n, grid_h, grid_w)](
        input_tensor,
        weight,
        bias,
        output,
        N, C_in, H, W, C_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_conv2d_1x1