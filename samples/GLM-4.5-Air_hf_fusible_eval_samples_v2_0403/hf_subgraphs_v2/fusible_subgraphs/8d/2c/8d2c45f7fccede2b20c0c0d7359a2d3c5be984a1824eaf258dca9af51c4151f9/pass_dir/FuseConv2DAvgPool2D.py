import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_1, in_0):
    """
    Conv2D followed by AvgPool2D pattern that matches the exact model computation
    """
    # tmp_0 = in_0 (weight tensor assignment)
    tmp_0 = in_0
    # tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    # tmp_0 = None (cleanup - NOT included in pattern)
    # tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Simple Triton kernel for fused Conv2D + AvgPool2D
@triton.jit
def simple_fused_conv_pool_kernel(
    x_ptr,  # Input tensor [N, C_in, H_in, W_in]
    weight_ptr,  # Weight tensor [C_out, C_in, 1, 1]
    out_ptr,  # Output tensor [N, C_out, H_out, W_out]
    N, C_in, C_out, H_in, W_in,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_H: tl.constexpr,  # Block size for height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for width
):
    # Calculate program IDs
    m = tl.program_id(0)  # Output channel block
    h = tl.program_id(1)  # Height block  
    w = tl.program_id(2)  # Width block
    
    # For simplicity, handle first batch only (N=1)
    n = 0
    
    # Calculate output coordinates
    out_h_base = h * BLOCK_SIZE_H
    out_w_base = w * BLOCK_SIZE_W
    
    # Calculate safe ranges for processing
    c_end_block = min(BLOCK_SIZE_M, C_out - m * BLOCK_SIZE_M)
    h_end_block = min(BLOCK_SIZE_H, H_in // 2 - out_h_base)
    w_end_block = min(BLOCK_SIZE_W, W_in // 2 - out_w_base)
    
    # Process each output position
    for c_off in range(c_end_block):
        for i in range(h_end_block):
            for j in range(w_end_block):
                # Average pooling with 2x2 kernel
                pool_sum = 0.0
                pool_count = 0
                
                # Sample 2x2 region from input
                input_h_base = out_h_base * 2 + i * 2
                input_w_base = out_w_base * 2 + j * 2
                
                for ph in range(2):
                    for pw in range(2):
                        conv_h = input_h_base + ph
                        conv_w = input_w_base + pw
                        
                        conv_h_in_bounds = conv_h < H_in
                        conv_w_in_bounds = conv_w < W_in
                        if conv_h_in_bounds and conv_w_in_bounds:
                            # Load input value and apply 1x1 convolution (per-channel)
                            idx = conv_h * W_in + conv_w
                            x_val = tl.load(x_ptr + idx, mask=True, other=0.0)
                            c_idx = (m * BLOCK_SIZE_M + c_off) * C_in + n * C_in * H_in * W_in
                            weight_val = tl.load(weight_ptr + c_idx, mask=True, other=0.0)
                            conv_val = weight_val * x_val
                            pool_sum += conv_val
                            pool_count += 1
                
                # Average pooling result
                if pool_count > 0:
                    avg_val = pool_sum * (1.0 / pool_count)
                else:
                    avg_val = 0.0
                
                # Store result
                out_idx = n * (C_out * (H_in // 2) * (W_in // 2)) + \
                         (m * BLOCK_SIZE_M + c_off) * ((H_in // 2) * (W_in // 2)) + \
                         (out_h_base + i) * (W_in // 2) + (out_w_base + j)
                tl.store(out_ptr + out_idx, avg_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_pool(in_1, in_0):
    """
    Fused Conv2D (1x1) + AvgPool2D (2x2, stride=2) kernel
    """
    # Get input dimensions
    N, C_in, H_in, W_in = in_1.shape
    # Weight shape: [C_out, C_in, 1, 1] -> extract [C_out, C_in]
    C_out, _, _, _ = in_0.shape
    
    # Output dimensions after 1x1 conv + 2x2 avg pool with stride 2
    H_out = (H_in + 1) // 2  # Since pooling stride=2 and padding=0 for now
    W_out = (W_in + 1) // 2
    
    # Create output tensor
    out = torch.empty((N, C_out, H_out, W_out), device=in_1.device, dtype=in_1.dtype)
    
    # Launch kernel with optimized block sizes
    BLOCK_SIZE_M = 128  # Number of output channels per block
    BLOCK_SIZE_H = 4    # Height blocks
    BLOCK_SIZE_W = 4    # Width blocks
    
    # Calculate grid dimensions
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H  
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch the kernel
    simple_fused_conv_pool_kernel[(grid_m, grid_h, grid_w)](
        x_ptr=in_1,
        weight_ptr=in_0,
        out_ptr=out,
        N=N, C_in=C_in, C_out=C_out, H_in=H_in, W_in=W_in,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_pool