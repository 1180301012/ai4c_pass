import torch
import triton
import triton.language as tl

# Pattern matching function - exactly matches the computation in model.py
def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel fused convolution and average pooling
@triton.jit
def fused_conv_avgpool_kernel(
    input_ptr,          # Input tensor [N, C_in, H, W]
    weight_ptr,         # Weight tensor [C_out, C_in, K_H, K_W]
    output_ptr,         # Output tensor [N, C_out, H_out, W_out]
    N, C_in, C_out, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Compute output dimensions (after conv: H×W, after pooling: H/2 × W/2)
    H_out = H // 2
    W_out = W // 2
    
    # Compute ranges for this program
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_out_range = pid_c_out * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Mask for bounds checking
    n_mask = n_range < N
    c_out_mask = c_out_range < C_out
    
    # Iterate over output patches for pooling
    for h_out_idx in range(0, max(H_out // BLOCK_SIZE_H, 1)):
        for w_out_idx in range(0, max(W_out // BLOCK_SIZE_W, 1)):
            # Output coordinates
            h_out_start = h_out_idx * BLOCK_SIZE_H
            w_out_start = w_out_idx * BLOCK_SIZE_W
            h_out_range = h_out_start + tl.arange(0, BLOCK_SIZE_H)
            w_out_range = w_out_start + tl.arange(0, BLOCK_SIZE_W)
            
            h_out_mask = h_out_range < H_out
            w_out_mask = w_out_range < W_out
            
            # Accumulator for the fused operation
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
            
            # Accumulate over the 2x2 pooling window
            for pool_h in range(2):
                for pool_w in range(2):
                    # Calculate input coordinates (accounting for convolution stride=1 and pooling stride=2)
                    h_in = h_out_range * 2 + pool_h
                    w_in = w_out_range * 2 + pool_w
                    h_in_mask = h_in < H
                    w_in_mask = w_in < W
                    
                    # Broadcast operations
                    for n in range(BLOCK_SIZE_N):
                        if n_range[n] < N:
                            for c_out in range(BLOCK_SIZE_C):
                                if c_out_range[c_out] < C_out:
                                    # Load weight (1x1 convolution)
                                    weight = tl.load(weight_ptr + c_out_range[c_out] * C_in, 
                                                   mask=c_out_mask[c_out], other=0.0)
                                    
                                    # Load input patch
                                    for h_idx in range(BLOCK_SIZE_H):
                                        if h_out_range[h_idx] < H_out:
                                            for w_idx in range(BLOCK_SIZE_W):
                                                if w_out_range[w_idx] < W_out:
                                                    h = h_in[h_idx]
                                                    w = w_in[w_idx]
                                                    if h < H and w < W:
                                                        # Load input element
                                                        input_val = tl.load(input_ptr + 
                                                                           n_range[n] * C_in * H * W + 
                                                                           c_out_range[c_out] * H * W + 
                                                                           h * W + w,
                                                                           mask=(n_mask[n] & c_out_mask[c_out] & 
                                                                                 h_in_mask[h_idx] & w_in_mask[w_idx]), 
                                                                           other=0.0)                                                        
                                                        # Convolution: multiply by weight
                                                        conv_val = input_val * weight
                                                        # Accumulate for average pooling
                                                        acc[n, c_out, h_idx, w_idx] += conv_val
            
            # Average pooling: divide by 4 (2x2 window)
            acc = acc / 4.0
            
            # Store output
            for n in range(BLOCK_SIZE_N):
                if n_range[n] < N:
                    for c_out in range(BLOCK_SIZE_C):
                        if c_out_range[c_out] < C_out:
                            for h_idx in range(BLOCK_SIZE_H):
                                if h_out_range[h_idx] < H_out:
                                    for w_idx in range(BLOCK_SIZE_W):
                                        if w_out_range[w_idx] < W_out:
                                            tl.store(output_ptr + 
                                                   n_range[n] * C_out * H_out * W_out + 
                                                   c_out_range[c_out] * H_out * W_out + 
                                                   h_out_range[h_idx] * W_out + 
                                                   w_out_range[w_idx],
                                                   acc[n, c_out, h_idx, w_idx],
                                                   mask=(n_mask[n] & c_out_mask[c_out] & 
                                                         h_out_mask[h_idx] & w_out_mask[w_idx]))

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_avgpool(input_tensor, weight_tensor):
    # Get tensor dimensions
    N, C_in, H, W = input_tensor.shape
    C_out, _, _, _ = weight_tensor.shape
    
    # Output dimensions after convolution (stride=1) and then pooling (stride=2)
    H_out = H // 2
    W_out = W // 2
    
    # Output tensor
    output_tensor = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose tile sizes based on tensor dimensions
    BLOCK_SIZE_N = min(4, N)
    BLOCK_SIZE_C = min(256, C_out)
    BLOCK_SIZE_H = min(4, H_out)
    BLOCK_SIZE_W = min(4, W_out)
    
    # Calculate grid size
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = max(1, (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H)
    grid_w = max(1, (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    
    # Launch kernel
    fused_conv_avgpool_kernel[(grid_n, grid_c, grid_h, grid_w)](
        input_tensor,
        weight_tensor,
        output_tensor,
        N, C_in, C_out, H, W,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W
    )
    
    return output_tensor

# Replacement function (returns the function reference)
def replacement_func():
    return fused_conv_avgpool