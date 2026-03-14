import torch
import triton
import triton.language as tl

def pattern(in_6, in_0):
    """Pattern matching for conv2d operation with 3x3 kernel, same padding"""
    tmp_5 = torch.conv2d(in_6, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    return tmp_5

def replacement_args(in_6, in_0):
    return (in_6, in_0)

@triton.jit
def conv2d_kernel_simple(
    x_ptr,           # Input tensor [B, C_in, H, W]
    weight_ptr,      # Weight tensor [C_out, C_in, K, K]
    out_ptr,         # Output tensor [B, C_out, H, W]
    B, C_in, H, W,   # Input dimensions
    C_out, K,        # Weight dimensions and kernel size
    stride_h, stride_w,  # Strides
    pad_h, pad_w,    # Padding
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    # Get program IDs using 3 dimensions
    b = tl.program_id(0)  # Batch index
    c_out = tl.program_id(1)  # Output channel
    spatial_idx = tl.program_id(2)  # Spatial position (flattened)
    
    # Calculate spatial position from flattened index
    h_out = spatial_idx // W
    w_out = spatial_idx % W
    
    # Check bounds (Triton requires nested conditions)
    if b >= B:
        return
    if c_out >= C_out:
        return
    if h_out >= H:
        return
    if w_out >= W:
        return
    
    # Initialize output accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kh in range(K):
            for kw in range(K):
                # Calculate input coordinates with padding
                h_in = h_out * stride_h + kh - pad_h
                w_in = w_out * stride_w + kw - pad_w
                
                # Check bounds (Triton requires nested conditions)
                if h_in >= 0:
                    if h_in < H:
                        if w_in >= 0:
                            if w_in < W:
                                # Load input value
                                x_offset = (b * C_in + c_in) * H * W + h_in * W + w_in
                                x_val = tl.load(x_ptr + x_offset)
                                
                                # Load weight value (flip kernel for convolution)
                                w_offset = (c_out * C_in + c_in) * K * K + (K-1-kh) * K + (K-1-kw)
                                w_val = tl.load(weight_ptr + w_offset)
                                
                                # Multiply accumulate
                                acc += x_val * w_val
    
    # Store output value
    out_offset = (b * C_out + c_out) * H * W + h_out * W + w_out
    tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def optimized_conv2d(x, weight):
    B, C_in, H, W = x.shape
    C_out, C_in_w, K, K = weight.shape
    
    # Validate input
    if C_in != C_in_w:
        raise ValueError(f"Channel dimension mismatch: {C_in} vs {C_in_w}")
    
    # For this specific pattern, we know the parameters
    stride = (1, 1)
    padding = (1, 1)
    
    # Output dimensions with same padding
    out_H = (H + 2 * padding[0] - K) // stride[0] + 1
    out_W = (W + 2 * padding[1] - K) // stride[1] + 1
    
    # Create output tensor
    out = torch.empty((B, C_out, out_H, out_W), dtype=x.dtype, device=x.device)
    
    # Block sizes for better GPU utilization
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid dimensions (only 3 dimensions: C_out, grid_h, grid_w)
    grid_h = (out_H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Instead of launching separately for each batch, we need to design differently
    # Launch kernel for the entire tensor at once, handling all batches in the kernel
    grid = (C_out, grid_h, grid_w)
    
    # Use a simpler kernel that handles all batches with 3D grid
    # Each program handles one batch, one output channel, and one spatial position
    spatial_grid = H * W
    grid = (B, C_out, spatial_grid)
    
    conv2d_kernel_simple[grid](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        B=B, C_in=C_in, H=H, W=W,
        C_out=C_out, K=K,
        stride_h=stride[0], stride_w=stride[1],
        pad_h=padding[0], pad_w=padding[1],
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out

def replacement_func():
    return optimized_conv2d