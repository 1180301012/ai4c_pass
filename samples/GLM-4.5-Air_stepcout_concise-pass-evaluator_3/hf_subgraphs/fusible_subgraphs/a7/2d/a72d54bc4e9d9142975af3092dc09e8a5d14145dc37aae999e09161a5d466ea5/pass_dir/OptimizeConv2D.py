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
def conv2d_kernel(
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
    # Get program IDs (using only 3 dimensions)
    c_out = tl.program_id(0)  # Output channel
    h_start = tl.program_id(1) * BLOCK_SIZE_H
    w_start = tl.program_id(2) * BLOCK_SIZE_W
    
    # Initialize output accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kh in range(K):
            for kw in range(K):
                # Calculate input coordinates with padding
                h_in = h_start * stride_h + kh - pad_h
                w_in = w_start * stride_w + kw - pad_w
                
                # Check bounds (Triton requires nested conditions)
                if h_in >= 0:
                    if h_in < H:
                        if w_in >= 0:
                            if w_in < W:
                                # Load input value
                                x_offset = (b * C_in + c_in) * H * W + h_in * W + w_in
                                x_val = tl.load(x_ptr + x_offset)
                                
                                # Load weight value
                                w_offset = (c_out * C_in + c_in) * K * K + kh * K + kw
                                w_val = tl.load(weight_ptr + w_offset)
                                
                                # Multiply accumulate
                                acc += x_val * w_val
    
    # Store output value
    out_offset = (b * C_out + c_out) * H * W + h_start * W + w_start
    tl.store(out_ptr + out_offset, acc)

@triton.jit
def conv2d_kernel(
    x_ptr,           # Input tensor [B, C_in, H, W]
    weight_ptr,      # Weight tensor [C_out, C_in, K, K]
    out_ptr,         # Output tensor [B, C_out, H, W]
    b,               # Batch index
    C_in,            # Input channels
    H, W,            # Input dimensions
    C_out, K,        # Weight dimensions and kernel size
    stride_h, stride_w,  # Strides
    pad_h, pad_w,    # Padding
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    # Get program IDs (using only 3 dimensions)
    c_out = tl.program_id(0)  # Output channel
    h_start = tl.program_id(1) * BLOCK_SIZE_H
    w_start = tl.program_id(2) * BLOCK_SIZE_W
    
    # Initialize output accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kh in range(K):
            for kw in range(K):
                # Calculate input coordinates with padding
                h_in = h_start * stride_h + kh - pad_h
                w_in = w_start * stride_w + kw - pad_w
                
                # Check bounds (Triton requires nested conditions)
                if h_in >= 0:
                    if h_in < H:
                        if w_in >= 0:
                            if w_in < W:
                                # Load input value
                                x_offset = (b * C_in + c_in) * H * W + h_in * W + w_in
                                x_val = tl.load(x_ptr + x_offset)
                                
                                # Load weight value
                                w_offset = (c_out * C_in + c_in) * K * K + kh * K + kw
                                w_val = tl.load(weight_ptr + w_offset)
                                
                                # Multiply accumulate
                                acc += x_val * w_val
    
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
    
    # Launch kernel for each batch separately
    for b_idx in range(B):
        # Initialize output for this batch
        out_b = out[b_idx]  # Get slice for current batch
        
        grid = (C_out, grid_h, grid_w)
        conv2d_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            out_ptr=out_b,
            b_idx,
            C_in, H, W,
            C_out, K,
            stride[0], stride[1],
            padding[0], padding[1],
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W
        )
    
    return out

def replacement_func():
    return optimized_conv2d