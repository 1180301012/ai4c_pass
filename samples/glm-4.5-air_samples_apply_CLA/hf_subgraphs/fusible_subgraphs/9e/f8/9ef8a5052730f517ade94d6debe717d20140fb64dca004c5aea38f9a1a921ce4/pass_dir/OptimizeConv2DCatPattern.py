import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,         # Input tensor [B, C_in, H, W]
    weight_ptr,    # Conv weights [C_out, C_in, 3, 3]
    out_ptr,       # Conv output [B, C_out, H, W]
    B, C_in, C_out, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ids for spatial dimensions and batch
    pid_m = tl.program_id(0)  # H dimension
    pid_n = tl.program_id(1)  # W dimension  
    pid_b = tl.program_id(2)  # Batch dimension
    
    # offsets for threads within block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_h = offs_m < H
    mask_w = offs_n < W
    
    # Process each output channel
    for c_out in range(C_out):
        # Initialize accumulator
        acc = 0.0
        
        # Compute convolution for this output channel
        for kh in range(3):
            for kw in range(3):
                # Calculate input position with padding (padding=1)
                h_in = offs_m[:, None] + kh - 1
                w_in = offs_n[None, :] + kw - 1
                
                # Apply mask to stay within bounds
                h_in_bound = h_in >= 0
                w_in_bound = w_in >= 0
                h_in_valid = h_in < H
                w_in_valid = w_in < W
                
                valid_mask = mask_h[:, None] & mask_w[None, :] & h_in_bound & w_in_bound & h_in_valid & w_in_valid
                
                if valid_mask.any():
                    # Load input and weight
                    h_in_flat = h_in[valid_mask]
                    w_in_flat = w_in[valid_mask]
                    
                    x_base = x_ptr + pid_b * C_in * H * W + h_in_flat * W + w_in_flat
                    x_val = tl.load(x_base, mask=valid_mask).to(tl.float32)
                    
                    weight_base = weight_ptr + c_out * C_in * 3 * 3 + kh * 3 + kw
                    weight_val = tl.load(weight_base + tl.arange(0, C_in), mask=tl.arange(0, C_in) < C_in).to(tl.float32)
                    
                    acc += tl.dot(x_val, weight_val)
        
        # Store result
        out_base = out_ptr + pid_b * C_out * H * W + c_out * H * W + offs_m[:, None] * W + offs_n[None, :]
        tl.store(out_base, acc, mask=mask_h[:, None] & mask_w[None, :])

@triton.jit
def simple_conv2d_kernel(
    x_ptr,         # Input tensor [B, C_in, H, W]
    weight_ptr,    # Conv weights [C_out, C_in, 3, 3]
    out_ptr,       # Output [B, C_out, H, W]
    B, C_in, C_out, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Simple kernel: each thread handles one spatial position and computes all channels
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    pid_b = tl.program_id(2)
    
    # Calculate spatial position for this thread
    h = pid_m * BLOCK_SIZE_M
    w = pid_n * BLOCK_SIZE_N
    
    # Process this spatial position for all output channels
    for c_out in range(C_out):
        # Compute convolution for this (h, w, c_out) position
        acc = 0.0
        
        for kh in range(3):
            for kw in range(3):
                # Calculate input position with padding (padding=1)
                h_in = h + kh - 1
                w_in = w + kw - 1
                
                # Check bounds
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    # Load input tensor for this position
                    x_base = x_ptr + pid_b * C_in * H * W + h_in * W + w_in
                    x_val = tl.load(x_base + tl.arange(0, C_in), mask=tl.arange(0, C_in) < C_in).to(tl.float32)
                    
                    # Load weight tensor for this kernel position and output channel  
                    weight_base = weight_ptr + c_out * C_in * 3 * 3 + kh * 3 + kw
                    weight_val = tl.load(weight_base + tl.arange(0, C_in), mask=tl.arange(0, C_in) < C_in).to(tl.float32)
                    
                    # Accumulate dot product
                    acc += tl.dot(x_val, weight_val)
        
        # Store result
        out_base = out_ptr + pid_b * C_out * H * W + c_out * H * W + h * W + w
        tl.store(out_base, acc)

@triton.jit
def conv_cat_simple_kernel(
    x_ptr,         # Input tensor [B, C_in, H, W]
    weight_ptr,    # Conv weights [C_out, C_in, 3, 3]
    y_ptr,         # Tensor to concatenate [B, C_cat, H, W]
    out_ptr,       # Output [B, C_out + C_cat, H, W]
    B, C_in, C_out, H, W, C_cat,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each thread handles one spatial position (h, w)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    h = pid_m * BLOCK_SIZE_M
    w = pid_n * BLOCK_SIZE_N
    
    # Process this spatial position for all channels
    for c_out in range(C_out + C_cat):
        if c_out < C_out:
            # Compute convolution for this (h, w, c_out) position
            acc = 0.0
            
            for kh in range(3):
                for kw in range(3):
                    # Calculate input position with padding (padding=1)
                    h_in = h + kh - 1
                    w_in = w + kw - 1
                    
                    # Check bounds and load data
                    if h_in >= 0:
                        if h_in < H:
                            if w_in >= 0:
                                if w_in < W:
                                    # Accumulate dot product by looping over channels
                                    for c in range(C_in):
                                        # Load input and weight for this channel
                                        x_base = x_ptr + pid_b * C_in * H * W + h_in * W + w_in + c
                                        x_val = tl.load(x_base).to(tl.float32)
                                        
                                        weight_base = weight_ptr + c_out * C_in * 3 * 3 + kh * 3 + kw + c
                                        weight_val = tl.load(weight_base).to(tl.float32)
                                        
                                        acc += x_val * weight_val
            
            # Store convolution result
            out_base = out_ptr + pid_b * (C_out + C_cat) * H * W + c_out * H * W + h * W + w
            tl.store(out_base, acc)
            
        else:
            # Copy from y tensor for concatenation part
            c_cat = c_out - C_out
            y_base = y_ptr + pid_b * C_cat * H * W + c_cat * H * W + h * W + w
            y_val = tl.load(y_base)
            
            # Store to output
            out_base = out_ptr + pid_b * (C_out + C_cat) * H * W + c_out * H * W + h * W + w
            tl.store(out_base, y_val)

@torch.fx.wrap  
def optimized_conv2d_cat(weight, x, y):
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    C_cat = y.shape[1]
    
    # Output tensor with concatenated channels
    result = torch.empty((B, C_out + C_cat, H, W), dtype=torch.float32, device=x.device)
    
    # Choose smaller block sizes to reduce overhead and improve occupancy
    BLOCK_SIZE_M = 8    # Block size for H dimension  
    BLOCK_SIZE_N = 8    # Block size for W dimension
    
    # Calculate grid dimensions
    grid_m = (H + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n, B)
    
    # Launch simple conv+cat kernel
    conv_cat_simple_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        y_ptr=y,
        out_ptr=result,
        B=B, C_in=C_in, C_out=C_out, H=H, W=W, C_cat=C_cat,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return result

def pattern(weight, x, y):
    tmp_1 = torch.conv2d(x, weight, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.cat((tmp_1, y), 1)
    return tmp_2

def replacement_args(weight, x, y):
    return (weight, x, y)

def replacement_func():
    return optimized_conv2d_cat