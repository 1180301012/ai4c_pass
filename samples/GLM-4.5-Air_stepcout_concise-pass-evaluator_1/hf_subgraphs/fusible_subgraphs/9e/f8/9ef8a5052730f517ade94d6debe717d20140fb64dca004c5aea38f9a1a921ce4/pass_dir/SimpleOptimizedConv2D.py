import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor):
    """Match only the Conv2D operation"""
    conv_result = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (1, 1), (1, 1), 1)
    return conv_result

def replacement_args(input_tensor, weight_tensor):
    return (input_tensor, weight_tensor)

@triton.jit
def simple_conv2d_kernel(
    input_ptr,  # [N, C_in, H, W]
    weight_ptr,  # [C_out, C_in, K_H, K_W]
    output_ptr,  # [N, C_out, H, W]
    N, C_in, H, W, C_out,
    K_H, K_W,
    stride_H, stride_W, pad_H, pad_W,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple 2D convolution kernel"""
    # Each program handles one output pixel and channel
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    out_c = pid // (N * H * W)
    pid_remaining = pid % (N * H * W)
    batch = pid_remaining // (H * W)
    h = (pid_remaining // W) % H
    w = pid_remaining % W
    
    # Boundary checks - split chained operators
    if batch >= N:
        return
    if h >= H or w >= W:
        return
    
    sum_val = 0.0
    
    # Convolution computation
    for c_in in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                # Calculate input coordinates with stride and padding
                in_h = h * stride_H + kh - pad_H
                in_w = w * stride_W + kw - pad_W
                
                # Only compute if coordinates are in bounds
                if (0 <= in_h):
                    if (in_h < H):
                        if (0 <= in_w):
                            if (in_w < W):
                                # Calculate input and weight indices
                                input_idx = batch * (C_in * H * W) + c_in * (H * W) + in_h * W + in_w
                                weight_idx = out_c * (C_in * K_H * K_W) + c_in * (K_H * K_W) + kh * K_W + kw
                                
                                # Load values
                                input_val = tl.load(input_ptr + input_idx)
                                weight_val = tl.load(weight_ptr + weight_idx)
                                
                                sum_val += input_val * weight_val
    
    # Store result
    output_idx = batch * (C_out * H * W) + out_c * (H * W) + h * W + w
    tl.store(output_ptr + output_idx, sum_val)

@torch.fx.wrap
def simple_optimized_conv2d(input, weight, bias=None):
    """Simple optimized 2D convolution using Triton"""
    N, C_in, H, W = input.shape
    C_out, _, K_H, K_W = weight.shape
    
    # Output dimensions (with stride=1, padding=1, dilation=1)
    output_shape = (N, C_out, H, W)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Calculate grid size
    total_elements = N * C_out * H * W
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel - pass grid as tuple
    simple_conv2d_kernel[(grid_size,)](
        input, weight, output,
        N, C_in, H, W, C_out,
        K_H, K_W,
        stride_H=1, stride_W=1, pad_H=1, pad_W=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    def optimized_conv2d_function(input_tensor, weight_tensor):
        """Optimized convolution function"""
        return simple_optimized_conv2d(input_tensor, weight_tensor)
    
    return optimized_conv2d_function