import torch
import triton
import triton.language as tl

def pattern(in_1, in_0, in_2):
    """Match Conv2D followed by concatenation along channel dimension"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_0 = None
    tmp_2 = torch.cat((tmp_1, in_2), 1)
    tmp_1 = None
    return (tmp_2,)

def replacement_args(in_1, in_0, in_2):
    return (in_1, in_0, in_2)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr,  # [N, C_in, H, W]
    weight_ptr,  # [C_out, C_in, K_H, K_W]
    output_ptr,  # [N, C_out, H, W]
    N, C_in, H, W, C_out,
    K_H, K_W,
    stride_H, stride_W, pad_H, pad_W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized 2D convolution kernel"""
    # Each program handles one output pixel and channel
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    out_c = pid // (N * H * W)
    pid_remaining = pid % (N * H * W)
    batch = pid_remaining // (H * W)
    h = (pid_remaining // W) % H
    w = pid_remaining % W
    
    # Boundary checks
    if batch >= N or out_c >= C_out or h >= H or w >= W:
        return
    
    sum_val = 0.0
    
    # Convolution computation - iterate over input channels and kernel
    for c_in in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                # Calculate input coordinates with stride and padding
                in_h = h * stride_H + kh - pad_H
                in_w = w * stride_W + kw - pad_W
                
                # Skip if out of bounds
                if not (0 <= in_h < H and 0 <= in_w < W):
                    continue
                
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
def optimized_conv2d(input, weight, bias=None):
    """Optimized 2D convolution using Triton"""
    N, C_in, H, W = input.shape
    C_out, _, K_H, K_W = weight.shape
    
    # Output dimensions (with stride=1, padding=1, dilation=1)
    out_H = H
    out_W = W
    output_shape = (N, C_out, out_H, out_W)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Calculate grid size
    total_elements = N * C_out * out_H * out_W
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_conv2d_kernel[grid_size](
        input, weight, output,
        N, C_in, H, W, C_out,
        K_H, K_W,
        stride_H=1, stride_W=1, pad_H=1, pad_W=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    def combined_kernel(in_1, in_0, in_2):
        """Combined function that performs optimized conv2d + cat"""
        # Perform optimized convolution
        conv_result = optimized_conv2d(in_1, in_0)
        
        # Perform concatenation
        result = torch.cat((conv_result, in_2), 1)
        
        return result
    
    return combined_kernel