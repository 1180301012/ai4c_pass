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
def conv2d_kernel(
    input_ptr,  # [N, C_in, H, W]
    weight_ptr,  # [C_out, C_in, K_H, K_W]
    output_ptr,  # [N, C_out, H, W]
    N, C_in, H, W, C_out,
    K_H, K_W, pad_H, pad_W,
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
                # Calculate input coordinates with padding
                in_h = h + kh - pad_H
                in_w = w + kw - pad_W
                
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
def fused_conv_concat(in_1, in_0, in_2):
    """Fused convolution and concatenation using optimized kernels"""
    # Get input shapes
    N, C_in, H, W = in_1.shape
    C_out, _, K_H, K_W = in_0.shape
    C_concat = in_2.shape[1]
    
    # Step 1: Perform optimized convolution
    conv_output = torch.empty((N, C_out, H, W), dtype=in_1.dtype, device=in_1.device)
    
    # Calculate grid size for convolution
    conv_elements = N * C_out * H * W
    BLOCK_SIZE = 1024
    conv_grid_size = (conv_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch convolution kernel
    conv2d_kernel[conv_grid_size](
        in_1, in_0, conv_output,
        N, C_in, H, W, C_out,
        K_H, K_W, 1, 1,
        BLOCK_SIZE
    )
    
    # Step 2: Efficient concatenation using torch.cat (now allowed after fx.wrap)
    # After @torch.fx.wrap, torch operations are allowed in the wrapper function
    result = torch.cat((conv_output, in_2), dim=1)
    
    return result

def replacement_func():
    return fused_conv_concat