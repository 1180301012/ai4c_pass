import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_2, in_1, in_0):
    # This matches torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    # Using exact positional arguments as in the original model
    tmp_0 = in_0
    tmp_1 = in_1
    result = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    return result

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Auto-tuned Triton kernel for 1x1 convolution
@triton.jit
def conv2d_1x1_kernel(
    x_ptr,          # Input tensor [B, C_in, H, W]
    w_ptr,          # Weight tensor [C_out, C_in, 1, 1]
    b_ptr,          # Bias tensor [C_out]
    y_ptr,          # Output tensor [B, C_out, H, W]
    B: tl.constexpr, # Batch size  
    IC: tl.constexpr,# Input channels
    OC: tl.constexpr,# Output channels
    H: tl.constexpr, # Height
    W: tl.constexpr, # Width
):
    # Each program calculates one output element
    pid = tl.program_id(0)
    
    # Calculate coordinates [batch, channel, h, w]
    total_spatial = H * W
    w = pid % W
    h = (pid // W) % total_spatial
    oc = (pid // total_spatial) % OC
    b = pid // (total_spatial * OC)
    
    # Bounds check using separate conditions
    valid = True
    if b >= B: valid = False
    if oc >= OC: valid = False
    if h >= H: valid = False
    if w >= W: valid = False
    if not valid: return
    
    # Optimized convolution with memory access patterns tailored for 1x1
    acc = 0.0
    
    # Vectorized processing of input channels
    # For small IC, we can optimize by unrolling or vectorizing
    if IC <= 64:
        # Faster path for smaller input channels
        for k in range(0, IC, 4):
            if k + 3 < IC:
                # Process 4 channels at once
                # Get addresses
                in_idx1 = b * IC * H * W + k * H * W + h * W + w
                in_idx2 = b * IC * H * W + (k + 1) * H * W + h * W + w
                in_idx3 = b * IC * H * W + (k + 2) * H * W + h * W + w
                in_idx4 = b * IC * H * W + (k + 3) * H * W + h * W + w
                
                w_idx1 = oc * IC + k
                w_idx2 = oc * IC + (k + 1)
                w_idx3 = oc * IC + (k + 2)
                w_idx4 = oc * IC + (k + 3)
                
                # Load values
                x1 = tl.load(x_ptr + in_idx1)
                x2 = tl.load(x_ptr + in_idx2)
                x3 = tl.load(x_ptr + in_idx3)
                x4 = tl.load(x_ptr + in_idx4)
                
                w1 = tl.load(w_ptr + w_idx1)
                w2 = tl.load(w_ptr + w_idx2)
                w3 = tl.load(w_ptr + w_idx3)
                w4 = tl.load(w_ptr + w_idx4)
                
                # Vectorized multiply-accumulate
                acc += x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4
            else:
                # Process remaining channels
                for k1 in range(k, IC):
                    in_idx = b * IC * H * W + k1 * H * W + h * W + w
                    w_idx = oc * IC + k1
                    x_val = tl.load(x_ptr + in_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val
    else:
        # Large IC - standard loop
        for k in range(IC):
            in_idx = b * IC * H * W + k * H * W + h * W + w
            w_idx = oc * IC + k
            x_val = tl.load(x_ptr + in_idx)
            w_val = tl.load(w_ptr + w_idx)
            acc += x_val * w_val
    
    # Load bias
    b_val = tl.load(b_ptr + oc)
    
    # Store result
    out_idx = b * OC * H * W + oc * H * W + h * W + w
    tl.store(y_ptr + out_idx, acc + b_val)

# Simple kernel wrapper for the optimized convolution
@torch.fx.wrap
def optimized_conv2d_1x1(input_tensor, weight_tensor, bias_tensor):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Ensure output tensor has correct shape
    output_tensor = torch.empty((batch_size, out_channels, height, width), 
                               device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate grid size - one program per output element
    total_elements = batch_size * out_channels * height * width
    
    # Launch Triton kernel
    grid = (total_elements,)
    
    conv2d_1x1_kernel[grid](
        x_ptr=input_tensor,
        w_ptr=weight_tensor,
        b_ptr=bias_tensor,
        y_ptr=output_tensor,
        B=batch_size,
        IC=in_channels,
        OC=out_channels,
        H=height,
        W=width,
    )
    
    return output_tensor

# Replacement function
def replacement_func():
    return optimized_conv2d_1x1