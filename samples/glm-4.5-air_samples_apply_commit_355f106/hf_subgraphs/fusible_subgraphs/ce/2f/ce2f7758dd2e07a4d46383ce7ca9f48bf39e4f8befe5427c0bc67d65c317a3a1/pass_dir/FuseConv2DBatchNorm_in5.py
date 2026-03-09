import torch
import triton
import triton.language as tl

def pattern():
    # This function takes no arguments - the framework binds actual tensors to these operations
    import torch
    
    # Create placeholder variables to match the computation pattern
    in_5 = torch.randn(1, 128, 28, 28)  # Example shape from the graph
    tmp_4 = torch.randn(128, 128, 3, 3)  # Example conv weight shape
    running_mean = torch.randn(128)
    running_var = torch.randn(128)
    weight = torch.randn(128)
    bias = torch.randn(128)
    
    # Conv2D operation with the exact parameters from the graphs (using in_5)
    tmp_5 = torch.conv2d(in_5, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
    # BatchNorm operation with exact parameters from the graphs  
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return tmp_5, tmp_6

def replacement_args():
    # No arguments needed - the framework will extract the actual tensors
    return ()

# Optimized fused Conv2D + BatchNorm kernel
@triton.jit
def fused_conv_bn_kernel(
    input_ptr, 
    weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_bn_ptr,
    bias_bn_ptr,
    output_ptr,
    N, C_out, H, W,
    C_in, K_H, K_W,
    stride_h, stride_w, pad_h, pad_w,
    eps: tl.constexpr,
):
    # Program ids: pid for spatial positions and channel
    pid_spatial = tl.program_id(0)  # Combined n*h_out*w_out
    pid_channel = tl.program_id(1)  # Output channel c_out
    
    # Extract coordinates
    c_out = pid_channel
    
    # Check bounds for this channel
    if c_out >= C_out:
        return
        
    # Extract spatial coordinates from linear position
    h_out = pid_spatial // (W * N)
    w_out = (pid_spatial // N) % W
    n = pid_spatial % N
    
    # Check bounds
    if n >= N or h_out >= H or w_out >= W:
        return
    
    # Compute input position considering padding
    h_in = h_out * stride_h - pad_h
    w_in = w_out * stride_w - pad_w
    
    # Initialize output value
    output_val = 0.0
    
    # Perform convolution for this output channel
    for c_in in range(C_in):
        # For each kernel position
        for kh in range(K_H):
            for kw in range(K_W):
                # Calculate input coordinates with padding
                in_h = h_in + kh
                in_w = w_in + kw
                
                # Check bounds for input access
                if 0 <= in_h < H and 0 <= in_w < W:
                    # Load input element: [N, C_in, H, W] layout
                    input_offset = (n * C_in + c_in) * H * W + in_h * W + in_w
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Load weight element: [C_out, C_in, K_H, K_W] layout  
                    weight_offset = (c_out * C_in + c_in) * K_H * K_W + kh * K_W + kw
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    output_val += input_val * weight_val
    
    # Load batch normalization parameters for this channel
    running_mean_val = tl.load(running_mean_ptr + c_out) if running_mean_ptr else 0.0
    running_var_val = tl.load(running_var_ptr + c_out) if running_var_ptr else 1.0
    weight_bn_val = tl.load(weight_bn_ptr + c_out) if weight_bn_ptr else 1.0
    bias_bn_val = tl.load(bias_bn_ptr + c_out) if bias_bn_ptr else 0.0
    
    # Apply batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
    output_val = (output_val - running_mean_val) / tl.sqrt(running_var_val + eps) * weight_bn_val + bias_bn_val
    
    # Store result: [N, C_out, H, W] layout
    output_offset = (n * C_out + c_out) * H * W + h_out * W + w_out
    tl.store(output_ptr + output_offset, output_val)

@torch.fx.wrap  
def fused_conv_bn_cuda(input, weight, running_mean, running_var, weight_bn, bias_bn):
    # Get input dimensions
    N, C_in, H, W = input.shape
    C_out, _, K_H, K_W = weight.shape
    eps = 1e-05
    
    # Calculate total spatial elements (n * h * w)
    total_spatial_elements = N * H * W
    
    # Set up 2D grid: (spatial_elements, output_channels)
    # Each program handles one output channel for each spatial position
    grid = (total_spatial_elements, C_out)
    
    # Initialize output tensor with correct dimensions
    output = torch.empty((N, C_out, H, W), dtype=input.dtype, device=input.device)
    
    # Launch kernel with 2D grid
    fused_conv_bn_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_bn_ptr=weight_bn,
        bias_bn_ptr=bias_bn,
        output_ptr=output,
        N=N, C_out=C_out, H=H, W=W,
        C_in=C_in, K_H=K_H, K_W=K_W,
        stride_h=1, stride_w=1, pad_h=1, pad_w=1,
        eps=eps,
    )
    
    return output

def replacement_func():
    return fused_conv_bn_cuda