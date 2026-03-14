import torch
import triton
import triton.language as tl

def pattern(in_2, weight, bias):
    """Pattern: conv2d operation"""
    tmp_2 = torch.conv2d(in_2, weight, bias, (1, 1), (1, 1), (1, 1), 1)
    return tmp_2

def replacement_args(in_2, weight, bias):
    """Extract arguments for the optimized conv2d operation"""
    return (in_2, weight, bias)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, H_in, W_in, C_out,
    stride0, stride1, padding0, padding1, dilation0, dilation1,
    BLOCK_SIZE_M: tl.constexpr
):
    """Optimized conv2d kernel using Triton"""
    pid_m = tl.program_id(0)
    
    # Determine which output position this program handles
    batch = pid_m // (C_out * H_in * W_in)
    channel = (pid_m % (C_out * H_in * W_in)) // (H_in * W_in)
    h = ((pid_m % (C_out * H_in * W_in)) % (H_in * W_in)) // W_in
    w = (pid_m % (C_out * H_in * W_in)) % W_in
    
    # Use separate conditions to avoid chained boolean operators
    if batch >= N:
        return
    if channel >= C_out:
        return  
    if h >= H_in:
        return
    if w >= W_in:
        return
    
    # Compute effective input coordinates with padding and dilation
    h_eff = h * stride0 - padding0
    w_eff = w * stride1 - padding1
    
    # Initialize accumulator
    acc = 0.0
    
    # Convolution computation
    BLOCK_SIZE_K = 64  # Use fixed constant since we removed it from parameters
    for ci in range(0, C_in, BLOCK_SIZE_K):
        ci_block = min(BLOCK_SIZE_K, C_in - ci)
        
        # Load weight slice for current channel and input channel block
        weight_coords = (channel, ci + tl.arange(0, ci_block), 0, 0)
        weight = tl.load(weight_ptr + weight_coords, mask=(tl.arange(0, ci_block) < C_in))
        
        # Load input neighborhood around current position
        input_coords = (
            batch,
            ci + tl.arange(0, ci_block),
            h_eff + tl.arange(0, 3) * dilation0,
            w_eff + tl.arange(0, 3) * dilation1
        )
        
        input_vals = tl.load(input_ptr + input_coords, mask=(
            (tl.arange(0, ci_block) < C_in) &
            (h_eff + tl.arange(0, 3) * dilation0 >= 0) & 
            (h_eff + tl.arange(0, 3) * dilation0 < H_in) &
            (w_eff + tl.arange(0, 3) * dilation1 >= 0) & 
            (w_eff + tl.arange(0, 3) * dilation1 < W_in)
        ))
        
        # Matrix multiplication and accumulation
        acc += (input_vals.to(tl.float32) * weight.to(tl.float32)).sum()
    
    # Add bias
    bias_val = tl.load(bias_ptr + channel)
    output_val = acc + bias_val
    
    # Store result
    output_coords = (batch, channel, h, w)
    tl.store(output_ptr + output_coords, output_val)

@torch.fx.wrap
def optimized_conv2d(input, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
    """Optimized conv2d function using Triton"""
    input_shape = input.shape
    weight_shape = weight.shape
    
    N, C_in, H_in, W_in = input_shape
    C_out = weight_shape[0]  # For groups=1, this is correct
    
    # For stride=1, padding=1, dilation=1, output size is same as input
    H_out, W_out = H_in, W_in
    
    output = torch.empty((N, C_out, H_out, W_out), dtype=input.dtype, device=input.device)
    
    # Set optimal block size for Triton
    BLOCK_SIZE_M = 128
    
    # Calculate grid dimensions
    total_elements = N * C_out * H_out * W_out
    grid = ((total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,)
    
    optimized_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out,
        stride0=stride[0], stride1=stride[1],
        padding0=padding[0], padding1=padding[1],
        dilation0=dilation[0], dilation1=dilation[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return output

def replacement_func():
    """Return the optimized conv2d function"""
    return optimized_conv2d