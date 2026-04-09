import torch
import triton
import triton.language as tl

# Pattern matching for depthwise conv2d + view
def pattern(input_tensor, weight):
    # Depthwise conv2d with exact parameters from model - use positional args for matching
    conv2d = torch.conv2d(input_tensor, weight, None, 1, 0, 1, 512)
    # Reshape operation
    tmp_5 = conv2d.view(1, 512, 64, 64)
    return tmp_5

# Argument extraction for the depthwise conv optimization
def replacement_args(input_tensor, weight):
    return (input_tensor, weight)

# Triton kernel for optimized depthwise convolution
@triton.jit
def depthwise_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N, C_in, H_in, W_in,
    C_out, kH, kW,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output channel (depthwise)
    pid = tl.program_id(0)
    if pid >= C_out:
        return
    
    # Calculate output dimensions
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1
    
    # Process output spatial positions efficiently
    for h_out in range(H_out):
        # Calculate input position with padding
        h_in = h_out * stride - padding
        
        # Process a row of output positions
        for w_out in range(0, W_out, BLOCK_SIZE):
            w_in_start = w_out * stride - padding
            w_off = w_in_start + tl.arange(0, BLOCK_SIZE)
            mask = w_off >= 0 & w_off < W_out
            
            # Initialize accumulator for each position
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Apply convolution: slide 7x7 window vertically
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input positions
                    h_valid = (h_in + kh) >= 0 & (h_in + kh) < H_in
                    w_valid = w_off + kw >= 0 & w_off + kw < W_in
                    valid_mask = h_valid & w_valid & mask
                    
                    # Calculate input indices using NCHW layout
                    input_offset = N * C_in * H_in * W_in + pid * H_in * W_in + (h_in + kh) * W_in + (w_off + kw)
                    
                    # Load input and weight
                    x = tl.load(input_ptr + input_offset, mask=valid_mask, other=0.0).to(tl.float32)
                    weight_idx = pid * kH * kW + kh * kW + kw
                    w = tl.load(weight_ptr + weight_idx).to(tl.float32)
                    
                    # Accumulate
                    acc += x * w
            
            # Store output using proper NCHW layout
            output_offset = N * C_out * H_out * W_out + pid * H_out * W_out + h_out * W_out + w_off
            tl.store(output_ptr + output_offset, acc, mask=mask)

@torch.fx.wrap
def optimized_depthwise_conv(input_tensor, weight):
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, kH, kW = weight.shape
    
    # For this specific case: input 70x70 -> output 64x64 with 7x7 kernel
    # With stride=1, padding=0: (70 + 0 - 7) // 1 + 1 = 63 + 1 = 64 ✓
    stride = 1
    padding = 0
    
    # Let me be more conservative and use smaller blocks with proper bounds checking
    out = torch.empty((N, C_out, 64, 64), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use smaller block size for better performance on this specific size
    BLOCK_SIZE = 64
    num_programs = C_out
    
    depthwise_conv_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        output_ptr=out,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, kH=kH, kW=kW,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function that returns the kernel wrapper
def replacement_func():
    return optimized_depthwise_conv