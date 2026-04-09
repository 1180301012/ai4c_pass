import torch
import triton
import triton.language as tl

# Pattern matching function for conv2d + add fusion
def pattern(conv_weight, context, value):
    """Match conv2d + add pattern commonly found in attention mechanisms"""
    # Note: groups must be hardcoded for pattern matching to work with symbolic tracing
    conv_result = torch.conv2d(value, conv_weight, None, (1, 1), (32, 0), (1, 1), 4)
    context_updated = context + conv_result
    return context_updated

# Argument extraction function
def replacement_args(conv_weight, context, value):
    return (conv_weight, context, value)

# Optimized kernel that fuses conv2d + add
@triton.jit
def fused_conv2d_add_kernel(
    # Input tensors
    value_ptr,         # [N, C_in, H_in, W_in] 
    weight_ptr,        # [groups, 1, K_h, K_w] = [C_out, 1, K_h, K_w] since groups=C_out
    context_ptr,       # [N, C_out, H_out, W_out]
    
    # Output tensor  
    output_ptr,        # [N, C_out, H_out, W_out]
    
    # Tensor shapes
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    
    # Conv2D parameters
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,    # Number of programs to execute
    BLOCK_SIZE_N: tl.constexpr,    # Number of output channels per program
    BLOCK_SIZE_K: tl.constexpr,    # Reduction dimension size
):
    # Calculate program indices
    m = tl.program_id(0)  # Batch * spatial position
    n = tl.program_id(1)  # Output channel
    
    # Extract batch and spatial coordinates from m
    batch = m // (H_out * W_out)
    spatial_idx = m % (H_out * W_out)
    h_out = spatial_idx // W_out
    w_out = spatial_idx % W_out
    
    # Calculate input coordinates with padding and dilation
    h_in = h_out * stride_h - pad_h
    w_in = w_out * stride_w - pad_w
    
    # Only process valid indices
    if h_in < 0 or h_in + (K_h - 1) * dilation_h + 1 > H_in or \
       w_in < 0 or w_in + (K_w - 1) * dilation_w + 1 > W_in:
        # For out-of-bound regions, just add context (conv result = 0)
        context_val = tl.load(context_ptr + batch * C_out * H_out * W_out + 
                             n * H_out * W_out + h_out * W_out + w_out,
                             mask=True)
        tl.store(output_ptr + batch * C_out * H_out * W_out + 
                n * H_out * W_out + h_out * W_out + w_out,
                context_val)
        return
    
    # Initialize convolution result
    conv_sum = 0.0
    
    # Compute convolution for this output position
    for k_h in range(0, K_h, BLOCK_SIZE_K):
        # Process a block of kernel height
        k_h_end = min(k_h + BLOCK_SIZE_K, K_h)
        
        for k_w in range(K_w):
            # Calculate input coordinate with dilation
            src_h = h_in + k_h * dilation_h
            src_w = w_in + k_w * dilation_w
            
            # Check bounds within this block
            if 0 <= src_h < H_in and 0 <= src_w < W_in:
                # Calculate group and input channel indices
                group_id = n // (C_out // groups) if groups > 1 else 0
                c_in = group_id  # Each group processes one input channel for groups=C_out
                
                # Load input value and weight
                input_val = tl.load(value_ptr + batch * C_in * H_in * W_in + 
                                   c_in * H_in * W_in + src_h * W_in + src_w,
                                   mask=True)
                weight_val = tl.load(weight_ptr + n * K_h * K_w + k_h * K_w + k_w,
                                   mask=True)
                
                # Multiply and accumulate
                conv_sum += input_val * weight_val
    
    # Load context value
    context_val = tl.load(context_ptr + batch * C_out * H_out * W_out + 
                         n * H_out * W_out + h_out * W_out + w_out,
                         mask=True)
    
    # Add context and conv result
    output_val = context_val + conv_sum
    
    # Store result
    tl.store(output_ptr + batch * C_out * H_out * W_out + 
            n * H_out * W_out + h_out * W_out + w_out,
            output_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv2d_add(conv_weight, context, value):
    # Get tensor shapes
    N, C_in, H_in, W_in = value.shape
    C_out, _, K_h, K_w = conv_weight.shape
    
    # Calculate output dimensions
    H_out = (H_in + 2 * 32 - 1 * (K_h - 1) - 1) // 1 + 1  # pad_h=32, dilation_h=1, stride_h=1
    W_out = (W_in + 2 * 0 - 1 * (K_w - 1) - 1) // 1 + 1    # pad_w=0, dilation_w=1, stride_w=1
    
    # Ensure output dimensions match context tensor
    assert H_out == context.shape[2] and W_out == context.shape[3], "Output dimensions don't match context"
    assert C_out == context.shape[1], "Output channels don't match context"
    
    # Create output tensor
    output = torch.empty_like(context)
    
    # Set block sizes for optimization
    BLOCK_SIZE_M = N * H_out * W_out  # Total number of spatial positions across all batches
    BLOCK_SIZE_N = 64  # Number of output channels per program
    BLOCK_SIZE_K = 16  # Block size for reduction dimension
    
    # Calculate grid size
    grid_M = (BLOCK_SIZE_M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_N = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv2d_add_kernel[(grid_M, grid_N, 1)](
        value_ptr=value,
        weight_ptr=conv_weight,
        context_ptr=context,
        output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, H_out=H_out, W_out=W_out,
        K_h=K_h, K_w=K_w,
        stride_h=1, stride_w=1,
        pad_h=32, pad_w=0,
        dilation_h=1, dilation_w=1,
        groups=C_out,  # Groups等于输出通道数
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv2d_add