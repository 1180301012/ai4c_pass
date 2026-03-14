import torch
import triton
import triton.language as tl
from math import ceil

@triton.jit
def conv2d_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    H_in, W_in, C_in, C_out,
    H_out, W_out,
    kernel_h, kernel_w,
    pad_h, pad_w,
    stride_h, stride_w,
    dilation_h, dilation_w,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for fused Conv2D + GELU operation
    Optimized for both depth-wise and regular convolutions
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output block ranges
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Compute output coordinates
    h_out = m_offset // W_out
    w_out = m_offset % W_out
    c_out = n_offset
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Depth-wise convolution optimization
    if groups == C_out and C_in == C_out:
        # Depth-wise convolution case
        for kh in range(0, kernel_h, BLOCK_SIZE_K):
            for kw in range(0, kernel_w, BLOCK_SIZE_K):
                # Load input block
                in_h = h_out * stride_h - pad_h + kh
                in_w = w_out * stride_w - pad_w + kw
                
                mask_h = (in_h >= 0) & (in_h < H_in)
                mask_w = (in_w >= 0) & (in_w < W_in)
                
                if mask_h and mask_w:
                    # Load input window for this output position
                    input_ptr_base = input_ptr + (in_h * W_in + in_w) * C_in
                    weight_ptr_base = weight_ptr + (kh * kernel_w + kw) * C_out
                    
                    for k in range(0, BLOCK_SIZE_K):
                        k_in = min(k, C_in - 1)
                        k_out = min(k, C_out - 1)
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr_base + k_in, mask=k_in < C_in, other=0.0)
                        weight_val = tl.load(weight_ptr_base + k_out, mask=k_out < C_out, other=0.0)
                        
                        # Multiply and accumulate
                        for m in range(BLOCK_SIZE_M):
                            for n in range(BLOCK_SIZE_N):
                                if n < C_out:
                                    acc[m, n] += input_val * weight_val
    else:
        # Regular/grouped convolution case
        for kh in range(0, kernel_h):
            for kw in range(0, kernel_w):
                # Load input window
                in_h = h_out * stride_h - pad_h + kh
                in_w = w_out * stride_w - pad_w + kw
                
                mask_h = (in_h >= 0) & (in_h < H_in)
                mask_w = (in_w >= 0) & (in_w < W_in)
                
                if mask_h and mask_w:
                    input_ptr_base = input_ptr + (in_h * W_in + in_w) * C_in
                    
                    for c_in_group in range(groups):
                        c_in_start = c_in_group * (C_in // groups)
                        c_in_end = (c_in_group + 1) * (C_in // groups)
                        c_out_start = c_in_group * (C_out // groups)
                        c_out_end = (c_in_group + 1) * (C_out // groups)
                        
                        for k in range(c_in_start, c_in_end):
                            for n in range(c_out_start, c_out_end):
                                if n < c_out_end and k < C_in:
                                    input_val = tl.load(input_ptr_base + k, mask=k < C_in, other=0.0)
                                    weight_val = tl.load(weight_ptr + (c_out * kernel_h * kernel_w + kh * kernel_w + kw) * C_in + k, 
                                                       mask=k < C_in, other=0.0)
                                    for m in range(BLOCK_SIZE_M):
                                        acc[m, n - c_out_start] += input_val * weight_val
    
    # Add bias
    bias_val = 0.0
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c_out, mask=c_out < C_out, other=0.0)
    acc += bias_val
    
    # Apply GELU activation
    # GELU(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            x = acc[m, n]
            x_cubed = x * x * x
            inner = x * 0.044715 + x_cubed * 0.035443  # sqrt(2/pi) * 0.044715 ≈ 0.035443
            inner = inner * 1.782405  # sqrt(2/pi) ≈ 1.782405
            gelu = x * 0.5 * (1.0 + tl.tanh(inner))
            acc[m, n] = gelu
    
    # Store output
    output_ptr_base = output_ptr + (m_offset + w_out) * C_out + n_offset
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            if (m_offset + m) < H_out * W_out:
                tl.store(output_ptr_base + (n + m * C_out), acc[m, n])

@torch.fx.wrap
def fused_conv2d_gelu(input, weight, bias, stride, padding, dilation, groups):
    """
    Wrapper function for fused Conv2D + GELU operation
    """
    # Get input dimensions
    batch_size = input.size(0)
    C_in, H_in, W_in = input.shape[1], input.shape[2], input.shape[3]
    C_out = weight.size(0)
    
    # Calculate output dimensions
    H_out = ceil((H_in + 2 * padding[0] - dilation[0] * (weight.shape[2] - 1) - 1) / stride[0] + 1)
    W_out = ceil((W_in + 2 * padding[1] - dilation[1] * (weight.shape[3] - 1) - 1) / stride[1] + 1)
    
    # Create output tensor
    output = torch.empty((batch_size, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Set kernel parameters
    BLOCK_SIZE_M = 16  # Output spatial block size
    BLOCK_SIZE_N = 32  # Output channel block size
    BLOCK_SIZE_K = 8   # Input channel block size
    
    # Calculate grid dimensions
    grid_x = (H_out * W_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_y = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    for batch in range(batch_size):
        conv2d_gelu_kernel[(
            grid_x, 
            grid_y,
            0
        )](
            input_ptr=input[batch].data_ptr(),
            weight_ptr=weight.data_ptr(),
            bias_ptr=bias.data_ptr() if bias is not None else None,
            output_ptr=output[batch].data_ptr(),
            H_in=H_in, W_in=W_in, C_in=C_in, C_out=C_out,
            H_out=H_out, W_out=W_out,
            kernel_h=weight.shape[2], kernel_w=weight.shape[3],
            pad_h=padding[0], pad_w=padding[1],
            stride_h=stride[0], stride_w=stride[1],
            dilation_h=dilation[0], dilation_w=dilation[1],
            groups=groups,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    return output

def pattern(x, w, b):
    """
    Pattern to match Conv2D → GELU sequence exactly as it appears in the models
    Note: The actual model calls use positional arguments (1,1), (1,1), (1,1), groups
    where groups is captured from the environment
    """
    # This matches the exact pattern in the models:
    # torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), groups)
    # We need to capture groups as a free variable in the pattern
    groups = 1024  # Default value, will be overridden by actual groups in the model
    
    conv_out = torch.conv2d(x, w, b, (1, 1), (1, 1), (1, 1), groups)
    gelu_out = torch.nn.functional.gelu(conv_out)
    return gelu_out

def replacement_args(x, w, b):
    """
    Extract arguments for the fused operation
    """
    return (x, w, b)

def replacement_func():
    """
    Return the fused Conv2D + GELU function
    """
    return fused_conv2d_gelu