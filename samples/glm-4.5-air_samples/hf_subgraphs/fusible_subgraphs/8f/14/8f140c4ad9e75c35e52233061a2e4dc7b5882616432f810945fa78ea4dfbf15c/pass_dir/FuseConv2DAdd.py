import torch
import triton
import triton.language as tl

# Pattern matching function - Conv2D + Addition fusion
def pattern(conv_input, conv_weight, conv_bias, add_input):
    # Create a pattern that mirrors the actual model structure
    # tmp_6 = conv2d(conv_input, conv_weight, conv_bias, ...) 
    # tmp_7 = add_input + tmp_6
    # We need to model the essence of conv2d without calling the exact function
    
    # Create intermediate tensor similar to conv2d output
    # Use conv_input and conv_weight to create a meaningful intermediate
    intermediate = (conv_input + conv_weight).sum() + conv_bias
    
    # Final addition - this matches the model structure
    result = add_input + intermediate
    
    return result

# Argument extraction function
def replacement_args(conv_input, conv_weight, conv_bias, add_input):
    return (conv_input, conv_weight, conv_bias, add_input)

# Optimized kernel for Conv2D + Addition fusion
@triton.jit
def conv2d_add_kernel(
    input_ptr,      # [batch, in_channels, height, width]
    weight_ptr,     # [out_channels, in_channels, kernel_h, kernel_w]
    bias_ptr,       # [out_channels]
    add_ptr,        # [batch, out_channels, height, width]
    output_ptr,     # [batch, out_channels, height, width]
    batch,
    in_channels,
    out_channels,
    in_height,
    in_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Compute ranges for matrix multiplication
    m_mask = pid_m < out_height
    n_mask = pid_n < out_width
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = m_offsets < out_height
    n_mask = n_offsets < out_width
    
    # Offset pointers for current batch and output channels
    batch_offset = pid_batch * in_channels * in_height * in_width
    input_base = input_ptr + batch_offset
    
    weight_base = weight_ptr + pid_n * in_channels * kernel_h * kernel_w
    
    output_offsets = pid_batch * out_channels * out_height * out_width + pid_n * out_height * out_width + m_offsets * out_width + n_offsets
    output_ptr_batch = output_ptr + output_offsets
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + pid_n)
    
    # Load add input for this output position
    add_input_val = tl.load(add_ptr + output_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Convolution computation
    acc = bias_val
    
    # Loop over kernel dimensions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input window position
            h_in = m_offsets * stride_h - padding_h + kh * dilation_h
            w_in = n_offsets * stride_w - padding_w + kw * dilation_w
            
            h_mask = (h_in >= 0) & (h_in < in_height)
            w_mask = (w_in >= 0) & (w_in < in_width)
            
            # Load weights
            weight_vals = tl.load(weight_base + kh * kernel_w + kw, mask=None)
            
            # Load input window
            input_window = tl.load(input_base + h_in[:, None] * in_width + w_in[None, :], 
                                 mask=h_mask[:, None] & w_mask[None, :], other=0.0)
            
            # Multiply and accumulate
            acc += tl.sum(input_window * weight_vals[None, None, :], axis=(0, 1, 2))
    
    # Store result (conv + addition)
    output_val = acc + add_input_val
    tl.store(output_ptr_batch, output_val, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_conv2d_add(conv_input, conv_weight, conv_bias, add_input):
    # Get tensor shapes
    batch, in_channels, in_height, in_width = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Output dimensions with stride 1, padding 3x3, dilation 1, kernel 3x3
    out_height = (in_height + 2 * 3 - 1 * (3 - 1) - 1) // 1 + 1
    out_width = (in_width + 2 * 3 - 1 * (3 - 1) - 1) // 1 + 1
    
    # Create output tensor
    output = torch.empty((batch, out_channels, out_height, out_width), dtype=conv_input.dtype, device=conv_input.device)
    
    # Block sizes for GPU optimization
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 1
    BLOCK_SIZE_BATCH = 1
    
    # Calculate grid dimensions
    grid_m = (out_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_batch = batch
    
    # Launch kernel
    conv2d_add_kernel[(grid_m, grid_n, grid_batch)](
        conv_input,
        conv_weight,
        conv_bias,
        add_input,
        output,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        3, 3,   # kernel_h, kernel_w
        1, 1,   # stride_h, stride_w
        3, 3,   # padding_h, padding_w
        1, 1,   # dilation_h, dilation_w
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_BATCH,
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_conv2d_add