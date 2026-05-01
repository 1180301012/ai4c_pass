import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, in_0):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode = False, return_indices = False)
    return tmp_3

# Argument extraction function
def replacement_args(in_1, in_0):
    # Extract tensor shapes
    batch = in_1.shape[0]
    in_channels = in_1.shape[1]
    input_height = in_1.shape[2]
    input_width = in_1.shape[3]
    out_channels = in_0.shape[0]
    kernel_h = in_0.shape[2]
    kernel_w = in_0.shape[3]
    
    # Hardcoded parameters from the model
    stride_h = 2
    stride_w = 2
    padding_h = 3
    padding_w = 3
    pool_kernel_h = 3
    pool_kernel_w = 3
    pool_stride_h = 2
    pool_stride_w = 2
    pool_padding_h = 1
    pool_padding_w = 1
    
    # Calculate output shapes
    conv_output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
    conv_output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
    final_output_height = (conv_output_height + 2 * pool_padding_h - pool_kernel_h) // pool_stride_h + 1
    final_output_width = (conv_output_width + 2 * pool_padding_w - pool_kernel_w) // pool_stride_w + 1
    
    return (in_1, in_0, batch, in_channels, out_channels, input_height, input_width, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w, pool_padding_h, pool_padding_w,
            conv_output_height, conv_output_width, final_output_height, final_output_width)

# Triton kernel
def get_block_sizes(final_output_height, final_output_width):
    BLOCK_OUT_H = min(32, final_output_height)
    BLOCK_OUT_W = min(32, final_output_width)
    BLOCK_OUT_C = min(8, 64)  # Limit to 8 for better utilization
    return BLOCK_OUT_H, BLOCK_OUT_W, BLOCK_OUT_C

@triton.jit
def fused_conv_max_pool_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_height,
    input_width,
    conv_output_height,
    conv_output_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    pool_kernel_h,
    pool_kernel_w,
    pool_stride_h,
    pool_stride_w,
    pool_padding_h,
    pool_padding_w,
    out_channels,
    batch_size,
    BLOCK_OUT_H: tl.constexpr,
    BLOCK_OUT_W: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
):
    # Identify which element we're processing
    batch = tl.program_id(0)
    out_c = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    
    # Calculate the range of convolution output positions that contribute to this max pool
    conv_out_h_start = out_h * pool_stride_h - pool_padding_h
    conv_out_h_end = conv_out_h_start + pool_kernel_h
    conv_out_w_start = out_w * pool_stride_w - pool_padding_w
    conv_out_w_end = conv_out_w_start + pool_kernel_w
    
    # Initialize max value to a very small number
    max_val = -1e38

    # Loop through all positions in the convolution output that contribute to this max pool
    for conv_h in range(conv_out_h_start, conv_out_h_end):
        for conv_w in range(conv_out_w_start, conv_out_w_end):
            # Skip if outside convolution output boundaries
            if conv_h < 0 or conv_h >= conv_output_height or conv_w < 0 or conv_w >= conv_output_width:
                continue
            
            # Calculate input patch for convolution
            input_h_start = conv_h * stride_h - padding_h
            input_w_start = conv_w * stride_w - padding_w
            
            # Compute convolution value for this position
            conv_val = 0.0
            for in_c in range(in_channels):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        input_h = input_h_start + kh
                        input_w = input_w_start + kw
                        
                        # Check if input location is valid
                        if input_h < 0 or input_h >= input_height or input_w < 0 or input_w >= input_width:
                            continue
                            
                        # Load input and weight
                        input_val = tl.load(input_ptr + 
                                          (batch * in_channels + in_c) * input_height * input_width +
                                          input_h * input_width + input_w)
                        weight_val = tl.load(weight_ptr + 
                                           (out_c * in_channels + in_c) * kernel_h * kernel_w +
                                           kh * kernel_w + kw)
                        
                        conv_val += input_val * weight_val
            
            # Update max value
            if conv_val > max_val:
                max_val = conv_val
    
    # Store the result
    output_idx = (batch * out_channels + out_c) * (conv_output_height * conv_output_width) + \
                 out_h * conv_output_width + out_w
    tl.store(output_ptr + output_idx, max_val)

# Kernel wrapper
@torch.fx.wrap
def fused_conv_max_pool(in_1, in_0, batch, in_channels, out_channels, input_height, input_width, kernel_h, kernel_w,
                       stride_h, stride_w, padding_h, padding_w,
                       pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w, pool_padding_h, pool_padding_w,
                       conv_output_height, conv_output_width, final_output_height, final_output_width):
    # Create output tensor
    out = torch.empty((batch, out_channels, final_output_height, final_output_width), 
                     dtype=in_1.dtype, device=in_1.device)

    # Determine block sizes based on output dimensions
    BLOCK_OUT_H, BLOCK_OUT_W, BLOCK_OUT_C = get_block_sizes(final_output_height, final_output_width)

    # Calculate grid dimensions
    num_blocks_h = (final_output_height + BLOCK_OUT_H - 1) // BLOCK_OUT_H
    num_blocks_w = (final_output_width + BLOCK_OUT_W - 1) // BLOCK_OUT_W
    num_blocks_c = (out_channels + BLOCK_OUT_C - 1) // BLOCK_OUT_C

    # Launch the Triton kernel
    fused_conv_max_pool_kernel[
        (batch, num_blocks_c, num_blocks_h, num_blocks_w)
    ](
        in_1,
        in_0,
        out,
        input_height,
        input_width,
        conv_output_height,
        conv_output_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_h,
        pool_stride_w,
        pool_padding_h,
        pool_padding_w,
        out_channels,
        batch,
        BLOCK_OUT_H,
        BLOCK_OUT_W,
        BLOCK_OUT_C
    )

    return out

# Replacement function
def replacement_func():
    return fused_conv_max_pool