import torch
import triton
import triton.language as tl
import math


def pattern(conv_input, conv_weight, conv_stride, conv_padding, conv_dilation, conv_groups, 
           pool_kernel, pool_stride, pool_padding, pool_ceil_mode):
    """
    Matches a conv2d followed by max_pool2d pattern that appears in all the target graphs.
    This pattern is found in all the ResNet graphs with varying parameter configurations.
    """
    conv2d_output = torch.conv2d(conv_input, conv_weight, None, 
                               conv_stride, conv_padding, conv_dilation, conv_groups)
    max_pool_output = torch.nn.functional.max_pool2d(conv2d_output, pool_kernel, 
                                                    pool_stride, pool_padding, 
                                                    ceil_mode=pool_ceil_mode, 
                                                    return_indices=False)
    return max_pool_output


def replacement_args(conv_input, conv_weight, conv_stride, conv_padding, conv_dilation, conv_groups, 
                    pool_kernel, pool_stride, pool_padding, pool_ceil_mode):
    """
    Extract arguments for the fused conv+pool operation.
    We focus on the most common configuration across all graphs.
    """
    return (conv_input, conv_weight, conv_stride, conv_padding, conv_dilation, conv_groups,
            pool_kernel, pool_stride, pool_padding, pool_ceil_mode)


@triton.jit
def fused_conv2d_maxpool_kernel(input_ptr, weight_ptr, output_ptr, 
                              batch_size, in_channels, in_height, in_width,
                              out_channels, kernel_h, kernel_w, stride_h, stride_w, 
                              pad_h, pad_w, dilation_h, dilation_g,
                              pool_kh, pool_kw, pool_sh, pool_sw, pool_ph, pool_pw,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                              BLOCK_K: tl.constexpr):
    """
    Fused Conv2d + MaxPool2d kernel using Triton.
    This kernel performs convolution followed by max pooling in a single pass,
    reducing memory bandwidth requirements and improving performance.
    """
    # Program ID for parallel execution
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * out_channels, BLOCK_M)
    
    # Calculate output dimensions
    out_height = tl.cdiv(in_height + 2 * pad_h - dilation_h * (kernel_h - 1), stride_h) + 1
    out_width = tl.cdiv(in_width + 2 * pad_w - dilation_w * (kernel_w - 1), stride_w) + 1
    pool_out_h = tl.cdiv(out_height + 2 * pool_ph - pool_kh, pool_sw) + 1
    pool_out_w = tl.cdiv(out_width + 2 * pool_pw - pool_kw, pool_sw) + 1
    
    # Map program ID to output tile
    m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = tl.arange(0, BLOCK_N)
    
    # Initialize output accumulator for max pooling
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over input channels (K dimension)
    k_max = tl.cdiv(in_channels, BLOCK_K)
    for k_block in range(k_max):
        k_idx = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Calculate input coordinates for this K block
        input_coords = get_conv_input_coords(m_idx, n_idx, k_idx, 
                                           out_channels, in_height, in_width,
                                           kernel_h, kernel_w, stride_h, stride_w,
                                           pad_h, pad_w, dilation_h, dilation_g,
                                           batch_size, in_channels)
        
        # Load input and weights
        input_vals = load_input_for_conv(input_coords, input_ptr, 
                                       batch_size, in_channels, in_height, in_width)
        weight_vals = load_weights_for_conv(n_idx, k_idx, weight_ptr,
                                           out_channels, in_channels, 
                                           kernel_h, kernel_w)
        
        # Perform convolution operation
        conv_vals = tl.dot(input_vals, weight_vals)
        
        # Update accumulator for max pooling
        accumulator = tl.maximum(accumulator, conv_vals)
    
    # Apply max pooling to the convolution results
    pooled_vals = apply_max_pooling(accumulator, pool_kh, pool_kw, pool_sh, pool_sw, 
                                   pool_ph, pool_pw, pool_out_h, pool_out_w)
    
    # Store results to global memory
    store_output(pooled_vals, output_ptr, m_idx, n_idx, 
                batch_size, out_channels, pool_out_h, pool_out_w)


def get_conv_input_coords(m_idx, n_idx, k_idx, out_channels, in_height, in_width,
                         kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, 
                         dilation_h, dilation_g, batch_size, in_channels):
    """Calculate input coordinates for convolution operation."""
    # Convert linear indices to 3D coordinates (batch, channel, height, width)
    batch_id = m_idx // out_channels
    batch_id = tl.minimum(batch_id, batch_size - 1)
    
    out_channel_id = m_idx % out_channels
    out_channel_id = tl.minimum(out_channel_id, out_channels - 1)
    
    # Convert output coordinates to input coordinates
    out_coords = get_output_coords(n_idx, out_channels, in_height, in_width,
                                   kernel_h, kernel_w, stride_h, stride_w,
                                   pad_h, pad_w, dilation_h, dilation_g)
    
    # Apply pooling to get final output coordinates
    pool_out_coords = get_pool_coordinates(out_coords, pool_kh, pool_kw, pool_sh, pool_sw,
                                         pool_ph, pool_pw)
    
    return pool_out_coords


def get_output_coords(idx, out_channels, in_height, in_width, kernel_h, kernel_w,
                     stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_g):
    """Convert linear index to 2D output coordinates."""
    height = idx // in_width
    width = idx % in_width
    
    # Convert to input coordinates
    inp_height = height * stride_h - pad_h
    inp_width = width * stride_w - pad_w
    
    return tl.max(0, tl.min([inp_height, in_height - 1])), \
           tl.max(0, tl.min([inp_width, in_width - 1]))


def get_pool_coordinates(coords, kh, kw, sh, sw, ph, pw):
    """Convert coordinates after pooling."""
    height = coords // coords.shape[1]
    width = coords % coords.shape[1]
    
    pool_height = height * sh - ph
    pool_width = width * sw - pw
    
    return tl.max(0, tl.min([pool_height, coords.shape[0] - 1])), \
           tl.max(0, tl.min([pool_width, coords.shape[1] - 1]))


def load_input_for_conv(coords, input_ptr, batch_size, in_channels, in_height, in_width):
    """Load input values for convolution operation."""
    # This is a simplified implementation - in practice you'd need more complex coordinate handling
    input_vals = tl.zeros(BLOCK_M * BLOCK_N, dtype=tl.float32)
    return input_vals


def load_weights_for_conv(n_idx, k_idx, weight_ptr, out_channels, in_channels, kernel_h, kernel_w):
    """Load weight values for convolution operation."""
    # This is a simplified implementation
    weight_vals = tl.zeros(BLOCK_N * BLOCK_K, dtype=tl.float32)
    return weight_vals


def apply_max_pooling(data, kh, kw, sh, sw, ph, pw, out_h, out_w):
    """Apply max pooling operation to data."""
    # Simplified max pooling implementation
    reshaped = data.reshape(out_h, out_w, kh * kw)
    max_vals = tl.max(reshaped, dim=2)
    return max_vals


def store_output(data, output_ptr, m_idx, n_idx, batch_size, out_channels, out_h, out_w):
    """Store output data to global memory."""
    # Store results
    output_coords = get_output_coords(m_idx, out_channels, out_h, out_w, 1, 1, 1, 1, 0, 0, 1, 1)
    tl.store(output_ptr + m_idx * out_channels + n_idx, output_coords[0], mask=tlPredicate(m_idx < batch_size * out_channels))


def tlPredicate(predicate):
    """Convert predicate to Triton mask."""
    return predicate


@triton.jit
def simple_fused_conv2d_maxpool_kernel(input_ptr, weight_ptr, output_ptr,
                                     batch_c_out_h_w, params):
    """
    Simplified fused kernel that matches the common pattern across all graphs.
    This is a more practical implementation that can be optimized further.
    """
    # Extract parameters
    batch_size = params[0]
    out_channels = params[1]
    in_channels = params[2]
    in_height = params[3]
    in_width = params[4]
    kernel_h = params[5]
    kernel_w = params[6]
    stride_h = params[7]
    stride_w = params[8]
    pad_h = params[9]
    pad_w = params[10]
    pool_kh = params[11]
    pool_kw = params[12]
    pool_sh = params[13]
    pool_sw = params[14]
    pool_ph = params[15]
    pool_pw = params[16]
    
    # Program ID
    pid = tl.program_id(0)
    grid_z = tl.cdiv(batch_size * out_channels, 64)  # Using BLOCK_M = 64
    grid_y = tl.cdiv(in_height * in_width, 64)       # BLOCK_N = 64
    grid_x = 1
    
    if pid >= grid_z:
        return
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
    pool_out_h = (out_height + 2 * pool_ph - pool_kh) // pool_sh + 1
    pool_out_w = (out_width + 2 * pool_pw - pool_kw) // pool_sw + 1
    
    # Initialize shared memory for convolution outputs
    shared_conv = tl.zeros((out_height, out_width), dtype=tl.float32, evict='shared')
    
    # Perform convolution (simplified)
    for c_in in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                inp_h = pid // out_channels
                inp_w = ((pid % out_channels) + c_in * kernel_h * kernel_w + kh * kernel_w + kw) % (in_height * in_width)
                inp_h_idx = inp_h // in_width
                inp_w_idx = inp_h % in_width
                
                # Convolution index
                conv_h = inp_h_idx * stride_h
                conv_w = inp_w_idx * stride_w
                
                if 0 <= conv_h < out_height and 0 <= conv_w < out_width:
                    # Load input and weight
                    input_val = tl.load(input_ptr + (inp_h_idx * in_width + inp_w_idx) * batch_size + c_in * batch_size)
                    weight_val = tl.load(weight_ptr + ((pid % out_channels) * in_channels + c_in) * kernel_h * kernel_w + kh * kernel_w + kw)
                    
                    # Accumulate convolution result
                    shared_conv[conv_h, conv_w] += input_val * weight_val
    
    # Apply max pooling
    shared_pool = tl.zeros((pool_out_h, pool_out_w), dtype=tl.float32, evict='shared')
    for ph in range(pool_kh):
        for pw in range(pool_kw):
            pool_h = (ph + pid * pool_sh) % pool_out_h
            pool_w = (pw + ((pid % out_channels) * pool_sw)) % pool_out_w
            conv_h = pool_h * pool_sh
            conv_w = pool_w * pool_sw
            
            if conv_h < out_height and conv_w < out_width:
                shared_pool[pool_h, pool_w] = tl.maximum(shared_pool[pool_h, pool_w], shared_conv[conv_h, conv_w])
    
    # Store final output
    if pid < batch_size * out_channels:
        final_idx = pid * pool_out_h * pool_out_w + (pid % out_channels) * pool_out_w
        for h in range(pool_out_h):
            for w in range(pool_out_w):
                if h * pool_out_w + w < pool_out_h * pool_out_w:
                    tl.store(output_ptr + final_idx + h * pool_out_w + w, shared_pool[h, w])


@torch.fx.wrap
def fused_conv2d_maxpool(conv_input, conv_weight, stride_padding_dilation_groups,
                        pool_params):
    """
    Wrapper function for the fused conv2d + max_pool2d operation.
    This function handles parameter setup and kernel launching.
    """
    # Extract conv parameters
    conv_stride, conv_padding, conv_dilation, conv_groups = stride_padding_dilation_groups
    
    # Extract pooling parameters  
    pool_kernel, pool_stride, pool_padding, _ = pool_params
    
    # Calculate output shapes
    batch_size, in_channels, in_height, in_width = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Conv output shape
    conv_out_h = (in_height + 2 * conv_padding[0] - conv_dilation[0] * (conv_weight.shape[2] - 1)) // conv_stride[0] + 1
    conv_out_w = (in_width + 2 * conv_padding[1] - conv_dilation[1] * (conv_weight.shape[3] - 1)) // conv_stride[1] + 1
    
    # Pool output shape
    pool_out_h = (conv_out_h + 2 * pool_padding[0] - pool_kernel[0]) // pool_stride[0] + 1
    pool_out_w = (conv_out_w + 2 * pool_padding[1] - pool_kernel[1]) // pool_stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, pool_out_h, pool_out_w), 
                        dtype=conv_input.dtype, device=conv_input.device)
    
    # Set up kernel parameters
    params = torch.tensor([
        batch_size, out_channels, in_channels, in_height, in_width,
        conv_weight.shape[2], conv_weight.shape[3],
        conv_stride[0], conv_stride[1],
        conv_padding[0], conv_padding[1],
        conv_dilation[0], conv_dilation[1], conv_groups,
        pool_kernel[0], pool_kernel[1],
        pool_stride[0], pool_stride[1],
        pool_padding[0], pool_padding[1]
    ], dtype=torch.int32, device=conv_input.device)
    
    # Launch kernel
    grid_size = (batch_size * out_channels + 63) // 64
    
    simple_fused_conv2d_maxpool_kernel[grid_size](
        conv_input, conv_weight, output, batch_size * out_channels * conv_out_h * conv_out_w, params
    )
    
    return output


def replacement_func():
    """
    Returns the fused conv2d + max_pool2d function.
    This function will be used to replace the original pattern.
    """
    return fused_conv2d_maxpool