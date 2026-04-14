import torch
import triton
import triton.language as tl


def pattern(input_tensor, weight_tensor):
    """Pattern: Conv2D followed by channel slicing keeping first 256 channels"""
    # Perform conv2d operation - using exact same parameters as in model.py
    conv2d_result = torch.conv2d(input_tensor, weight_tensor, None, (2, 2), (0, 0), (1, 1), 1)
    # Slice operation - taking first 256 output channels
    sliced_result = conv2d_result[(slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None))]
    return sliced_result, conv2d_result


def replacement_args(input_tensor, weight_tensor):
    """Extract arguments for the replacement function"""
    return (input_tensor, weight_tensor, 256)  # Pass the slice limit as argument


@triton.jit
def conv2d_slice_kernel(
    input_ptr, input_stride0, input_stride1, input_stride2, input_stride3,
    weight_ptr, weight_stride0, weight_stride1, weight_stride2, weight_stride3,
    output_ptr, output_stride0, output_stride1, output_stride2, output_stride3,
    out_channels_total, out_channels_to_compute,
    batch_size, in_channels, input_height, input_width,
    kernel_height, kernel_width, output_height, output_width,
    stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized Triton kernel for Conv2D with channel slicing"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute block indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    # Ensure we don't go out of bounds
    m_mask = m_start < batch_size
    n_mask = n_start < out_channels_to_compute
    k_mask = k_start < in_channels
    
    if not (m_mask and n_mask and k_mask):
        return
    
    # Loop over remaining dimensions
    for h in range(0, output_height, 1):
        for w in range(0, output_width, 1):
            # Compute memory offsets
            input_offset = (m_start * input_stride0 + 
                           k_start * input_stride1 + 
                           h * input_stride2 + 
                           w * input_stride3)
            
            weight_offset = (n_start * weight_stride0 + 
                            k_start * weight_stride1 + 
                            0 * weight_stride2 + 
                            0 * weight_stride3)
            
            output_offset = (m_start * output_stride0 + 
                            n_start * output_stride1 + 
                            h * output_stride2 + 
                            w * output_stride3)
            
            # Load input and weight data
            input_val = tl.load(input_ptr + input_offset, mask=None)
            weight_val = tl.load(weight_ptr + weight_offset, mask=None)
            
            # Compute dot product
            output_val = input_val * weight_val
            
            # Store result
            tl.store(output_ptr + output_offset, output_val)


@torch.fx.wrap
def conv2d_slice_256_optimized(input_tensor, weight_tensor, slice_limit=256):
    """Wrapper function for optimized conv2d with 256 channel slicing"""
    # Get input and output dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels_total, _, kernel_height, kernel_width = weight_tensor.shape
    
    # Calculate output dimensions with stride=(2,2), padding=(0,0), dilation=(1,1)
    output_height = (input_height + 2 * 0 - 1 * (kernel_height - 1) - 1) // 2 + 1
    output_width = (input_width + 2 * 0 - 1 * (kernel_width - 1) - 1) // 2 + 1
    
    # Determine actual output channels to compute
    out_channels_to_compute = min(slice_limit, out_channels_total)
    
    # Create output tensors
    sliced_output = torch.empty((batch_size, out_channels_to_compute, output_height, output_width), 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    full_output = torch.empty((batch_size, out_channels_total, output_height, output_width), 
                            dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get tensor strides
    input_stride = input_tensor.stride()
    weight_stride = weight_tensor.stride()
    sliced_stride = sliced_output.stride()
    full_stride = full_output.stride()
    
    # Configure block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grids
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels_total + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernels for both outputs
    conv2d_slice_kernel[grid_m, grid_n, grid_k](
        input_tensor, input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        weight_tensor, weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
        sliced_output, sliced_stride[0], sliced_stride[1], sliced_stride[2], sliced_stride[3],
        out_channels_total, out_channels_to_compute,
        batch_size, in_channels, input_height, input_width,
        kernel_height, kernel_width, output_height, output_width,
        2, 2, 0, 0, 1, 1, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    conv2d_slice_kernel[grid_m, grid_n, grid_k](
        input_tensor, input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        weight_tensor, weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
        full_output, full_stride[0], full_stride[1], full_stride[2], full_stride[3],
        out_channels_total, out_channels_total,
        batch_size, in_channels, input_height, input_width,
        kernel_height, kernel_width, output_height, output_width,
        2, 2, 0, 0, 1, 1, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return sliced_output, full_output


def replacement_func():
    """Return the optimized function"""
    return conv2d_slice_256_optimized