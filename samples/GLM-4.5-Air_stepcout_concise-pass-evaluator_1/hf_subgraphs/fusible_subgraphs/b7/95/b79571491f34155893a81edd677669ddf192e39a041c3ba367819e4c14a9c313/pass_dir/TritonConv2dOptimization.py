import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    # Match the exact conv2d call from the model
    result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    return result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def triton_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, 
    input_height, input_width, kernel_height, kernel_width,
    output_height, output_width,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Get program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute range each program should process
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, batch_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, out_channels)
    
    # Allocate shared memory for partial results
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels (K dimension)
    k_loop = tl.cdiv(in_channels * kernel_height * kernel_width, BLOCK_SIZE_K)
    
    for k in range(k_loop):
        # Compute range for current K block
        k_offset = k * BLOCK_SIZE_K
        k_end = min(k_offset + BLOCK_SIZE_K, in_channels * kernel_height * kernel_width)
        
        # Load weight block
        weight_offset = k_offset
        if k_end - k_offset > 0:
            # Calculate weight indices
            weight_k_idx = k_offset // (kernel_height * kernel_width)
            weight_hw_offset = k_offset % (kernel_height * kernel_width)
            weight_h = weight_hw_offset // kernel_width
            weight_w = weight_hw_offset % kernel_width
            
            # Load weight values for all output channels in this block
            weight_values = tl.load(
                weight_ptr + weight_k_idx * kernel_height * kernel_width * out_channels + 
                weight_h * kernel_width * out_channels + weight_w * out_channels +
                tl.arange(0, BLOCK_SIZE_N),
                mask=tl.arange(0, BLOCK_SIZE_N) < min(BLOCK_SIZE_N, n_end - n_start),
                other=0.0
            )
            
            # Process each batch and output channel in the block
            for m in range(m_start, m_end):
                for n in range(n_start, n_end):
                    # Calculate input region
                    in_h = (m // batch_size) * stride_h + padding_h + weight_h
                    in_w = (m % output_width) * stride_w + padding_w + weight_w
                    
                    if (0 <= in_h < input_height and 0 <= in_w < input_width and 
                        weight_k_idx < in_channels):
                        input_offset = (m // output_width) * in_channels * input_height * input_width + \
                                     weight_k_idx * input_height * input_width + \
                                     in_h * input_width + in_w
                        
                        input_val = tl.load(
                            input_ptr + input_offset,
                            mask=True,
                            other=0.0
                        )
                        
                        accumulator[m - m_start, n - n_start] += input_val * weight_values[n - n_start]
    
    # Load bias
    bias_values = tl.load(
        bias_ptr + tl.arange(0, BLOCK_SIZE_N),
        mask=tl.arange(0, BLOCK_SIZE_N) < min(BLOCK_SIZE_N, n_end - n_start),
        other=0.0
    )
    
    # Add bias and store result
    for m in range(m_start, m_end):
        output_base = (m // output_width) * out_channels * output_height * output_width + \
                     (m % output_width) * out_channels
        
        for n in range(n_start, n_end):
            output_offset = output_base + n
            tl.store(
                output_ptr + output_offset,
                accumulator[m - m_start, n - n_start] + bias_values[n - n_start],
                mask=(m < batch_size) and (n < out_channels)
            )

@torch.fx.wrap
def triton_conv2d(input_tensor, weight_tensor, bias_tensor):
    # Get tensor shapes
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, kernel_in_channels, kernel_height, kernel_width = weight_tensor.shape
    
    # Verify input channels match
    assert in_channels == kernel_in_channels, f"Input channels {in_channels} must match weight in channels {kernel_in_channels}"
    
    # Calculate output dimensions
    output_height = (input_height + 2 * 0 - 1 * (kernel_height - 1) - 1) // 1 + 1
    output_width = (input_width + 2 * 0 - 1 * (kernel_width - 1) - 1) // 1 + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block sizes for GPU optimization
    BLOCK_SIZE_M = 8  # Batch dimension
    BLOCK_SIZE_N = 64  # Output channels dimension
    BLOCK_SIZE_K = 64  # Input channels * kernel spatial dimension
    
    # Launch kernel
    grid = (
        triton.cdiv(batch_size * output_height * output_width, BLOCK_SIZE_M),
        triton.cdiv(out_channels, BLOCK_SIZE_N),
    )
    
    triton_conv2d_kernel[grid](
        input_tensor, weight_tensor, bias_tensor, output,
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_height, kernel_width,
        output_height, output_width,
        1, 1, 0, 0, 1, 1,  # stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return triton_conv2d