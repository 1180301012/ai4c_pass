import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    """Pattern matching for conv2d operation with specific parameters"""
    # This needs to match the exact signature used in the model
    result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)
    return result

def replacement_args(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    """Extract arguments for the optimized conv2d kernel"""
    return (input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels, kernel_height, kernel_width,
    output_height, output_width,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """Optimized Triton kernel for conv2d operation"""
    # Program ids for tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Generate offsets for output tensor
    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = m_offs < output_batch
    n_mask = n_offs < output_channels
    
    # Calculate input tensor offsets with spatial tiling
    h_offs = tl.arange(0, BLOCK_SIZE_H)
    w_offs = tl.arange(0, BLOCK_SIZE_W)
    
    # Load bias
    bias_ptrs = bias_ptr + n_offs
    bias = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    
    # Accumulate result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input channels and kernel spatial dimensions
    for c_acc in range(0, input_channels, BLOCK_SIZE_K):
        for kh_acc in range(0, kernel_height):
            for kw_acc in range(0, kernel_width):
                # Calculate input coordinates
                in_h = (m_offs.reshape(1, -1) // output_width * stride_h + 
                       kh_acc * dilation_h).to(tl.int32)
                in_w = (m_offs.reshape(1, -1) % output_width * stride_w + 
                       kw_acc * dilation_w).to(tl.int32)
                
                # Load input and weight
                input_ptrs = input_ptr + (
                    m_offs.reshape(-1, 1) * input_channels * input_height * input_width +
                    c_acc.reshape(-1, 1) * input_height * input_width +
                    in_h.reshape(-1, 1) * input_width +
                    in_w.reshape(-1, 1)
                )
                weight_ptrs = weight_ptr + (
                    n_offs.reshape(1, -1) * input_channels * kernel_height * kernel_width +
                    c_acc.reshape(1, -1) * kernel_height * kernel_width +
                    kh_acc.reshape(1, -1) * kernel_width +
                    kw_acc.reshape(1, -1)
                )
                
                # Load with masking
                input_mask = (in_h >= 0) & (in_h < input_height) & (in_w >= 0) & (in_w < input_width)
                input_data = tl.load(input_ptrs, mask=input_mask, other=0.0)
                weight_data = tl.load(weight_ptrs, mask=(n_mask.reshape(1, -1) & tl.arange(0, len(input_channels)) < BLOCK_SIZE_K), other=0.0)
                
                # Matrix multiplication
                acc += input_data.to(tl.float32) @ weight_data.to(tl.float32)
    
    # Add bias and store result
    acc = acc + bias.reshape(1, -1)
    output_ptrs = output_ptr + (
        m_offs.reshape(-1, 1) * output_channels * output_height * output_width +
        n_offs.reshape(1, -1) * output_height * output_width
    )
    output_mask = m_mask.reshape(-1, 1) & (n_mask.reshape(1, -1) * output_height * output_width).reshape(-1, 1)
    tl.store(output_ptrs, acc.to(output_ptr.dtype.element_type), mask=output_mask)

@torch.fx.wrap
def optimized_conv2d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    """Wrapper function for optimized conv2d"""
    input_shape = input_tensor.shape
    weight_shape = weight_tensor.shape
    
    input_batch, input_channels, input_height, input_width = input_shape
    output_channels, _, kernel_height, kernel_width = weight_shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    
    # Output tensor
    output = torch.empty((input_batch, output_channels, output_height, output_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set tile sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid size
    grid_m = (input_batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    optimized_conv2d_kernel[(grid_m, grid_n), (
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_H, BLOCK_SIZE_W
    )](
        input_tensor, weight_tensor, bias_tensor, output,
        input_batch, input_channels, input_height, input_width,
        output_channels, kernel_height, kernel_width,
        output_height, output_width,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1], groups
    )
    
    return output

def replacement_func():
    """Return the optimized conv2d function"""
    return optimized_conv2d