import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Pattern: Conv2D only for testing"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def simple_conv2d_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Spatial position block
    
    # Compute ranges for this block
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, output_channels)
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, input_height * input_width)
    
    # Process spatial positions
    for n in range(pid_n * BLOCK_SIZE_N, n_end):
        # Convert linear index to 2D position
        h = n // input_width
        w = n % input_width
        
        # Load input pixel (all channels at this spatial location)
        input_ptrs = input_ptr + input_batch * input_channels * input_height * input_width + \
                   tl.arange(0, input_channels) * input_height * input_width + h * input_width + w
        input_vals = tl.load(input_ptrs, mask=tl.arange(0, input_channels) < input_channels, other=0.0)
        
        # Load weights for this output channel block
        weight_ptrs = weight_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)).to(tl.int64) * \
                    input_channels + tl.arange(0, input_channels)
        weight_vals = tl.load(weight_ptrs, mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)) < output_channels, other=0.0)
        
        # Biases for this output channel block
        bias_ptrs = bias_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)
        bias_vals = tl.load(bias_ptrs, mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)) < output_channels, other=0.0)
        
        # 1x1 Conv2D operation: dot product of input channels with weights
        conv_vals = tl.sum(input_vals * weight_vals, axis=0) + bias_vals
        
        # Store output for this pixel and output channel block
        output_ptrs = output_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)).to(tl.int64) * \
                    input_height * input_width + h * input_width + w
        tl.store(output_ptrs, conv_vals, mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)) < output_channels)

@torch.fx.wrap
def simple_conv2d(in_2, in_1, in_0):
    input_shape = in_2.shape
    weight_shape = in_1.shape
    
    input_batch, input_channels, input_height, input_width = input_shape
    output_channels, _, _, _ = weight_shape
    
    output_shape = (input_batch, output_channels, input_height, input_width)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 128
    
    num_ctas_m = (output_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pixels = input_height * input_width
    num_ctas_n = (num_pixels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    simple_conv2d_kernel[(num_ctas_m, num_ctas_n)](
        in_2, in_1, in_0, output,
        input_batch, input_channels, input_height, input_width,
        output_channels, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return simple_conv2d