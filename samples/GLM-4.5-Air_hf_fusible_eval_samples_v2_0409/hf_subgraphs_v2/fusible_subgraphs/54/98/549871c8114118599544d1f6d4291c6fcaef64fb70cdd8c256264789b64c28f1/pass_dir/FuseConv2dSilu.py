import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Pattern: Conv2D + SILU fusion - exact replication of model structure"""
    # Conv2D operation with EXACT parameters from model: torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # SILU activation with EXACT parameters from model: torch.nn.functional.silu(conv2d, inplace = False)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace = False)
    # Return the SILU result (this is what gets used in the model before dropout)
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    """Extract arguments for the fused kernel"""
    return (in_2, in_1, in_0)

@triton.jit
def conv2d_silu_kernel_1x1(
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels,
    BLOCK_SIZE_M: tl.constexpr,  # Output channels per CTA  
    BLOCK_SIZE_N: tl.constexpr,  # Pixels per CTA
):
    # Program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Spatial pixel block
    
    # Compute ranges for this block
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, output_channels)
    
    # Linear index for the pixel (flatten spatial dimensions)
    spatial_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    h = spatial_idx // input_width
    w = spatial_idx % input_width
    
    # Mask to ensure we're within bounds
    mask = h < input_height
    
    # Load biases for this output channel block
    bias_ptrs = bias_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)
    bias = tl.load(bias_ptrs, mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)) < output_channels, other=0.0)
    
    # For 1x1 convolution, we process each pixel independently
    for i in range(BLOCK_SIZE_N):
        if mask[i]:
            h_idx = h[i]
            w_idx = w[i]
            
            # Load input pixel (all channels at this spatial location)
            input_ptrs = input_ptr + input_batch * input_channels * input_height * input_width + \
                       tl.arange(0, input_channels) * input_height * input_width + h_idx * input_width + w_idx
            input_vals = tl.load(input_ptrs, mask=tl.arange(0, input_channels) < input_channels, other=0.0)
            
            # Load weights for this output channel block
            weight_ptrs = weight_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)).to(tl.int64) * \
                        input_channels + tl.arange(0, input_channels)
            weight_vals = tl.load(weight_ptrs, mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)) < output_channels, other=0.0)
            
            # 1x1 Conv2D operation: dot product of input channels with weights
            conv_val = tl.sum(input_vals * weight_vals, axis=0)
            
            # Add bias and apply SILU activation: x * sigmoid(x + bias)
            silu_val = conv_val * tl.sigmoid(conv_val + bias)
            
            # Store output for this pixel and output channel block
            output_ptrs = output_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)).to(tl.int64) * \
                        input_height * input_width + h_idx * input_width + w_idx
            tl.store(output_ptrs, silu_val, mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, m_end - pid_m * BLOCK_SIZE_M)) < output_channels)

@torch.fx.wrap
def fused_conv2d_silu(in_2, in_1, in_0):
    """Fused Conv2D + SILU operation optimized for 1x1 convolution"""
    input_shape = in_2.shape
    weight_shape = in_1.shape
    
    # Input dimensions
    input_batch, input_channels, input_height, input_width = input_shape
    output_channels, input_channels_w, kernel_height, kernel_width = weight_shape
    
    # Verify this is a 1x1 convolution
    assert kernel_height == 1 and kernel_width == 1, "This kernel is optimized only for 1x1 convolutions"
    
    # Output dimensions (same as input due to 1x1 convolution)
    output_height = input_height
    output_width = input_width
    
    # Initialize output tensor
    output_shape = (input_batch, output_channels, output_height, output_width)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Block sizes optimized for our specific tensor shapes
    BLOCK_SIZE_M = 256  # Output channels per CTA (divides 256 evenly)
    BLOCK_SIZE_N = 128  # Pixels per CTA (4 * 256 = 1024 pixels total, 128 pixels per CTA)
    
    # Launch grid
    num_ctas_m = (output_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pixels = input_height * input_width
    num_ctas_n = (num_pixels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    conv2d_silu_kernel_1x1[(num_ctas_m, num_ctas_n)](
        in_2,
        in_1,
        in_0,
        output,
        input_batch, input_channels, input_height, input_width,
        output_channels,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    """Return the fused Conv2D+Silu function"""
    return fused_conv2d_silu