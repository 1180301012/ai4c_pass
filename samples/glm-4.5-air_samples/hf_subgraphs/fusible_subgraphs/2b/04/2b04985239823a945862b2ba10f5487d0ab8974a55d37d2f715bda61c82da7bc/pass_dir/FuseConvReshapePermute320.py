import torch
import triton
import triton.language as tl

def pattern(in_4, tmp_3, tmp_2):
    # Conv2D + Reshape + Permute fusion pattern for 320 channels
    tmp_4 = torch.conv2d(in_4, tmp_3, tmp_2, (2, 2), (0, 0), (1, 1), 1)
    tmp_5 = tmp_4.reshape(32, 320, -1)
    tmp_6 = tmp_5.permute(0, 2, 1)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_3, in_2)

@triton.jit
def fused_conv_reshape_permute_320_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    kernel_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ID and offsets
    pid = tl.program_id(0)
    m_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    
    # Load bias once per output channel
    bias = tl.load(bias_ptr + n_offset, mask=n_offset < out_channels, other=0.0)
    
    # Compute conv2 output dimensions
    out_height = (in_height + 2 * 0 - 1 * (kernel_size - 1) - 1) // 1 + 1
    out_width = (in_width + 2 * 0 - 1 * (kernel_size - 1) - 1) // 1 + 1
    total_pixels = out_height * out_width
    
    # Reshape to (pixels, batch, channels) and permute in kernel
    m_global = m_offset + pid * BLOCK_M
    mask_m = m_global < total_pixels
    
    for k_offset in range(0, in_channels, BLOCK_N):
        k_end = min(k_offset + BLOCK_N, in_channels)
        k_block = tl.arange(k_offset, k_end)
        
        # Process input pixels
        for i in range(0, batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    m_idx = h * out_width + w
                    if m_idx >= total_pixels:
                        continue
                        
                    # Accumulate convolution result
                    out_val = bias[0]  # Initialize with bias
                    for kc in range(len(k_block)):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                # Calculate input coordinates
                                ih = h * 1 + kh - 0
                                iw = w * 1 + kw - 0
                                
                                if ih >= 0 and ih < in_height and iw >= 0 and iw < in_width:
                                    # Load input value
                                    input_idx = i * in_height * in_width * in_channels + \
                                               k_block[kc] * in_height * in_width + \
                                               ih * in_width + iw
                                    input_val = tl.load(x_ptr + input_idx, other=0.0)
                                    
                                    # Load weight value
                                    weight_idx = n_offset[0] * in_channels * kernel_size * kernel_size + \
                                                k_block[kc] * kernel_size * kernel_size + \
                                                kh * kernel_size + kw
                                    weight_val = tl.load(weight_ptr + weight_idx, other=0.0)
                                    
                                    out_val += input_val * weight_val
                    
                    # Store output in permuted order: (pixels, batch, channels)
                    out_ptr_global = out_ptr + (m_idx * batch_size * out_channels + \
                                               i * out_channels + \
                                               n_offset[0])
                    tl.store(out_ptr_global, out_val, mask=(m_idx < total_pixels) & (n_offset[0] < out_channels))

@torch.fx.wrap
def fused_conv_reshape_permute_320(x, weight, bias, in_4):
    # Get input tensor dimensions
    batch_size, in_channels, in_height, in_width = in_4.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    
    # Calculate conv2d output dimensions
    out_height = (in_height + 2 * 0 - 1 * (kernel_size - 1) - 1) // 1 + 1
    out_width = (in_width + 2 * 0 - 1 * (kernel_size - 1) - 1) // 1 + 1
    total_pixels = out_height * out_width
    
    # Create output tensor with permuted shape: (pixels, batch, channels)
    output_shape = (total_pixels, batch_size, out_channels)
    out = torch.empty(output_shape, dtype=in_4.dtype, device=in_4.device)
    
    # Configure block sizes - optimized for 320 channels
    BLOCK_M = 256  # Process 256 pixels at a time
    BLOCK_N = 320  # Process all output channels at once
    
    # Calculate launch grid
    num_programs = (total_pixels + BLOCK_M - 1) // BLOCK_M
    
    # Launch kernel with correct grid dimensions
    fused_conv_reshape_permute_320_kernel[(num_programs,)](
        x_ptr=in_4,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        kernel_size=kernel_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return out

def replacement_func():
    return fused_conv_reshape_permute_320