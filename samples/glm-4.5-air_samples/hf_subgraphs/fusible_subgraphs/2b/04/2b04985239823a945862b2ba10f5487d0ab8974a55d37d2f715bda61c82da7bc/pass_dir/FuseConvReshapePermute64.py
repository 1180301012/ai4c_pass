import torch
import triton
import triton.language as tl

def pattern(in_4, tmp_3, tmp_2):
    # Conv2D + Reshape + Permute fusion pattern
    tmp_4 = torch.conv2d(in_4, tmp_3, tmp_2, (8, 8), (0, 0), (1, 1), 1)
    tmp_5 = tmp_4.reshape(32, 64, -1)
    tmp_6 = tmp_5.permute(0, 2, 1)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_3, in_2)

@triton.jit
def fused_conv_reshape_permute_kernel(
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
    stride,
    padding,
    dilation,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID and offsets
    pid = tl.program_id(0)
    m_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    
    # Load bias once per output channel
    bias = tl.load(bias_ptr + n_offset, mask=n_offset < out_channels, other=0.0)
    
    # Compute conv2 output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    total_pixels = out_height * out_width
    
    # Reshape to (pixels, batch, channels) and permute in kernel
    m_global = m_offset + pid * BLOCK_M
    mask_m = m_global < total_pixels
    
    for k_offset in range(0, in_channels, BLOCK_K):
        k_block = tl.arange(k_offset, min(k_offset + BLOCK_K, in_channels))
        mask_k = k_block < in_channels
        
        # Load weight slice
        weight_ptr_base = weight_ptr + (n_offset[:, None, None, None] * 
                                       in_channels * kernel_size * kernel_size + 
                                       k_block[None, :, None, None] * 
                                       kernel_size * kernel_size)
        weight = tl.load(weight_ptr_base + 
                        (tl.arange(BLOCK_N)[:, None, None, None] * 
                         in_channels * kernel_size * kernel_size + 
                         tl.arange(kernel_size)[None, None, :, None] * 
                         kernel_size + 
                         tl.arange(kernel_size)[None, None, None, :]),
                        mask=(n_offset[:, None, None, None] < out_channels)[:, :] &
                             (k_block[None, :, None, None] < in_channels)[:, :] &
                             (tl.arange(BLOCK_N)[:, None, None, None] < out_channels)[:, :] &
                             (tl.arange(kernel_size)[None, None, :, None] < kernel_size)[:, :] &
                             (tl.arange(kernel_size)[None, None, None, :] < kernel_size)[:, :],
                        other=0.0)
        
        # Process input pixels
        for i in range(0, batch_size):
            # Load input slice for this batch
            x_ptr_base = x_ptr + (i * in_height * in_width * in_channels + 
                                 k_block[None, None, None] * 
                                 in_height * in_width)
            x = tl.load(x_ptr_base + 
                       (tl.arange(kernel_size)[None, None, :, None] * 
                        in_width * stride + 
                        tl.arange(kernel_size)[None, None, None, :] * stride),
                       mask=(k_block[None, None, None] < in_channels)[:, :] &
                            (tl.arange(kernel_size)[None, None, :, None] < kernel_size)[:, :] &
                            (tl.arange(kernel_size)[None, None, None, :] < kernel_size)[:, :],
                       other=0.0)
            
            # Convolution computation (simplified for Triton)
            out_val = 0.0
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    for kc in range(len(k_block)):
                        input_val = x[kc, kh, kw] if kh < kernel_size and kw < kernel_size and kc < len(k_block) else 0.0
                        weight_val = weight[:, kc, kh, kw] if kc < len(k_block) and kh < kernel_size and kw < kernel_size else 0.0
                        out_val += input_val * weight_val
            
            # Apply bias
            out_val += bias[n_offset] if n_offset < out_channels else 0.0
            
            # Store output in permuted order: (pixels, batch, channels)
            out_ptr_global = out_ptr + (m_global[:, None] * batch_size * out_channels + 
                                       i * out_channels + 
                                       n_offset[None, :])
            tl.store(out_ptr_global, out_val[:, None], mask=mask_m[:, None])

@torch.fx.wrap
def fused_conv_reshape_permute(x, weight, bias, in_4):
    # Get input tensor dimensions
    batch_size, in_channels, in_height, in_width = in_4.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    stride = 1
    padding = 0
    dilation = 1
    
    # Calculate conv2d output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    total_pixels = out_height * out_width
    
    # Create output tensor with permuted shape: (pixels, batch, channels)
    output_shape = (total_pixels, batch_size, out_channels)
    out = torch.empty(output_shape, dtype=in_4.dtype, device=in_4.device)
    
    # Configure block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    # Calculate launch grid
    num_programs = (total_pixels + BLOCK_M - 1) // BLOCK_M
    
    # Launch kernel with correct grid dimensions
    fused_conv_reshape_permute_kernel[(num_programs,)](
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
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out

def replacement_func():
    return fused_conv_reshape_permute