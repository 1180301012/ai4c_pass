import torch
import triton
import triton.language as tl

# Pattern matching for Conv2D (3x3 stride 2) + MaxPool2D (3x3 stride 2) on CPU
def pattern(*args):
    weight = args[0]
    input_in = args[1]
    tmp_0 = weight
    tmp_1 = input_in
    tmp_2 = torch.conv2d(tmp_1, tmp_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.max_pool2d(tmp_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    tmp_2 = None
    return (tmp_3,)

# Extract arguments for the replacement
def replacement_args(*args):
    weight = args[0]
    input_in = args[1]
    return (weight, input_in)

# Optimized fused kernel for Conv2D + MaxPool2D
@triton.jit
def fused_conv_maxpool_kernel_cpu(
    weight_ptr,      # [out_channels, in_channels, kernel_h, kernel_w]
    input_ptr,       # [batch, in_channels, input_h, input_w] 
    output_ptr,      # [batch, out_channels, output_h, output_w]
    batch, in_channels, in_h, in_w, out_channels,
    conv_kernel_h, conv_kernel_w,
    BLOCK_SIZE_M: tl.constexpr,    # Number of K output channels per CTA
    BLOCK_SIZE_N: tl.constexpr,    # Number of NHW output elements per CTA
    CONV_TILE_H: tl.constexpr,     # Tile size for convolution spatial dimensions
    CONV_TILE_W: tl.constexpr,     # Tile size for convolution spatial dimensions
    POOL_TILE_H: tl.constexpr,     # Tile size for pooling spatial dimensions
    POOL_TILE_W: tl.constexpr,     # Tile size for pooling spatial dimensions
):
    # Program ID mapping: M (output channels), NHW (spatial positions)
    m = tl.program_id(0)
    n_hw = tl.program_id(1)
    
    # Calculate output dimensions
    output_h = (in_h + 1) // 4  # Conv stride 2, then MaxPool stride 2 = total stride 4
    output_w = (in_w + 1) // 4  # Conv stride 2, then MaxPool stride 2 = total stride 4
    
    batch_idx = n_hw // (output_h * output_w)
    spatial_idx = n_hw % (output_h * output_w)
    output_y = spatial_idx // output_w
    output_x = spatial_idx % output_w
    
    # Calculate intermediate conv output dimensions
    conv_output_h = (in_h + 1) // 2  # Conv stride 2
    conv_output_w = (in_w + 1) // 2  # Conv stride 2
    
    # Only compute this CTA's portion
    if m * BLOCK_SIZE_M >= out_channels:
        return
    
    # Convolution phase: compute for this output channel
    conv_output = tl.zeros((CONV_TILE_H, CONV_TILE_W), dtype=tl.float32)
    
    # Compute convolution for all input channels at the maxpool output location
    for i in range(in_channels):
        for kh in range(conv_kernel_h):
            for kw in range(conv_kernel_w):
                # Compute input coordinates for this convolution kernel element
                # This maps from maxpool output back to conv output, then to input
                conv_out_y = output_y * 2 + kh - 1  # MaxPool stride maps conv output to maxpool output
                conv_out_x = output_x * 2 + kw - 1  # MaxPool stride maps conv output to maxpool output
                in_y = conv_out_y * 2 + kh - 1      # Conv stride maps input to conv output  
                in_x = conv_out_x * 2 + kw - 1      # Conv stride maps input to conv output
                
                # Boundary check for input
                if (0 <= in_y < in_h and 0 <= in_x < in_w and
                    0 <= conv_out_y < conv_output_h and 0 <= conv_out_x < conv_output_w and
                    0 <= m * BLOCK_SIZE_M < out_channels):
                    
                    # Load weight and input
                    weight_offset = (m * BLOCK_SIZE_M) * in_channels * conv_kernel_h * conv_kernel_w + \
                                   i * conv_kernel_h * conv_kernel_w + kh * conv_kernel_w + kw
                    input_offset = batch_idx * in_channels * in_h * in_w + \
                                 i * in_h * in_w + in_y * in_w + in_x
                    
                    weight_val = tl.load(weight_ptr + weight_offset)
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Convolution accumulation at conv output location
                    conv_output[conv_out_y % CONV_TILE_H, conv_out_x % CONV_TILE_W] += weight_val * input_val
    
    # MaxPool phase: 3x3 max pool with stride 2 on conv output
    max_val = tl.float32(-3.38e+38)  # Initialize to minimum float32 value
    
    for pool_h in range(3):
        for pool_w in range(3):
            pool_y = output_y * 2 + pool_h - 1  # stride 2 maps conv output to maxpool output
            pool_x = output_x * 2 + pool_w - 1  # stride 2 maps conv output to maxpool output
            
            # Boundary check for conv output region
            if (0 <= pool_y < conv_output_h and 0 <= pool_x < conv_output_w):
                tile_y = pool_y % CONV_TILE_H
                tile_x = pool_x % CONV_TILE_W
                max_val = tl.maximum(max_val, conv_output[tile_y, tile_x])
    
    # Store final result, with bounds checking
    if (0 <= batch_idx < batch and 0 < m < out_channels and 
        0 <= output_y < output_h and 0 <= output_x < output_w):
        
        output_offset = batch_idx * out_channels * output_h * output_w + \
                       m * output_h * output_w + output_y * output_w + output_x
        tl.store(output_ptr + output_offset, max_val)

@torch.fx.wrap
def fused_conv_maxpool_cpu(weight, input):
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, conv_kernel_h, conv_kernel_w = weight.shape
    
    # Output dimensions after conv stride 2 and maxpool stride 2 (total stride 4)
    output_h = (in_h + 1) // 4
    output_w = (in_w + 1) // 4
    
    # Output tensor
    output = torch.empty((batch, out_channels, output_h, output_w), dtype=input.dtype, device=input.device)
    
    # Optimization parameters for CPU
    BLOCK_SIZE_M = 32   # Number of output channels per CTA (smaller for CPU)
    BLOCK_SIZE_N = 512  # Number of spatial positions per CTA
    
    # Calculate grid dimensions
    num_M = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_N = batch * output_h * output_w
    grid = (num_M, num_N)
    
    # Launch kernel
    fused_conv_maxpool_kernel_cpu[grid](
        weight_ptr=weight,
        input_ptr=input,
        output_ptr=output,
        batch=batch,
        in_channels=in_channels,
        in_h=in_h,
        in_w=in_w,
        out_channels=out_channels,
        conv_kernel_h=conv_kernel_h,
        conv_kernel_w=conv_kernel_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        CONV_TILE_H=16,
        CONV_TILE_W=16,
        POOL_TILE_H=8,
        POOL_TILE_W=8,
    )
    
    return output

def replacement_func():
    return fused_conv_maxpool_cpu