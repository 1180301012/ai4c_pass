import torch
import triton
import triton.language as tl

# Pattern matching for Conv2D (1x1 stride 1) + MaxPool2D (3x3 stride 2) on CUDA
def pattern(*args):
    weight = args[0]
    input_in = args[1]
    tmp_0 = weight
    tmp_1 = torch.conv2d(input_in, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_0 = None
    tmp_2 = torch.nn.functional.max_pool2d(tmp_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    tmp_1 = None
    return (tmp_2,)

# Extract arguments for the replacement
def replacement_args(*args):
    weight = args[0]
    input_in = args[1]
    return (weight, input_in)

# Optimized fused kernel for Conv2D + MaxPool2D
@triton.jit
def fused_conv_maxpool_kernel_cuda(
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
    
    # Calculate batch and output position
    output_h = (in_h + 1) // 2  # MaxPool stride 2 reduces height by half
    output_w = (in_w + 1) // 2  # MaxPool stride 2 reduces width by half
    
    batch_idx = n_hw // (output_h * output_w)
    spatial_idx = n_hw % (output_h * output_w)
    output_y = spatial_idx // output_w
    output_x = spatial_idx % output_w
    
    # For conv output (which will be input to maxpool)
    conv_output_h = in_h
    conv_output_w = in_w
    
    # Convolution phase: output channels loop
    conv_output = tl.zeros((CONV_TILE_H, CONV_TILE_W), dtype=tl.float32)
    
    # Only compute this CTA's portion
    if m * BLOCK_SIZE_M >= out_channels:
        return
        
    # Compute convolution for this output channel and spatial region
    for i in range(0, in_channels):
        for kh in range(conv_kernel_h):
            for kw in range(conv_kernel_w):
                # Input coordinates with padding
                in_y = output_y * 2 - 1 + kh  # stride 2 from maxpool, plus padding
                in_x = output_x * 2 - 1 + kw
                
                # Boundary check for input
                if (0 <= in_y < in_h and 0 <= in_x < in_w and
                    0 <= m * BLOCK_SIZE_M + 0 < out_channels):
                    
                    # Load weight and input
                    weight_offset = (m * BLOCK_SIZE_M) * in_channels * conv_kernel_h * conv_kernel_w + \
                                   i * conv_kernel_h * conv_kernel_w + kh * conv_kernel_w + kw
                    input_offset = batch_idx * in_channels * conv_output_h * conv_output_w + \
                                 i * conv_output_h * conv_output_w + in_y * conv_output_w + in_x
                    
                    weight_val = tl.load(weight_ptr + weight_offset)
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Convolution accumulation
                    conv_output[in_y % CONV_TILE_H, in_x % CONV_TILE_W] += weight_val * input_val
    
    # MaxPool phase: 3x3 max pool with stride 2
    max_val = tl.float32(-3.38e+38)  # Initialize to minimum float32 value
    
    for pool_h in range(3):
        for pool_w in range(3):
            pool_y = output_y * 2 - 1 + pool_h
            pool_x = output_x * 2 - 1 + pool_w
            
            # Boundary check for conv output region
            if (0 <= pool_y < conv_output_h and 0 <= pool_x < conv_output_w):
                tile_y = pool_y % CONV_TILE_H
                tile_x = pool_x % CONV_TILE_W
                max_val = tl.maximum(max_val, conv_output[tile_y, tile_x])
    
    # Store result, with bounds checking
    if (0 <= batch_idx < batch and 0 <= m < out_channels and 
        0 <= output_y < output_h and 0 <= output_x < output_w):
        
        output_offset = batch_idx * out_channels * output_h * output_w + \
                       m * output_h * output_w + output_y * output_w + output_x
        tl.store(output_ptr + output_offset, max_val)

@torch.fx.wrap
def fused_conv_maxpool_cuda(weight, input):
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, conv_kernel_h, conv_kernel_w = weight.shape
    
    # Output dimensions after maxpool stride 2
    output_h = (in_h + 1) // 2
    output_w = (in_w + 1) // 2
    
    # Output tensor
    output = torch.empty((batch, out_channels, output_h, output_w), dtype=input.dtype, device=input.device)
    
    # Autotune optimization
    BLOCK_SIZE_M = 64  # Number of output channels per CTA
    BLOCK_SIZE_N = 2048  # Number of spatial positions per CTA
    
    # Calculate grid dimensions
    num_M = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_N = batch * output_h * output_w
    grid = (num_M, num_N)
    
    # Launch kernel
    fused_conv_maxpool_kernel_cuda[grid](
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
        CONV_TILE_H=32,
        CONV_TILE_W=32,
        POOL_TILE_H=16,
        POOL_TILE_W=16,
    )
    
    return output

def replacement_func():
    return fused_conv_maxpool_cuda