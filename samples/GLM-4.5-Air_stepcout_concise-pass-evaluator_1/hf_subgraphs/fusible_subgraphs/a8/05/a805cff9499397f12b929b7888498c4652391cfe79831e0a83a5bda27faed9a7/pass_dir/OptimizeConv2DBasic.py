import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    # Simple pattern: just match conv2d
    result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (16, 16), (0, 0), (1, 1), 1)
    return result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def simple_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, height, width, out_channels,
    kh, kw, stride_h, stride_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate output shape
    out_height = height // stride_h  # 224 / 16 = 14
    out_width = width // stride_w    # 224 / 16 = 14
    total_patches = out_height * out_width  # 14 * 14 = 196
    
    # Determine which patch and channel we're computing
    patch_idx = pid_m
    channel_idx = pid_n
    
    # Convert to 2D patch coordinates
    patch_h = patch_idx // out_width
    patch_w = patch_idx % out_width
    
    # Get bias
    bias_val = tl.load(bias_ptr + channel_idx)
    accumulator = bias_val
    
    # Do convolution
    for ki in range(kh):
        for kj in range(kw):
            # Input position
            ih = patch_h * stride_h + ki
            iw = patch_w * stride_w + kj
            
            if ih < height and iw < width:
                # Input pointer index
                input_idx = (0 * 3 + 0) * height * width + ih * width + iw
                # Weight pointer index (assuming weight shape [out_channels, in_channels, kh, kw] = [768, 3, 16, 16])
                weight_idx = channel_idx * 3 * kh * kw + 0 * kh * kw + ki * kw + kj
                
                input_val = tl.load(input_ptr + input_idx)
                weight_val = tl.load(weight_ptr + weight_idx)
                accumulator += input_val * weight_val
    
    # Store result
    output_idx = patch_idx * out_channels + channel_idx
    tl.store(output_ptr + output_idx, accumulator)

@torch.fx.wrap
def optimized_conv2d(input_tensor, weight_tensor, bias_tensor):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kh, kw = weight_tensor.shape
    
    # Conv2d params
    stride_h, stride_w = 16, 16
    padding_h, padding_w = 0, 0
    dilation_h, dilation_w = 1, 1
    groups = 1
    
    # Output shape calculation
    out_height = (height + 2 * padding_h - dilation_h * (kh - 1) - 1) // stride_h + 1
    out_width = (width + 2 * padding_w - dilation_w * (kw - 1) - 1) // stride_w + 1
    total_patches = out_height * out_width  # Should be 196
    
    # Create output tensor in correct format: [batch, patches, channels] = [1, 196, 768]
    output = torch.empty(batch_size, total_patches, out_channels,
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch grid
    num_patches = total_patches
    num_channels = out_channels
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    
    num_patches_programs = (num_patches + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_channels_programs = (num_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    simple_conv2d_kernel[(num_patches_programs, num_channels_programs)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        kh=kh,
        kw=kw,
        stride_h=stride_h,
        stride_w=stride_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return optimized_conv2d