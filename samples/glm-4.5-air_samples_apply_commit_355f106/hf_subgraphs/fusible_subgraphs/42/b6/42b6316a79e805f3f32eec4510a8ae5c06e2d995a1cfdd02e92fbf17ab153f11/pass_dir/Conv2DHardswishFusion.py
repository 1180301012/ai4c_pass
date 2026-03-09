import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    conv_out = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hardswish_out = torch.nn.functional.hardswish(conv_out, True)
    return hardswish_out

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def conv2d_hardswish_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_batch, n_channels_out, n_channels_in,
    height_in, width_in, height_out, width_out,
    MAX_CHANNELS: tl.constexpr,
):
    # Program ID and position calculation
    batch_id = tl.program_id(0)
    channel_out = tl.program_id(1)
    element_id = tl.program_id(2)
    
    # Calculate input and output pointers
    input_base = input_ptr + batch_id * n_channels_in * height_in * width_in
    weight_base = weight_ptr + channel_out * n_channels_in * 1 * 1
    bias_base = bias_ptr + channel_out
    output_base = output_ptr + batch_id * n_channels_out * height_out * width_out + channel_out * height_out * width_out
    
    # Since we have 1x1 kernels, we can process each output element independently
    if element_id < height_out * width_out:
        output_idx = channel_out * height_out * width_out + batch_id * n_channels_out * height_out * width_out
        ptr_base = input_base + element_id
        
        # Process all 960 channels efficiently in chunks
        conv_result = 0.0
        
        # Process channels in chunks of 32 to handle all 960 channels
        for i in range(32):
            if i < n_channels_in:
                # Input: position batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                
                # Weights: position channel_out * n_channels_in + i
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(32, 64):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(64, 96):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(96, 128):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(128, 160):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(160, 192):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(192, 224):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(224, 256):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(256, 288):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(288, 320):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(320, 352):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(352, 384):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(384, 416):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(416, 448):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(448, 480):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(480, 512):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(512, 544):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(544, 576):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(576, 608):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(608, 640):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(640, 672):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(672, 704):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(704, 736):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(736, 768):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(768, 800):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(800, 832):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(832, 864):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(864, 896):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(896, 928):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
                
        for i in range(928, 960):
            if i < n_channels_in:
                input_ptr_val = input_ptr + batch_id * n_channels_in * height_in * width_in + i * height_in * width_in + element_id
                weight_ptr_val = weight_ptr + channel_out * n_channels_in + i
                
                input_val = tl.load(input_ptr_val)
                weight_val = tl.load(weight_ptr_val)
                
                conv_result += input_val * weight_val
        
        # Load bias
        bias_val = tl.load(bias_base)
        
        # Apply hardswish: x * relu6(x + 3) / 6
        x = conv_result + bias_val
        relu6_val = tl.maximum(0.0, tl.minimum(x, 6.0))
        hardswish_result = x * relu6_val / 6.0
        
        # Store result
        output_idx = output_base + element_id
        tl.store(output_ptr + output_idx, hardswish_result, mask=element_id < height_out * width_out)

@torch.fx.wrap
def conv2d_hardswish_fused(input, weight, bias):
    n_batch, n_channels_in, height_in, width_in = input.shape
    n_channels_out, _, _, _ = weight.shape
    
    # For 1x1 convolution with stride 1x1, output dimensions equal input dimensions
    height_out, width_out = height_in, width_in
    
    # Determine output shape
    output_shape = (n_batch, n_channels_out, height_out, width_out)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Calculate grid dimensions
    n_elements_out = height_out * width_out
    
    # Set MAX_CHANNELS to safely handle the maximum input channels (960 from the meta + padding)
    MAX_CHANNELS = 1024
    
    # Use different block sizes for better performance
    BLOCK_SIZE = 32
    
    # Create grid: (batch, output_channels, elements_per_output_channel)
    grid = (
        (n_batch + 63) // 64,  # Adjust grid size for batch
        (n_channels_out + 63) // 64,  # Adjust grid size for output channels  
        (n_elements_out + BLOCK_SIZE - 1) // BLOCK_SIZE,  # Elements per output channel
    )
    
    conv2d_hardswish_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_batch=n_batch,
        n_channels_out=n_channels_out,
        n_channels_in=n_channels_in,
        height_in=height_in,
        width_in=width_in,
        height_out=height_out,
        width_out=width_out,
        MAX_CHANNELS=MAX_CHANNELS,
    )
    
    return output

def replacement_func():
    return conv2d_hardswish_fused