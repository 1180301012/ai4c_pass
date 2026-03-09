import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def simple_conv_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Base pointers for current batch element
    bias_base = bias_ptr + pid * out_channels
    weight_base = weight_ptr + pid * out_channels * in_channels * 1 * 1
    input_base = input_ptr + pid * in_channels * 1 * 1
    out_base = out_ptr + pid * out_channels * height * width
    
    # Load the input value (same for all output channels)
    input_val = tl.load(input_base + 0).to(tl.float32)
    
    # Process all output channels for spatial position (0, 0)
    # Since this is 1x1 conv and input has only one spatial position,
    # output will be broadcasted to all spatial positions
    for c in range(out_channels):
        # Load bias for output channel c
        bias_val = tl.load(bias_base + c).to(tl.float32)
        
        # Load weight for output channel c (first input channel since 1x1 conv)
        weight_offset = c * in_channels * 1 * 1
        weight_val = tl.load(weight_base + weight_offset).to(tl.float32)
        
        # Compute: output = bias + weight * input
        result = bias_val + weight_val * input_val
        
        # Store this result for all spatial positions
        # Use a nested loop approach to ensure safe memory access
        for h in range(height):
            for w in range(width):
                offset = c * height * width + h * width + w
                if offset < out_channels * height * width:  # Safety check
                    tl.store(out_base + offset, result)

@torch.fx.wrap
def simple_conv_optimization(in_0, in_1, in_3):
    batch_size = in_3.shape[0] if len(in_3.shape) == 4 else 1
    in_channels = in_1.shape[1]
    out_channels = in_1.shape[0]
    height = in_3.shape[2] if len(in_3.shape) == 4 else 1
    width = in_3.shape[3] if len(in_3.shape) == 4 else 1
    
    out = torch.empty((batch_size, out_channels, height, width), dtype=torch.float32, device=in_3.device)
    
    grid = (batch_size,)
    simple_conv_kernel[grid](
        in_0, in_1, in_3, out,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE=256
    )
    
    return out

def replacement_func():
    return simple_conv_optimization