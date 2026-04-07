import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    """Pattern: simple conv2d + interpolate"""
    tmp = torch.conv2d(a, b, c, (1, 1), (0, 0), (1, 1), 1)
    res = torch.nn.functional.interpolate(tmp, size=(512, 512), mode='bilinear', align_corners=False)
    return res

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def fused_conv_interpolate_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    scale_factor: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # For this specific pattern: [1,512,128,128] -> [1,150,512,512] with 1x1 conv + 4x upsample
    # Each thread handles one output pixel and one output channel
    
    # Calculate output coordinates (within [0, 511, 511] for [1,150,512,512])
    batch = pid // (150 * 512 * 512)  # Should always be 0
    out_c = (pid // (512 * 512)) % 150
    out_h = (pid // 512) % 512
    out_w = pid % 512
    
    # Map back to original coordinates before 4x upsample
    orig_out_h = out_h // 4
    orig_out_w = out_w // 4
    
    # Only compute if within original bounds
    if orig_out_h < 128 and orig_out_w < 128:
        # For 1x1 conv: input[batch=0, in_c, orig_out_h, orig_out_w] * weight[out_c, in_c, 0, 0] + bias[out_c]
        in_c = out_c  # 1x1 conv channels directly connected
        
        # Compute linear index for input: [1,512,128,128] -> 1*512*128*128 + in_c*128*128 + orig_out_h*128 + orig_out_w
        input_idx = (in_c * 128 * 128) + (orig_out_h * 128) + orig_out_w
        
        # Compute linear indices for weight and bias
        weight_idx = out_c
        bias_idx = out_c
        
        # Load values
        input_val = tl.load(input_ptr + input_idx)
        weight_val = tl.load(weight_ptr + weight_idx)
        bias_val = tl.load(bias_ptr + bias_idx)
        
        # Perform 1x1 convolution
        result = input_val * weight_val + bias_val
        
        # Store result for this output pixel and channel
        output_idx = pid
        tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_conv_interpolate(input, weight, bias):
    # For this specific pattern: [1,512,128,128] with [150,512,1,1] weight -> [1,150,512,512]
    
    # Total output elements: 1 * 150 * 512 * 512
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels = weight.shape[0]
    output_height, output_width = 512, 512
    
    # Create output tensor
    output = torch.empty((1, output_channels, output_height, output_width), 
                        dtype=input.dtype, device=input.device)
    
    # Flatten tensors for kernel access
    input_flat = input.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Launch kernel with correct grid size
    grid_size = 1 * output_channels * output_height * output_width
    
    fused_conv_interpolate_kernel[(grid_size,)](
        input_flat, weight_flat, bias_flat, output_flat, 4
    )
    
    # Reshape flat output back to proper dimensions
    output = output.reshape((1, output_channels, output_height, output_width))
    return output

def replacement_func():
    return fused_conv_interpolate