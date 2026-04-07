import torch
import triton
import triton.language as tl

def pattern(in_0):
    # ReLU activation
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    
    # Three identical max_pool2d operations
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    
    # Concatenate all four results
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_max_pool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * out_height * out_width
    if pid >= total_elements:
        return
    
    # Calculate indices
    batch = pid // (out_channels * out_height * out_width)
    channel_remainder = pid % (out_channels * out_height * out_width)
    out_channel = channel_remainder // (out_height * out_width)
    spatial_remainder = channel_remainder % (out_height * out_width)
    out_h = spatial_remainder // out_width
    out_w = spatial_remainder % out_width
    
    # Calculate which of the 4 outputs we're processing (0=original, 1-3=pooled)
    output_group = out_channel // in_channels
    in_channel = out_channel % in_channels
    
    # Process different output groups
    if output_group == 0:
        # Original output - just copy with ReLU applied
        in_h_base = out_h * stride
        in_w_base = out_w * stride
        if (in_h_base < in_height and in_w_base < in_width):
            input_val = tl.load(input_ptr + batch * in_channels * in_height * in_width + 
                              in_channel * in_height * in_width + 
                              in_h_base * in_width + in_w_base)
            output_val = max(0.0, input_val)
            tl.store(output_ptr + pid, output_val)
        else:
            tl.store(output_ptr + pid, 0.0)
    else:
        # Max pooled outputs
        in_h_base = out_h * stride - padding
        in_w_base = out_w * stride - padding
        
        # Find max in kernel window
        max_val = -float('inf')
        found_valid = False
        
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                in_h = in_h_base + kh * dilation
                in_w = in_w_base + kw * dilation
                
                if (0 <= in_h < in_height and 0 <= in_w < in_width):
                    input_val = tl.load(input_ptr + batch * in_channels * in_height * in_width + 
                                      in_channel * in_height * in_width + 
                                      in_h * in_width + in_w)
                    valid_val = max(0.0, input_val)  # Apply ReLU first
                    if not found_valid or valid_val > max_val:
                        max_val = valid_val
                        found_valid = True
        
        if found_valid:
            tl.store(output_ptr + pid, max_val)
        else:
            tl.store(output_ptr + pid, 0.0)

@torch.fx.wrap
def fused_relu_concat_max_pool2d(input_tensor):
    # Get input shape
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    
    # Calculate output dimensions after max_pool2d
    out_height = in_height
    out_width = in_width
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    
    # Final output has 4x channels (original + 3 pooled versions)
    out_channels = in_channels * 4
    
    # Create output tensor
    output_tensor = torch.empty((batch_size, out_channels, out_height, out_width), 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    total_elements = batch_size * out_channels * out_height * out_width
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_max_pool_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        out_height=out_height,
        out_width=out_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    return fused_relu_concat_max_pool2d