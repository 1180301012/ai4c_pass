import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Match depthwise conv2d followed by mean over spatial dimensions."""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_0 = None
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function."""
    return (in_0, in_1)


# Depthwise convolution kernel
@triton.jit
def depthwise_conv_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_ch, out_ch, height, width,
    kernel_h, kernel_w, stride, padding
):
    """Triton kernel for depthwise convolution."""
    pid = tl.program_id(0)
    
    # Calculate output indices
    out_c = pid % out_ch
    remaining = pid // out_ch
    b = remaining % batch
    remaining = remaining // batch
    out_h = remaining % height
    out_w = remaining // height
    
    # Input position (considering padding)
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Compute convolution
    sum_val = 0.0
    in_c = out_c  # For depthwise: input channel = output channel
    
    weight_base = out_c * kernel_h * kernel_w
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            
            # Check bounds (padding) - split chained boolean operators
            cond1 = in_h >= 0
            cond2 = in_h < height
            cond3 = in_w >= 0
            cond4 = in_w < width
            
            # Use nested if to avoid chained boolean operators
            if cond1:
                if cond2:
                    if cond3:
                        if cond4:
                input_offset = b * in_ch * height * width + in_c * height * width + in_h * width + in_w
                val = tl.load(input_ptr + input_offset)
                
                weight_offset = weight_base + kh * kernel_w + kw
                w_val = tl.load(weight_ptr + weight_offset)
                
                sum_val = sum_val + val * w_val
    
    output_offset = b * out_ch * height * width + out_c * height * width + out_h * width + out_w
    tl.store(output_ptr + output_offset, sum_val)


# Mean kernel
@triton.jit
def mean_kernel_impl(
    input_ptr, output_ptr,
    batch, channels, height, width
):
    """Triton kernel for mean over spatial dimensions."""
    pid = tl.program_id(0)
    
    b = pid // channels
    c = pid % channels
    
    base_offset = b * channels * height * width + c * height * width
    
    sum_val = 0.0
    for h in range(height):
        for w in range(width):
            offset = base_offset + h * width + w
            val = tl.load(input_ptr + offset)
            sum_val = sum_val + val
    
    mean_val = sum_val / (height * width)
    
    output_offset = b * channels + c
    tl.store(output_ptr + output_offset, mean_val)


@torch.fx.wrap
def triton_depthwise_conv(input_tensor, weight):
    """Depthwise convolution using Triton kernel."""
    batch = input_tensor.shape[0]
    in_ch = input_tensor.shape[1]
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    
    out_ch = weight.shape[0]
    kernel_h = weight.shape[2]
    kernel_w = weight.shape[3]
    
    # Output size with same spatial dimensions (stride=1, padding=1)
    out_height = height
    out_width = width
    
    # Flatten weight for kernel
    weight_flat = weight.view(out_ch, -1).contiguous()
    
    # Allocate output
    output = torch.empty((batch, out_ch, out_height, out_width), 
                        dtype=torch.float32, device=input_tensor.device)
    
    # Grid: batch * out_ch * out_h * out_w
    num_programs = batch * out_ch * out_height * out_width
    
    depthwise_conv_kernel[(num_programs,)](
        input_ptr=input_tensor.contiguous(),
        weight_ptr=weight_flat,
        output_ptr=output,
        batch=batch, in_ch=in_ch, out_ch=out_ch,
        height=height, width=width,
        kernel_h=kernel_h, kernel_w=kernel_w,
        stride=1, padding=1
    )
    
    return output


@torch.fx.wrap
def triton_mean(input_tensor):
    """Mean over spatial dimensions using Triton kernel."""
    batch = input_tensor.shape[0]
    channels = input_tensor.shape[1]
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    
    output = torch.empty((batch, channels, 1, 1), dtype=torch.float32, device=input_tensor.device)
    
    num_programs = batch * channels
    
    mean_kernel_impl[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch=batch, channels=channels,
        height=height, width=width
    )
    
    return output


def replacement_func():
    """Return the replacement function."""
    
    def fused_op(weight, input_tensor):
        # Use Triton kernel for depthwise convolution
        conv_output = triton_depthwise_conv(input_tensor, weight)
        
        # Use Triton kernel for mean
        mean_output = triton_mean(conv_output)
        
        return conv_output, mean_output
    
    return fused_op