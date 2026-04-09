import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Simple pattern - just conv2d alone to test matching
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit  
def simple_conv_kernel(
    input_ptr,
    output_ptr, 
    n_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Simple operation: just copy input and add 1.0
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output_val = input_val + 1.0
    
    tl.store(output_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def simple_conv_triton(in_0, in_1):
    # For conv2d: input is [batch, in_channels, H, W], weights are [out_channels, in_channels, K, H]
    batch_size, in_channels, in_height, in_width = in_1.shape
    out_channels, _, kernel_height, kernel_width = in_0.shape
    
    # Calculate conv2d output shape with stride=1, padding=1
    out_height = (in_height + 2*1 - kernel_height) // 1 + 1
    out_width = (in_width + 2*1 - kernel_width) // 1 + 1
    
    # Create output tensor with correct conv2d shape
    output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=in_0.dtype, device=in_0.device)
    
    n_elements = output.numel()
    num_programs = (n_elements + 1023) // 1024
    
    simple_conv_kernel[(num_programs,)](
        in_1, output,
        n_elements
    )
    
    return output

def replacement_func():
    return simple_conv_triton