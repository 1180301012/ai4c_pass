import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match: conv2d followed by channel slice [:, :N, :, :]
    Returns both the sliced output and full conv output
    """
    # Conv2d operation: in_1 @ in_0 (weight)
    # Conv2d args: input, weight, bias, stride, padding, dilation, groups
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    
    # Channel slicing to extract first N channels
    tmp_2 = tmp_1[slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None)]
    
    # Return both sliced and full output
    return (tmp_2, tmp_1)


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the optimized implementation.
    We need the weight and input tensors.
    """
    return (in_0, in_1)


# Optimized Triton kernel for conv2d with channel slicing
@triton.jit
def conv_slice_kernel(
    input_ptr, weight_ptr, output_sliced_ptr, output_full_ptr,
    batch_size, in_channels, out_channels, out_channels_needed,
    in_height, in_width, out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w,
    # Block size for tiling
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused conv2d + channel slice kernel.
    Computes both the sliced output (first N channels) and full output.
    Uses BLOCK_SIZE to parallelize over output elements.
    """
    # Get program ID and calculate output element range
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Total output elements (batch * out_channels * out_height * out_width)
    total_elements = batch_size * out_channels * out_height * out_width
    elements_per_program = (total_elements + num_programs - 1) // num_programs
    
    start_elem = pid * elements_per_program
    end_elem = min(start_elem + elements_per_program, total_elements)
    
    # Process elements in this program's range
    for elem_idx in range(start_elem, end_elem):
        # Decode flat index to (batch, out_c, out_h, out_w)
        tmp = elem_idx
        out_w = tmp % out_width
        tmp = tmp // out_width
        out_h = tmp % out_height
        tmp = tmp // out_height
        out_c = tmp % out_channels
        batch_idx = tmp // out_channels
        
        # Compute convolution: sum over input channels and kernel
        sum_val = 0.0
        
        # Loop over input channels and kernel dimensions
        for in_c in range(in_channels):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate input position
                    in_h = out_h * stride_h + kh - padding_h
                    in_w = out_w * stride_w + kw - padding_w
                    
                    # Check bounds
                    if in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                        # Flattened index for input: (batch, in_c, in_h, in_w)
                        in_idx = ((batch_idx * in_channels + in_c) * in_height + in_h) * in_width + in_w
                        inp_val = tl.load(input_ptr + in_idx)
                        
                        # Flattened index for weight: (out_c, in_c, kh, kw)
                        w_idx = (((out_c * in_channels + in_c) * kernel_h + kh) * kernel_w + kw)
                        w_val = tl.load(weight_ptr + w_idx)
                        
                        sum_val += inp_val * w_val
        
        # Store full output
        full_idx = ((batch_idx * out_channels + out_c) * out_height + out_h) * out_width + out_w
        tl.store(output_full_ptr + full_idx, sum_val)
        
        # Store sliced output if this channel is in the sliced range
        if out_c < out_channels_needed:
            sliced_idx = ((batch_idx * out_channels_needed + out_c) * out_height + out_h) * out_width + out_w
            tl.store(output_sliced_ptr + sliced_idx, sum_val)


@torch.fx.wrap
def conv_slice_kernel_wrapper(in_0, in_1):
    """
    Wrapper for the fused conv + slice kernel.
    in_0: weight tensor [out_channels, in_channels, kH, kW]
    in_1: input tensor [batch, in_channels, H, W]
    
    Returns: (sliced_output, full_output)
    """
    # Get tensor dimensions
    out_channels, in_channels, kernel_h, kernel_w = in_0.shape
    batch_size, _, in_height, in_width = in_1.shape
    
    # Convolution parameters
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 0, 0
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Number of channels to slice (1024 from the pattern)
    out_channels_needed = min(1024, out_channels)
    
    # Allocate output tensors on the same device as input
    device = in_1.device
    output_sliced = torch.empty((batch_size, out_channels_needed, out_height, out_width), 
                                dtype=torch.float32, device=device)
    output_full = torch.empty((batch_size, out_channels, out_height, out_width), 
                               dtype=torch.float32, device=device)
    
    # Flatten input and weight for kernel
    input_flat = in_1.contiguous().view(-1)
    weight_flat = in_0.contiguous().view(-1)
    
    # Calculate grid size
    total_elements = batch_size * out_channels * out_height * out_width
    BLOCK_SIZE = 128
    num_programs = min(4096, (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    conv_slice_kernel[(num_programs)](
        input_ptr=input_flat,
        weight_ptr=weight_flat,
        output_sliced_ptr=output_sliced.view(-1),
        output_full_ptr=output_full.view(-1),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        out_channels_needed=out_channels_needed,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (output_sliced, output_full)


def replacement_func():
    return conv_slice_kernel_wrapper