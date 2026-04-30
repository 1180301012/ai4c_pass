import torch
import triton
import triton.language as tl

# Pattern matching the conv2d + slice operation (stride 1,1, slice 64)
def pattern(in_0, in_1):
    # Conv2d with 1x1 kernel, stride 1
    # Parameters: input, weight, bias(None), stride, padding, dilation, groups
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    # Slice the first 64 channels
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 64, None), slice(None, None, None), slice(None, None, None))]
    return tmp_2, conv2d

# Extract arguments for the replacement function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def triton_conv2d_slice_kernel(
    # Pointers
    weight_ptr, input_ptr, output_sliced_ptr, output_full_ptr,
    # Weight strides
    weight_out_channel_stride, weight_in_channel_stride,
    # Input strides  
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    # Output strides
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    # Sizes
    batch_size, in_channels, out_channels,
    output_height, output_width,
    num_needed, 
    stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    # Get current program position
    pid = tl.program_id(0)
    
    # Compute b, h, w from flat pid
    n_elements_per_batch = output_height * output_width
    b = pid // n_elements_per_batch
    rest = pid % n_elements_per_batch
    h = rest // output_width
    w = rest % output_width
    
    # Compute input position (for 1x1 conv with stride 1, input position equals output position)
    input_h = h
    input_w = w
    
    # Compute output offset for this position
    output_offset = (b * output_batch_stride + 
                     h * output_h_stride + 
                     w * output_w_stride)
    
    # Compute input base offset
    input_base_offset = (b * input_batch_stride + 
                         input_h * input_h_stride + 
                         input_w * input_w_stride)
    
    # Compute sliced output (first num_needed channels) and full output in one pass
    for oc in range(out_channels):
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ic in range(in_channels):
            # Load weight: weight[oc, ic, 0, 0]
            weight_offset = (oc * weight_out_channel_stride + 
                            ic * weight_in_channel_stride)
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Load input: input[b, ic, input_h, input_w]
            input_offset = input_base_offset + ic * input_channel_stride
            input_val = tl.load(input_ptr + input_offset)
            
            acc += input_val * weight_val
        
        # Store to full output
        full_offset = output_offset + oc * output_channel_stride
        tl.store(output_full_ptr + full_offset, acc)
        
        # Store to sliced output if within range
        if oc < num_needed:
            sliced_offset = output_offset + oc * output_channel_stride
            tl.store(output_sliced_ptr + sliced_offset, acc)

@torch.fx.wrap
def triton_conv2d_slice(weight, input_tensor):
    """
    Optimized conv2d with slicing using Triton.
    """
    # Get tensor info
    batch_size, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Conv parameters (stride 1,1 for this pattern)
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 0, 0
    dilation_h, dilation_w = 1, 1
    
    # Compute output spatial dimensions
    out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Number of channels to slice (from pattern)
    num_needed = 64
    
    # Allocate output tensors - use float32 for accumulation
    output_sliced = torch.empty((batch_size, num_needed, out_h, out_w), 
                                dtype=torch.float32, device=input_tensor.device)
    output_full = torch.empty((batch_size, out_channels, out_h, out_w),
                              dtype=torch.float32, device=input_tensor.device)
    
    # Prepare strides (in elements, not bytes)
    weight_out_channel_stride = weight.stride(0)
    weight_in_channel_stride = weight.stride(1)
    
    input_batch_stride = input_tensor.stride(0)
    input_channel_stride = input_tensor.stride(1)
    input_h_stride = input_tensor.stride(2)
    input_w_stride = input_tensor.stride(3)
    
    output_batch_stride = output_full.stride(0)
    output_channel_stride = output_full.stride(1)
    output_h_stride = output_full.stride(2)
    output_w_stride = output_full.stride(3)
    
    # Grid configuration - one program per output spatial position per batch
    n_elements = batch_size * out_h * out_w
    BLOCK_SIZE = 1024
    
    # Launch kernel
    triton_conv2d_slice_kernel[(n_elements,)](
        weight, input_tensor, output_sliced, output_full,
        weight_out_channel_stride, weight_in_channel_stride,
        input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
        output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
        batch_size, in_channels, out_channels,
        out_h, out_w,
        num_needed,
        stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Convert to match input dtype (bfloat16 or float16)
    dtype = input_tensor.dtype
    output_sliced = output_sliced.to(dtype)
    output_full = output_full.to(dtype)
    
    return output_sliced, output_full

def replacement_func():
    return triton_conv2d_slice