import torch
import triton
import triton.language as tl

# Pattern matching the conv2d + slice operation (stride 2,2, slice 2048)
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    return tmp_2, conv2d

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def triton_conv2d_slice_kernel(
    weight_ptr, input_ptr, output_sliced_ptr, output_full_ptr,
    weight_out_channel_stride, weight_in_channel_stride,
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    batch_size, in_channels, out_channels,
    output_height, output_width,
    num_needed, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_elements_per_batch = output_height * output_width
    b = pid // n_elements_per_batch
    rest = pid % n_elements_per_batch
    h = rest // output_width
    w = rest % output_width
    
    input_h = h * stride_h
    input_w = w * stride_w
    output_offset = (b * output_batch_stride + h * output_h_stride + w * output_w_stride)
    input_base_offset = (b * input_batch_stride + input_h * input_h_stride + input_w * input_w_stride)
    
    for oc in range(out_channels):
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ic in range(in_channels):
            weight_offset = (oc * weight_out_channel_stride + ic * weight_in_channel_stride)
            weight_val = tl.load(weight_ptr + weight_offset)
            input_offset = input_base_offset + ic * input_channel_stride
            input_val = tl.load(input_ptr + input_offset)
            acc += input_val * weight_val
        
        full_offset = output_offset + oc * output_channel_stride
        tl.store(output_full_ptr + full_offset, acc)
        
        if oc < num_needed:
            sliced_offset = output_offset + oc * output_channel_stride
            tl.store(output_sliced_ptr + sliced_offset, acc)

@torch.fx.wrap
def triton_conv2d_slice(weight, input_tensor):
    batch_size, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    stride_h, stride_w = 2, 2
    padding_h, padding_w = 0, 0
    dilation_h, dilation_w = 1, 1
    
    out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    num_needed = 2048
    
    output_sliced = torch.empty((batch_size, num_needed, out_h, out_w), dtype=torch.float32, device=input_tensor.device)
    output_full = torch.empty((batch_size, out_channels, out_h, out_w), dtype=torch.float32, device=input_tensor.device)
    
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
    
    n_elements = batch_size * out_h * out_w
    BLOCK_SIZE = 1024
    
    triton_conv2d_slice_kernel[(n_elements,)](
        weight, input_tensor, output_sliced, output_full,
        weight_out_channel_stride, weight_in_channel_stride,
        input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
        output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
        batch_size, in_channels, out_channels,
        out_h, out_w, num_needed, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    dtype = input_tensor.dtype
    return output_sliced.to(dtype), output_full.to(dtype)

def replacement_func():
    return triton_conv2d_slice