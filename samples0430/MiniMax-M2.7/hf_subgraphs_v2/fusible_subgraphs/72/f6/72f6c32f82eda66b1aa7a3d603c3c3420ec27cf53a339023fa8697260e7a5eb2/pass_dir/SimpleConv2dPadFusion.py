import torch
import triton
import triton.language as tl


@triton.jit
def simple_conv_pad_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_out_channel_stride, weight_in_channel_stride,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple fused conv2d + pad kernel"""
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over output channels (simplified - just compute for one output channel per thread)
    out_ch = pid % out_channels
    
    for c in range(0, in_channels, BLOCK_SIZE):
        # Load input (with padding)
        pad_h = h_idx + 2
        pad_w = w_idx + 2
        actual_h = pad_h if pad_h < height + 2 else 0
        actual_w = pad_w if pad_w < width + 2 else 0
        orig_h = actual_h - 2 if actual_h >= 2 else 0
        orig_w = actual_w - 2 if actual_w >= 2 else 0
        
        for ci in range(tl.minimum(BLOCK_SIZE, in_channels - c)):
            in_offset = (batch_idx * input_batch_stride + 
                        (c + ci) * input_channel_stride + 
                        orig_h * input_h_stride + 
                        orig_w * input_w_stride)
            in_val = tl.load(input_ptr + in_offset)
            
            weight_offset = (out_ch * weight_out_channel_stride + 
                            (c + ci) * weight_in_channel_stride)
            w_val = tl.load(weight_ptr + weight_offset)
            
            acc[ci] += in_val * w_val
    
    result = tl.sum(acc, axis=0)
    out_offset = (batch_idx * out_channels * height * width + 
                  out_ch * height * width + 
                  h_idx * width + w_idx)
    tl.store(output_ptr + out_offset, result.to(tl.float16))


@torch.fx.wrap
def simple_conv_pad_wrapper(in_0, in_1):
    """Fused conv2d + pad"""
    batch, in_channels, height, width = in_1.shape
    out_channels = in_0.shape[0]
    
    input_batch_stride = in_1.stride(0)
    input_channel_stride = in_1.stride(1)
    input_h_stride = in_1.stride(2)
    input_w_stride = in_1.stride(3)
    
    weight_out_channel_stride = in_0.stride(0)
    weight_in_channel_stride = in_0.stride(1)
    
    out = torch.empty((batch, out_channels, height, width), 
                      dtype=in_1.dtype, device=in_1.device)
    
    grid = (batch * height * width,)
    BLOCK_SIZE = 64
    
    simple_conv_pad_kernel[grid](
        in_1, in_0, out,
        input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
        weight_out_channel_stride, weight_in_channel_stride,
        batch, in_channels, out_channels, height, width,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """Match conv2d + pad pattern"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    return tmp_2


def replacement_args(in_0, in_1):
    """Extract arguments"""
    return (in_0, in_1)


def replacement_func():
    """Return the replacement function"""
    return simple_conv_pad_wrapper