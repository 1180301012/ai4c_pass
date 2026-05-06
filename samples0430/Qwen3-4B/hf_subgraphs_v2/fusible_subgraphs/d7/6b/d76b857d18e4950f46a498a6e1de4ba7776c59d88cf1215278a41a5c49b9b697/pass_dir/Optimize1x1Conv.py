import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def optimized_conv_kernel(
    input_ptr,  
    weight_ptr,  
    bias_ptr,  
    output_ptr,  
    batch_size: tl.int32,  
    channels_in: tl.int32,  
    channels_out: tl.int32,  
    h: tl.int32,  
    w: tl.int32,  
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block index
    pid = tl.program_id(0)
    # Process channels in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the current channel block
    channel_block = pid * BLOCK_SIZE
    # For each spatial location
    for i in range(h):
        for j in range(w):
            # For each channel
            for channel in tl.arange(0, channels_out):
                # Accumulate channel-wise dot product
                total = 0.0
                for c in tl.arange(0, channels_in):
                    input_val = tl.load(input_ptr + (batch_size * channels_in + c + i * w + j))
                    weight_val = tl.load(weight_ptr + (channel * channels_in + c))
                    total += input_val * weight_val
                # Add bias
                bias_val = tl.load(bias_ptr + channel)
                tl.store(output_ptr + (i * w + j + channel), total + bias_val)

@torch.fx.wrap
def optimized_conv(in_2, in_1, in_0):
    batch_size = in_2.shape[0]
    channels_in = in_2.shape[1]
    channels_out = in_1.shape[0]
    h = in_2.shape[2]
    w = in_2.shape[3]
    output = torch.empty_like(in_2)
    optimized_conv_kernel[tl.cshape((batch_size, channels_out, h, w))](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        channels_in=channels_in,
        channels_out=channels_out,
        h=h,
        w=w,
        BLOCK_SIZE=128,
    )
    return output

def replacement_func():
    return optimized_conv