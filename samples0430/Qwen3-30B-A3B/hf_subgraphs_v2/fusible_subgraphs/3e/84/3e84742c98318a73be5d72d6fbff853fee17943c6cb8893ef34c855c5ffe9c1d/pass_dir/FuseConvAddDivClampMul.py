import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_and_postprocess_kernel(
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    in_2_ptr,
    out_ptr,
    batch_size,
    channels_in,
    channels_out,
    spatial_h,
    spatial_w,
    BLOCK_SIZE_CH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr
):
    ch_id = tl.program_id(0)
    h_id = tl.program_id(1)
    w_id = tl.program_id(2)
    
    start_ch = ch_id * BLOCK_SIZE_CH
    end_ch = tl.minimum(start_ch + BLOCK_SIZE_CH, channels_out)
    
    start_h = h_id * BLOCK_SIZE_SPATIAL
    end_h = tl.minimum(start_h + BLOCK_SIZE_SPATIAL, spatial_h)
    start_w = w_id * BLOCK_SIZE_SPATIAL
    end_w = tl.minimum(start_w + BLOCK_SIZE_SPATIAL, spatial_w)
    
    weight_ptr = in_1_ptr + start_ch * channels_in
    weight = tl.load(weight_ptr + tl.arange(0, channels_in), 
                    mask=tl.arange(0, channels_in) < channels_in, 
                    other=0.0)
    bias = tl.load(in_0_ptr + start_ch, 
                  mask=start_ch < channels_out, 
                  other=0.0)
    
    for b in range(batch_size):
        input_ptr = in_3_ptr + b * channels_in
        input = tl.load(input_ptr + tl.arange(0, channels_in), 
                       mask=tl.arange(0, channels_in) < channels_in, 
                       other=0.0)
        
        conv = tl.dot(input, weight)
        conv += bias
        
        conv = (conv + 1.0) / 2.0
        conv = tl.minimum(tl.maximum(conv, 0.0), 1.0)
        
        for h in range(start_h, end_h):
            for w in range(start_w, end_w):
                out_offset = b * channels_out * spatial_h * spatial_w + start_ch * spatial_h * spatial_w + h * spatial_w + w
                in_2_val = tl.load(in_2_ptr + out_offset, 
                                  mask=(h < spatial_h) & (w < spatial_w), 
                                  other=0.0)
                tl.store(out_ptr + out_offset, conv * in_2_val, 
                        mask=(h < spatial_h) & (w < spatial_w))

def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch_size, channels_in, _, _ = in_3.shape
    channels_out = in_1.shape[0]
    spatial_h, spatial_w = in_2.shape[2], in_2.shape[3]
    
    output = torch.empty_like(in_2)
    
    num_ch_blocks = (channels_out + 64 - 1) // 64
    num_h_blocks = (spatial_h + 16 - 1) // 16
    num_w_blocks = (spatial_w + 16 - 1) // 16
    
    fused_conv_and_postprocess_kernel[(num_ch_blocks, num_h_blocks, num_w_blocks)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=output,
        batch_size=batch_size,
        channels_in=channels_in,
        channels_out=channels_out,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        BLOCK_SIZE_CH=64,
        BLOCK_SIZE_SPATIAL=16
    )
    
    return output

@torch.fx.wrap
def fused_conv_and_postprocess(in_0, in_1, in_2, in_3):
    return kernel_wrapper(in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_conv_and_postprocess