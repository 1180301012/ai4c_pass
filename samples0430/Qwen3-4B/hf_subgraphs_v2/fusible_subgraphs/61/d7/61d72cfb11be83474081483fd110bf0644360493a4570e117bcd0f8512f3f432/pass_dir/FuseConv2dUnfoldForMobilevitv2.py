import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_unfold_kernel(in_1_ptr, in_0_ptr, out_ptr, batch_size, channels, in_height, in_width, out_height, out_width, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    if row_idx >= out_height or col_idx >= out_width:
        return
    
    start_row = row_idx * 2
    start_col = col_idx * 2
    
    # Compute 1x1 convolution (channel-wise) for each window position
    for channel_id in tl.arange(0, channels):
        # Load 2x2 window from input
        window_data = tl.load(in_1_ptr + (start_row, start_col, channel_id, tl.arange(0, 4)), mask=tl.full([4], 1, dtype=tl.int1))
        
        # Apply 1x1 convolution weights (in_0 has shape [128, 256, 1, 1])
        weighted = tl.zeros(4, dtype=in_1.dtype)
        for i in range(4):
            weighted[i] = tl.dot(window_data[i], in_0_ptr[channel_id])
        
        # Store output in final tensor
        tl.store(out_ptr + (row_idx * out_width + col_idx) * 4 + channel_id, weighted, mask=tl.full([4], 1, dtype=tl.int1))

def kernel_wrapper(in_1, in_0):
    batch_size = in_1.shape[0]
    channels = in_0.shape[0]
    in_height = in_1.shape[2]
    in_width = in_1.shape[3]
    out_height = in_height // 2
    out_width = in_width // 2
    
    output = torch.empty((batch_size, channels, 4, out_height * out_width),
                        device=in_1.device,
                        dtype=in_1.dtype)
    
    fused_conv_unfold_kernel[ (out_height, out_width) ](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=output,
        batch_size=batch_size,
        channels=channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE=128
    )
    return (output,)

def replacement_func():
    return kernel_wrapper