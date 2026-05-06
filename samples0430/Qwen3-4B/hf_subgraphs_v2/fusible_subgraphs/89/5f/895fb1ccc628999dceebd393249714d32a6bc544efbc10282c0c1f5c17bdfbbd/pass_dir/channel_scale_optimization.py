import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    scale = torch.sigmoid(in_0)
    scale = scale.view(1, 512, 1, 1)
    scaled = in_1 * scale
    add = in_1 + scaled
    relu_out = torch.relu_(add)
    out = relu_out * 0.9
    return (out,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def channel_scale_kernel(
    in_0_ptr: tl.ptr,
    in_1_ptr: tl.ptr,
    out_ptr: tl.ptr,
    N_channels: tl.int,
    N_height: tl.int,
    N_width: tl.int,
    BLOCK_SIZE: tl.int,
):
    c = tl.program_id(0)
    if c >= N_channels:
        return
    
    in_0_val = tl.load(in_0_ptr + c)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-in_0_val))
    
    h = tl.arange(0, N_height)
    w = tl.arange(0, N_width)
    
    for h_i in h:
        for w_i in w:
            in_1_val = tl.load(in_1_ptr + c * (N_height * N_width) + h_i * N_width + w_i)
            scaled_val = in_1_val * (1.0 + sigmoid_val)
            relu_val = tl.max(scaled_val, 0.0)
            out_val = relu_val * 0.9
            tl.store(out_ptr + c * (N_height * N_width) + h_i * N_width + w_i, out_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    N_channels = in_0.shape[1]
    N_height = in_1.shape[2]
    N_width = in_1.shape[3]
    out = torch.empty_like(in_1)
    
    channel_scale_kernel[(1, 1)](\n        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        N_channels=N_channels,
        N_height=N_height,
        N_width=N_width,
        BLOCK_SIZE=1,
    )
    return out

def replacement_func():
    return kernel_wrapper