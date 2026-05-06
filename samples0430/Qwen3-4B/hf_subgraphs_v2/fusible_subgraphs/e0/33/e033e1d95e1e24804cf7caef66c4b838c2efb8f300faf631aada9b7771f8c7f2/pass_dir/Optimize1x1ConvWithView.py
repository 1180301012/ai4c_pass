import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(2, 256, -1)
    return (tmp_3,)

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def optimized_conv_kernel(
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    n_batches: tl.int32,
    n_out_channels: tl.int32,
    n_in_channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Minimal kernel implementation (to be optimized)
    pass

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_3):
    n_batches = in_3.shape[0]
    n_out_channels = in_0.shape[0]
    n_in_channels = in_1.shape[1]
    height = in_3.shape[2]
    width = in_3.shape[3]
    
    out = torch.empty((n_batches, n_out_channels, height * width),
                      dtype=in_3.dtype)
    
    optimized_conv_kernel[(n_batches * n_out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE,](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        n_batches=n_batches,
        n_out_channels=n_out_channels,
        n_in_channels=n_in_channels,
        height=height,
        width=width,
        BLOCK_SIZE=128,
    )
    
    return out

def replacement_func():
    return kernel_wrapper