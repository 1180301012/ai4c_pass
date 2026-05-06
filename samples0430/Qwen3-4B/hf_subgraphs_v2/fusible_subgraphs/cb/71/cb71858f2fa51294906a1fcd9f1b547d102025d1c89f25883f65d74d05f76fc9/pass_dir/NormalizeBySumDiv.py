import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def normalize_kernel(
    in_3_ptr,
    out_ptr,
    n_batch,
    n_channels,
    n_height,
    n_width,
    BLOCK_SIZE: tl.constexpr,
):
    pass

@torch.fx.wrap
def normalize(in_3):
    batch, channels, height, width = in_3.shape
    n_batch = batch
    n_channels = channels
    n_height = height
    n_width = width
    out = torch.empty_like(in_3)
    normalize_kernel[(1,)](
        in_3_ptr=in_3,
        out_ptr=out,
        n_batch=n_batch,
        n_channels=n_channels,
        n_height=n_height,
        n_width=n_width,
        BLOCK_SIZE=128,
    )
    return out

def replacement_func():
    return normalize