import torch
import triton
import triton.language as tl

@triton.jit
def expand_kernel(
    in_ptr,
    out_ptr,
    batch: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    target_ch: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (channels * target_ch * height * width)
    c = (pid // (target_ch * height * width)) % channels
    tc = (pid // (height * width)) % target_ch
    h = (pid // width) % height
    w = pid % width

    idx_in = b * channels * height * width + c * height * width + h * width + w
    x = tl.load(in_ptr + idx_in)

    idx_out = b * channels * target_ch * height * width + c * target_ch * height * width + tc * height * width + h * width + w
    tl.store(out_ptr + idx_out, x)

@torch.fx.wrap
def expand_wrapper(in_0):
    batch, channels, height, width = in_0.shape
    out_tensor = torch.empty((batch, channels, 64, height, width), dtype=in_0.dtype, device=in_0.device)
    num_blocks = batch * channels * height * width
    expand_kernel[(num_blocks,)](
        in_0,
        out_tensor,
        batch,
        channels,
        height,
        width,
        64,
        8
    )
    return out_tensor

def pattern(in_0):
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return expand_wrapper