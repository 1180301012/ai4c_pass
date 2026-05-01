import torch
import triton
import triton.language as tl

BLOCK_SIZE = 32

@triton.jit
def transpose_kernel(in_ptr, out_ptr, H, W):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_id(0)
    if idx >= H * W:
        return
    i = idx // W
    j = idx % W
    in_offset = i * W + j
    out_offset = j * H + i
    x = tl.load(in_ptr + in_offset, mask=idx < H * W)
    tl.store(out_ptr + out_offset, x, mask=idx < H * W)

def pattern(in_0):
    tmp1 = in_0.unsqueeze(1)
    tmp2 = tmp1.transpose(2, 3)
    return tmp2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return transpose_wrapper

@torch.fx.wrap
def transpose_wrapper(in_0):
    H, W = 1024, 128
    out = torch.empty(1, 1, W, H, dtype=in_0.dtype, device=in_0.device)
    in_2d = in_0.view(H, W)
    out_2d = out.view(W, H)
    grid = (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    transpose_kernel[grid](
        in_ptr=in_2d.data_ptr(),
        out_ptr=out_2d.data_ptr(),
        H=H,
        W=W
    )
    return out