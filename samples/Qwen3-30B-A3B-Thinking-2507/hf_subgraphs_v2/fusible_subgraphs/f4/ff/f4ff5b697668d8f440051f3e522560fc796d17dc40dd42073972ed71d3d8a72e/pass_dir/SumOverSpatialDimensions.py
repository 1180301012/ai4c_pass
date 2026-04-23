import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_0):
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(-1, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim = 2, keepdim = True)
    return tmp_6

def replacement_args(tmp_3, in_0):
    tmp_4 = tmp_3.mul(in_0)
    return (tmp_4,)

@triton.jit
def sum_spatial_kernel(
    x_ptr,
    out_ptr,
    batch,
    n_chans,
):
    idx = tl.program_id(0)
    b = idx // n_chans
    c = idx % n_chans

    x_offset = b * n_chans * 64 * 64 + c * 64 * 64
    total = 0.0
    for i in range(64 * 64):
        total += tl.load(x_ptr + x_offset + i)
    out_idx = b * n_chans + c
    tl.store(out_ptr + out_idx, total)

@torch.fx.wrap
def sum_spatial(x):
    batch = x.shape[0]
    n_chans = x.shape[1]
    out = torch.empty((batch, n_chans, 1), dtype=x.dtype, device=x.device)
    num_blocks = batch * n_chans
    sum_spatial_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        batch=batch,
        n_chans=n_chans,
    )
    return out

def replacement_func():
    return sum_spatial