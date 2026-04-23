import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror model.py exactly for the matched subgraph.
def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_4, in_5):
    return (in_4, in_5)


@triton.jit
def fused_add_mean_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    base = pid * S

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean_val = tl.sum(x + y, axis=0) / S
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def fused_add_mean(in_4, in_5):
    n = in_4.shape[0]
    c = in_4.shape[1]
    s = in_4.shape[2] * in_4.shape[3]
    nc = n * c

    out = torch.empty((n, c), device=in_4.device, dtype=in_4.dtype)
    if s <= 64:
        block_s = 64
        num_warps = 1
    elif s <= 128:
        block_s = 128
        num_warps = 2
    else:
        block_s = 256
        num_warps = 4

    fused_add_mean_kernel[(nc,)](
        in_5,
        in_4,
        out,
        s,
        BLOCK_S=block_s,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_add_mean