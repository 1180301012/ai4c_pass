import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None)]
    return tmp_3, tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def embed_kernel(
    idx_ptr,
    wt_ptr, 
    out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    v = tl.load(idx_ptr + pid)
    base_w = v * D
    base_o = pid * D
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(wt_ptr + base_w + offs, mask=mask, other=0.0)
    tl.store(out_ptr + base_o + offs, x, mask=mask)


@torch.fx.wrap 
def triton_embed(idx, wt):
    b, s = idx.shape
    d = wt.shape[1]
    n = b * s
    out = torch.empty(b, s, d, dtype=wt.dtype, device=wt.device)
    embed_kernel[(n,)](idx.view(-1), wt, out.view(-1), d, BLOCK=128, num_warps=2)
    return out


def replacement(idx, wt):
    out = triton_embed(idx, wt)
    sl = out[slice(None, None, None), slice(1, None, None)]
    return sl, out


def replacement_func():
    return replacement