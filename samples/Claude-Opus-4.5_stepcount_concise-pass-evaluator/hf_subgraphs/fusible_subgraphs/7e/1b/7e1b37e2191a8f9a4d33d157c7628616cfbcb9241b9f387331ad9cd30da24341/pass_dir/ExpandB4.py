import torch
import triton
import triton.language as tl

# Pattern for expand: batch=4, seq=512
def pattern(x):
    tmp = x[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    out = tmp.expand(4, 4, 4, 512, 128)
    return out

def replacement_args(x):
    return (x,)

@triton.jit
def expand_kernel_b4(
    in_ptr, out_ptr,
    n_elements,
    in_s0, in_s1, in_s2, in_s3,
    d0, d1, d2, d3, d4,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    
    idx = offs
    i4 = idx % d4
    idx = idx // d4
    i3 = idx % d3
    idx = idx // d3
    i2 = idx % d2
    idx = idx // d2
    i1 = idx % d1
    i0 = idx // d1
    
    in_idx = i0 * in_s0 + i1 * in_s1 + i3 * in_s2 + i4 * in_s3
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)

@torch.fx.wrap
def expand_b4(x):
    b, h, s, d = x.shape
    r = 4
    out = torch.empty((b, h, r, s, d), dtype=x.dtype, device=x.device)
    n = b * h * r * s * d
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    expand_kernel_b4[grid](
        x, out, n,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        b, h, r, s, d,
        BLOCK=BLOCK,
    )
    return out

def replacement_func():
    return expand_b4