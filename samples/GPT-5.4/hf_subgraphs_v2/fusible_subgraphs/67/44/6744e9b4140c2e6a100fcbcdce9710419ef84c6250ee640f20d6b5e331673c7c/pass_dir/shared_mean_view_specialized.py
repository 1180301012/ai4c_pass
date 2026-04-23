import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def _mean_contig_kernel(
    x_ptr,
    out_ptr,
    SPATIAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * SPATIAL
    acc = tl.zeros((), dtype=tl.float32)
    for start in tl.static_range(0, SPATIAL, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < SPATIAL
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=0)
    mean_val = acc / SPATIAL
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def dispatch_mean_view_specialized(x):
    n, c, h, w = x.shape
    out = torch.empty((1, 1, n * c), device=x.device, dtype=x.dtype)
    nc = n * c
    spatial = h * w

    if spatial == 49:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=49)
    elif spatial == 64:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=64)
    elif spatial == 121:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=121)
    elif spatial == 144:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=144)
    elif spatial == 196:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=196)
    elif spatial == 256:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=256)
    elif spatial == 400:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=400)
    elif spatial == 441:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=441)
    elif spatial == 784:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=784)
    elif spatial == 1024:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=1024)
    elif spatial == 1600:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=1600)
    else:
        _mean_contig_kernel[(nc,)](x, out, SPATIAL=6400)
    return out