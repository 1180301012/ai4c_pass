import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ],
    key=[],
)
@triton.jit
def _softmax_4096_kernel(
    x_ptr,
    out_ptr,
    stride_xb,
    stride_ob,
    B,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    x_row_ptr = x_ptr + pid * stride_xb
    out_row_ptr = out_ptr + pid * stride_ob

    row_max = tl.full([], -float("inf"), tl.float32)
    for start in range(0, 4096, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row_ptr + offs).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = tl.zeros([], dtype=tl.float32)
    for start in range(0, 4096, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row_ptr + offs).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    for start in range(0, 4096, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row_ptr + offs).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(out_row_ptr + offs, y)


@torch.fx.wrap
def spatial_softmax_4096_from_nchw1(x):
    batch = x.shape[0]
    out = torch.empty((batch, 1, 4096), device=x.device, dtype=x.dtype)
    _softmax_4096_kernel[(batch,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        batch,
    )
    return out