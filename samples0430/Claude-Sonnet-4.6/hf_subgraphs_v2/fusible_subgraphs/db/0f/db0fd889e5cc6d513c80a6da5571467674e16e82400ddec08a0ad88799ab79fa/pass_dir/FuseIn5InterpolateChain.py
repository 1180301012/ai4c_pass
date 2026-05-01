import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=['n_batches', 'inner_n'],
)
@triton.jit
def _strided_copy_3d_kernel(
    src_ptr,
    dst_ptr,
    n_batches,
    inner_n,
    src_batch_stride,
    BLOCK: tl.constexpr,
):
    """
    Copy a 3D tensor [n_batches, *, *] where the inner elements
    (inner_n total) are contiguous within each batch, but batches
    may be separated by src_batch_stride elements.
    Output is written contiguously.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total = n_batches * inner_n
    mask = offsets < total

    batch_idx = offsets // inner_n
    inner_idx = offsets % inner_n

    src_off = batch_idx * src_batch_stride + inner_idx
    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def in5_chain_func(x):
    """
    Replace the chain:
      transpose(1,2) -> view(1,32,15,15) ->
      bicubic_interpolate(15,15) -> flatten(2) -> transpose(1,2)
    which is mathematically identity (same-size bicubic is identity).
    Input x: [1, 225, 32], possibly non-contiguous (sliced from in_5).
    Output:  [1, 225, 32], contiguous copy of x.
    """
    n_batches = 1
    inner_n = 225 * 32       # 7200 elements per batch
    src_batch_stride = x.stride(0)  # stride between batches (7552 for in_5)
    out = torch.empty(1, 225, 32, dtype=x.dtype, device=x.device)
    total = n_batches * inner_n
    grid = lambda meta: ((total + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _strided_copy_3d_kernel[grid](
        x, out,
        n_batches, inner_n, src_batch_stride,
    )
    return out


def pattern(x):
    t = x.transpose(1, 2)
    v = t.view(1, 32, 15, 15)
    i = torch.nn.functional.interpolate(v, size=(15, 15), mode='bicubic', align_corners=False)
    f = i.flatten(2)
    o = f.transpose(1, 2)
    return o


def replacement_args(x):
    return (x,)


def replacement_func():
    return in5_chain_func