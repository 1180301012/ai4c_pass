import torch
import triton
import triton.language as tl


@triton.jit
def _flatten_h1d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    d_size,
    stride_h,
    stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    h_idx = offsets // d_size
    d_idx = offsets % d_size
    in_offsets = h_idx * stride_h + d_idx * stride_d

    values = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


@torch.fx.wrap
def singleton_attention_flatten(x):
    h_size = x.shape[0]
    d_size = x.shape[2]
    n_elements = h_size * d_size
    out = torch.empty((1, 1, n_elements), device=x.device, dtype=x.dtype)

    _flatten_h1d_kernel[(triton.cdiv(n_elements, 1024),)](
        x,
        out,
        n_elements,
        d_size,
        x.stride(0),
        x.stride(2),
        BLOCK_SIZE=1024,
        num_warps=1,
        num_stages=1,
    )
    return out