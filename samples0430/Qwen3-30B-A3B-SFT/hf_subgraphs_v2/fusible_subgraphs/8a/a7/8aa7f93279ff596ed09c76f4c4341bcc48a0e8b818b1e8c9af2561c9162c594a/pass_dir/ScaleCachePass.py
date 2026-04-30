"""
Two-pass cache approach:
  Pass 1 (ScaleCachePass): matches scale, runs fused Triton kernel (scale+transpose),
    stores trans in _trans_cache, returns scale output only.
  Pass 2 (CacheTransposePass): matches transpose, retrieves cached trans output.

Avoids the tuple-return limitation that crashes multi-output FuseScaleAndTranspose.
"""
import torch
import triton
import triton.language as tl

_trans_cache: dict = {}


def pattern(in_1):
    return in_1 * 0.1767766952966369


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def scale_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Compute scale in fp32, cast back to original dtype
    y = (x.to(tl.float32) * 0.1767766952966369).to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    B, S, D, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    SD = S * D
    DS = D * S

    b = offsets // DS
    rem = offsets % DS
    d = rem // S
    s = rem % S

    in_idx = b * SD + s * D + d
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_scale_and_cache(in_1):
    n = in_1.numel()
    B = in_1.shape[0]
    S = in_1.shape[1]
    D = in_1.shape[2]
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    out_scale = torch.empty_like(in_1)
    scale_kernel[(num_blocks,)](in_1, out_scale, n, BLOCK_SIZE=BLOCK_SIZE)

    # Compute transpose and cache
    out_trans = torch.empty(B, D, S, device=in_1.device, dtype=in_1.dtype)
    transpose_kernel[(num_blocks,)](in_1, out_trans, B, S, D, n, BLOCK_SIZE=BLOCK_SIZE)
    _trans_cache[in_1.data_ptr()] = out_trans

    return out_scale


def replacement_func():
    return fused_scale_and_cache