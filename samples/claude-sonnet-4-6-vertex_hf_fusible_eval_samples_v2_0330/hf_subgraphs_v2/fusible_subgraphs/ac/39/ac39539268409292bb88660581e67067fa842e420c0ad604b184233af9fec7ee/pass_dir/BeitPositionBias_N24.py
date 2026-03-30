import torch
import triton
import triton.language as tl


@triton.jit
def _cat24_kernel(b_ptr, a_ptr, out_ptr, b_numel, total_numel, BLOCK: tl.constexpr):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offsets < total_numel
    from_b  = offsets < b_numel
    a_off   = offsets - b_numel
    b_data  = tl.load(b_ptr + offsets, mask=from_b  & mask, other=0.0)
    a_data  = tl.load(a_ptr + a_off,   mask=~from_b & mask, other=0.0)
    result  = tl.where(from_b, b_data, a_data)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def beit_cat_N24(in_0, in_1):
    """Triton cat([in_1, in_0]) for N=24 graphs."""
    device = in_0.device
    dtype  = in_0.dtype
    BLOCK  = 1024
    b_numel     = in_1.numel()
    total_numel = b_numel + in_0.numel()
    cat_flat    = torch.empty(total_numel, dtype=dtype, device=device)
    _cat24_kernel[((total_numel + BLOCK - 1) // BLOCK,)](
        in_1, in_0, cat_flat, b_numel, total_numel, BLOCK=BLOCK)
    return cat_flat.view(in_1.shape[0] + in_0.shape[0], in_1.shape[1])


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return beit_cat_N24