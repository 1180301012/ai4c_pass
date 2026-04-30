import torch
import triton
import triton.language as tl


def pattern(cat_0, cat_1, cat_2, cat_3, cat_4):
    cat_out = torch.cat([cat_0, cat_1, cat_2, cat_3, cat_4], dim=1)
    return cat_out


def replacement_args(cat_0, cat_1, cat_2, cat_3, cat_4):
    return (cat_0, cat_1, cat_2, cat_3, cat_4)


@triton.jit
def _cat_kernel(
    out_ptr,
    cat0_ptr, cat1_ptr, cat2_ptr, cat3_ptr, cat4_ptr,
    n0, n01, n012, n0123, n_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_total

    # Determine which region this block falls in
    block_start = pid * BLOCK_SIZE

    if block_start < n0:
        v = tl.load(cat0_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, v, mask=mask)
    elif block_start < n01:
        off = offs - n0
        v = tl.load(cat1_ptr + off, mask=mask, other=0.0)
        tl.store(out_ptr + offs, v, mask=mask)
    elif block_start < n012:
        off = offs - n01
        v = tl.load(cat2_ptr + off, mask=mask, other=0.0)
        tl.store(out_ptr + offs, v, mask=mask)
    elif block_start < n0123:
        off = offs - n012
        v = tl.load(cat3_ptr + off, mask=mask, other=0.0)
        tl.store(out_ptr + offs, v, mask=mask)
    else:
        off = offs - n0123
        v = tl.load(cat4_ptr + off, mask=mask, other=0.0)
        tl.store(out_ptr + offs, v, mask=mask)


@torch.fx.wrap
def fused_cat_fn(cat_0, cat_1, cat_2, cat_3, cat_4):
    dtype = cat_0.dtype
    device = cat_0.device

    H = cat_0.shape[2]
    W = cat_0.shape[3]
    C_total = cat_0.shape[1] + cat_1.shape[1] + cat_2.shape[1] + cat_3.shape[1] + cat_4.shape[1]
    output = torch.empty(1, C_total, H, W, dtype=dtype, device=device)

    n0 = cat_0.numel()
    n01 = n0 + cat_1.numel()
    n012 = n01 + cat_2.numel()
    n0123 = n012 + cat_3.numel()
    n_total = n0123 + cat_4.numel()

    BLOCK_SIZE = 4096
    grid = ((n_total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _cat_kernel[grid](
        output,
        cat_0, cat_1, cat_2, cat_3, cat_4,
        n0, n01, n012, n0123, n_total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return output


def replacement_func():
    return fused_cat_fn