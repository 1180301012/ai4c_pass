import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def l2_normalize_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Process row 0
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x0 = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm_sq0 = tl.sum(x0 * x0)
    inv_norm0 = 1.0 / tl.sqrt(norm_sq0)
    r0 = x0 * inv_norm0
    tl.store(out_ptr + offsets, r0.to(tl.bfloat16), mask=mask)

    # Process row 1
    x1 = tl.load(in_ptr + N + offsets, mask=mask, other=0.0).to(tl.float32)
    norm_sq1 = tl.sum(x1 * x1)
    inv_norm1 = 1.0 / tl.sqrt(norm_sq1)
    r1 = x1 * inv_norm1
    tl.store(out_ptr + N + offsets, r1.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def l2_normalize(in_1):
    N = in_1.shape[1]
    out = torch.empty_like(in_1)
    BLOCK_SIZE = 1024 if N <= 1024 else 2048
    l2_normalize_kernel[(1,)](
        in_ptr=in_1,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return l2_normalize