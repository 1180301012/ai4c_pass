import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['H', 'W'],
)
@triton.jit
def mean_only_kernel(
    in_ptr,
    mean_out_ptr,
    N, C, H, W,
    BLOCK_HW: tl.constexpr,
):
    nc = tl.program_id(0)
    n = nc // C
    c = nc % C

    base_offset = n * (C * H * W) + c * (H * W)
    hw = H * W

    acc = 0.0

    for hw_start in range(0, hw, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < hw
        idx = base_offset + offsets

        val = tl.load(in_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(val.to(tl.float32), axis=0)

    mean_val = acc / hw
    tl.store(mean_out_ptr + nc, mean_val)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['H', 'W'],
)
@triton.jit
def add2_mean_kernel(
    in0_ptr, in1_ptr,
    sum_out_ptr, mean_out_ptr,
    N, C, H, W,
    BLOCK_HW: tl.constexpr,
):
    nc = tl.program_id(0)
    n = nc // C
    c = nc % C

    base_offset = n * (C * H * W) + c * (H * W)
    hw = H * W

    acc = 0.0

    for hw_start in range(0, hw, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < hw
        idx = base_offset + offsets

        v0 = tl.load(in0_ptr + idx, mask=mask, other=0.0)
        v1 = tl.load(in1_ptr + idx, mask=mask, other=0.0)

        s = v0 + v1
        tl.store(sum_out_ptr + idx, s, mask=mask)

        acc += tl.sum(s.to(tl.float32), axis=0)

    mean_val = acc / hw
    tl.store(mean_out_ptr + nc, mean_val)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['H', 'W'],
)
@triton.jit
def add3_mean_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    sum_out_ptr, mean_out_ptr,
    N, C, H, W,
    BLOCK_HW: tl.constexpr,
):
    nc = tl.program_id(0)
    n = nc // C
    c = nc % C

    base_offset = n * (C * H * W) + c * (H * W)
    hw = H * W

    acc = 0.0

    for hw_start in range(0, hw, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < hw
        idx = base_offset + offsets

        v0 = tl.load(in0_ptr + idx, mask=mask, other=0.0)
        v1 = tl.load(in1_ptr + idx, mask=mask, other=0.0)
        v2 = tl.load(in2_ptr + idx, mask=mask, other=0.0)

        s = v0 + v1 + v2
        tl.store(sum_out_ptr + idx, s, mask=mask)

        acc += tl.sum(s.to(tl.float32), axis=0)

    mean_val = acc / hw
    tl.store(mean_out_ptr + nc, mean_val)


def _identity_mean(in_0):
    N, C, H, W = in_0.shape
    mean_out = torch.empty((N, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    grid = (N * C,)
    mean_only_kernel[grid](
        in_ptr=in_0,
        mean_out_ptr=mean_out,
        N=N, C=C, H=H, W=W,
    )
    return (in_0, mean_out)


def _add2_mean(in_0, in_1):
    N, C, H, W = in_0.shape
    sum_out = torch.empty_like(in_0)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    grid = (N * C,)
    add2_mean_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        sum_out_ptr=sum_out,
        mean_out_ptr=mean_out,
        N=N, C=C, H=H, W=W,
    )
    return (sum_out, mean_out)


def _add3_mean(in_0, in_1, in_2):
    N, C, H, W = in_0.shape
    sum_out = torch.empty_like(in_0)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    grid = (N * C,)
    add3_mean_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        sum_out_ptr=sum_out,
        mean_out_ptr=mean_out,
        N=N, C=C, H=H, W=W,
    )
    return (sum_out, mean_out)


@torch.fx.wrap
def fused_dispatch(*args):
    route = args[-1]
    if route == "identity_mean":
        return _identity_mean(args[0])
    elif route == "add0_mean":
        return _add2_mean(args[0], args[1])
    elif route == "add_add_mean":
        return _add3_mean(args[0], args[1], args[2])
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return fused_dispatch