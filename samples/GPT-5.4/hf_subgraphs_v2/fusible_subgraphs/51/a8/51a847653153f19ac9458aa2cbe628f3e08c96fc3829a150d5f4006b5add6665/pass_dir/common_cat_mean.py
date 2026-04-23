import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def _cat_mean_hw_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    mean_ptr,
    C,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    TILES: tl.constexpr,
):
    n = tl.program_id(0)
    c_local = tl.program_id(1)

    in_base = ((n * C + c_local) * H) * W
    out0_base = ((n * (2 * C) + c_local) * H) * W
    out1_base = ((n * (2 * C) + (c_local + C)) * H) * W

    acc0 = 0.0
    acc1 = 0.0

    for tile_idx in tl.static_range(0, TILES):
        offs = tile_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs < HW

        vals0 = tl.load(in0_ptr + in_base + offs, mask=mask, other=0.0)
        vals1 = tl.load(in1_ptr + in_base + offs, mask=mask, other=0.0)

        tl.store(out_ptr + out0_base + offs, vals0, mask=mask)
        tl.store(out_ptr + out1_base + offs, vals1, mask=mask)

        acc0 += tl.sum(vals0.to(tl.float32), axis=0)
        acc1 += tl.sum(vals1.to(tl.float32), axis=0)

    mean0 = acc0 / HW
    mean1 = acc1 / HW
    tl.store(mean_ptr + n * (2 * C) + c_local, mean0)
    tl.store(mean_ptr + n * (2 * C) + (c_local + C), mean1)


def _launch_hw(in_0, in_1, out, mean, h, w, block_hw, num_warps):
    n, c, _, _ = in_0.size()
    hw = h * w
    tiles = (hw + block_hw - 1) // block_hw
    grid = (n, c)
    _cat_mean_hw_kernel[grid](
        in_0,
        in_1,
        out,
        mean,
        c,
        H=h,
        W=w,
        HW=hw,
        BLOCK_HW=block_hw,
        TILES=tiles,
        num_warps=num_warps,
    )


@torch.fx.wrap
def fused_cat_mean(in_0, in_1):
    in_0 = unwrap_tensor(in_0)
    in_1 = unwrap_tensor(in_1)

    n, c, h, w = in_0.size()
    out = torch.empty((n, c * 2, h, w), device=in_0.device, dtype=in_0.dtype)
    mean = torch.empty((n, c * 2, 1, 1), device=in_0.device, dtype=in_0.dtype)

    if h == 8 and w == 8:
        _launch_hw(in_0, in_1, out, mean, 8, 8, 64, 2)
    elif h == 14 and w == 14:
        _launch_hw(in_0, in_1, out, mean, 14, 14, 256, 4)
    elif h == 16 and w == 16:
        _launch_hw(in_0, in_1, out, mean, 16, 16, 256, 4)
    elif h == 24 and w == 24:
        _launch_hw(in_0, in_1, out, mean, 24, 24, 256, 4)
    elif h == 28 and w == 28:
        _launch_hw(in_0, in_1, out, mean, 28, 28, 256, 4)
    elif h == 32 and w == 32:
        _launch_hw(in_0, in_1, out, mean, 32, 32, 256, 4)
    elif h == 48 and w == 48:
        _launch_hw(in_0, in_1, out, mean, 48, 48, 256, 8)
    else:
        raise RuntimeError(f"Unsupported spatial shape: {(h, w)}")

    return out, mean


def shared_replacement_func():
    return fused_cat_mean