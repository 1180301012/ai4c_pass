import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_gelu_mean_hw_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    num_planes,
    PLANE_SIZE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)

    # One program handles one [H, W] plane for a fixed (N, C).
    base = pid * PLANE_SIZE
    acc = tl.zeros((1,), dtype=tl.float32)

    for start in range(0, PLANE_SIZE, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < PLANE_SIZE

        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        y_f32 = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

        tl.store(out_ptr + base + offs, y_f32, mask=mask)

        partial = tl.sum(tl.where(mask, y_f32, 0.0), axis=0)
        acc += partial

    mean = acc / PLANE_SIZE
    tl.store(mean_ptr + pid + tl.arange(0, 1), mean)


@torch.fx.wrap
def _fused_gelu_mean_hw_impl(in_0):
    n, c, h, w = in_0.shape
    num_planes = n * c
    plane_size = h * w

    out = torch.empty_like(in_0)
    out_mean = torch.empty((n, c, 1, 1), device=in_0.device, dtype=in_0.dtype)

    grid = (num_planes,)
    fused_gelu_mean_hw_kernel[grid](
        in_0,
        out,
        out_mean,
        num_planes,
        PLANE_SIZE=plane_size,
        BLOCK_HW=256,
        num_warps=4,
        num_stages=2,
    )

    return (out, out_mean)


@torch.fx.wrap
def fused_gelu_mean_hw(in_0):
    outs = _fused_gelu_mean_hw_impl(in_0)
    return (outs[0], outs[1])


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_gelu_mean_hw