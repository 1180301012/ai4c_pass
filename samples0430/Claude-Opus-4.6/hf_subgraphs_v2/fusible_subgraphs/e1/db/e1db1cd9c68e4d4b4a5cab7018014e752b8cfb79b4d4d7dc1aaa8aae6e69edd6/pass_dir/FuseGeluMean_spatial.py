import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "route_gelu")


@triton.jit
def gelu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_val = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

    tl.store(out_ptr + offsets, gelu_val.to(x.dtype), mask=mask)


def _run_gelu(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    gelu_kernel[(num_programs,)](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out


@triton.jit
def mean_spatial_kernel(
    in_ptr,
    out_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * spatial_size

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size

    x = tl.load(in_ptr + base_offset + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    acc = tl.where(mask, x_f32, 0.0)
    mean_val = tl.sum(acc, axis=0) / spatial_size

    tl.store(out_ptr + pid, mean_val.to(x.dtype))


def _run_mean_spatial(x):
    B, C, H, W = x.shape
    spatial_size = H * W

    out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    num_programs = B * C
    BLOCK_SIZE = 4096

    mean_spatial_kernel[(num_programs,)](
        x,
        out,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return out


@torch.fx.wrap
def dispatch_wrapper(x, route):
    if route == "route_gelu":
        return _run_gelu(x)
    elif route == "route_mean_spatial":
        return _run_mean_spatial(x)
    return x


def replacement_func():
    return dispatch_wrapper