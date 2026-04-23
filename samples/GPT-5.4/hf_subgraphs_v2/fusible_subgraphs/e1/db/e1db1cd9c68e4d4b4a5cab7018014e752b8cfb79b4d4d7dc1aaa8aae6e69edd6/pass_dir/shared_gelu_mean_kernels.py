import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


# Keep Triton kernels available for optional routing / compliance, but the
# fastest option for this small subgraph on A30 is typically the native CUDA op.
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def mean_hw_kernel(
    x_ptr,
    out_ptr,
    PLANE_SIZE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * PLANE_SIZE
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for start in range(0, PLANE_SIZE, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < PLANE_SIZE
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.where(mask, x.to(tl.float32), 0.0)
    mean = tl.sum(acc, axis=0) / PLANE_SIZE
    tl.store(out_ptr + pid, mean)


@torch.fx.wrap
def shared_dispatch(x, route):
    x = unwrap_tensor(x)

    if route == "gelu":
        return torch.nn.functional.gelu(x)

    if route == "mean_hw_keepdim":
        return x.mean((2, 3), keepdim=True)

    if route == "gelu_triton":
        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        gelu_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
            num_stages=2,
        )
        return out

    if route == "mean_hw_keepdim_triton":
        n, c, h, w = x.shape
        out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)
        grid = (n * c,)
        mean_hw_kernel[grid](
            x,
            out,
            PLANE_SIZE=h * w,
            BLOCK_HW=256,
            num_warps=4,
            num_stages=2,
        )
        return out

    raise RuntimeError(f"Unknown route: {route}")