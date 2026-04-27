import torch
import triton
import triton.language as tl


def pattern(x):
    # The _decomposed graphs use aten-level ops
    silu_out = torch.ops.aten.silu_.default(x)
    mean_out = torch.ops.aten.mean.dim(silu_out, [2, 3], True)
    return silu_out, mean_out


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _silu_mean_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one (n, c) pair.
    Loads H*W elements, applies SiLU, stores result, and accumulates mean.
    """
    pid = tl.program_id(0)  # linearized (n, c) index
    base = pid * HW
    acc = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW

        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x.to(tl.float32)
        # SiLU: y * sigmoid(y)
        z = y * tl.sigmoid(y)
        # Store back in original dtype
        tl.store(out_ptr + base + offs, z.to(x.dtype), mask=mask)
        # Accumulate sum for mean
        acc += tl.sum(tl.where(mask, z, 0.0), axis=0)

    # Store mean
    tl.store(mean_ptr + pid, acc / HW)


@torch.fx.wrap
def fused_silu_mean_wrapper(x):
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty_like(x)
    # Accumulate mean in float32 for accuracy, shape (N*C,)
    mean_f32 = torch.empty((N * C,), dtype=torch.float32, device=x.device)

    _silu_mean_kernel[(N * C,)](
        x, out, mean_f32,
        C, HW,
    )

    # Cast mean to input dtype and reshape to [N, C, 1, 1]
    mean_out = mean_f32.to(x.dtype).view(N, C, 1, 1)
    return out, mean_out


def replacement_func():
    return fused_silu_mean_wrapper