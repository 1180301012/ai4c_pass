import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large-HW configs work well for big tensors; small-HW configs for BC loops
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=5),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_HW': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 4096}, num_warps=4, num_stages=4),
        # Small-HW compact configs: loop over HW with few pipeline stages
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=2),
    ],
    key=['HW', 'BC'],
)
@triton.jit
def _bn_inf_kernel(
    x_ptr, rm_ptr, rv_ptr, w_ptr, b_ptr, out_ptr,
    HW, C, BC,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each CTA handles ONE (batch, channel) pair and loops over HW tiles.
    # Using a 1-D grid of size BC keeps CTA count low even for small-B tensors
    # (e.g. BC=10 instead of 31 360 for B=1, C=10, HW=3136), which dramatically
    # cuts scheduling overhead for small-tensor inference.
    pid_bc = tl.program_id(0)
    c      = pid_bc % C

    # Load per-channel BN parameters once per CTA (broadcast across all tiles)
    rm = tl.load(rm_ptr + c).to(tl.float32)
    rv = tl.load(rv_ptr + c).to(tl.float32)
    w  = tl.load(w_ptr  + c).to(tl.float32)
    b  = tl.load(b_ptr  + c).to(tl.float32)

    # Precompute the fused affine coefficients in fp32 for numerical stability
    inv_std = tl.rsqrt(rv + eps)
    scale   = w * inv_std
    offset  = b - scale * rm

    base = pid_bc * HW

    # Loop over HW tiles — num_stages pipelines loads across iterations
    for start in range(0, HW, BLOCK_HW):
        hw_off = start + tl.arange(0, BLOCK_HW)
        mask   = hw_off < HW

        x = tl.load(x_ptr + base + hw_off, mask=mask, other=0.0).to(tl.float32)
        y = scale * x + offset

        tl.store(out_ptr + base + hw_off, y, mask=mask)


@torch.fx.wrap
def triton_bn_inference(x, running_mean, running_var, weight, bias):
    B, C, H, W = x.shape
    HW  = H * W
    BC  = B * C
    EPS = 0.001   # matches model.py: eps=0.001

    out = torch.empty_like(x)

    # 1-D grid: one CTA per (batch, channel) pair
    _bn_inf_kernel[(BC,)](
        x, running_mean, running_var, weight, bias, out,
        HW, C, BC,
        EPS,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 0.001
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


def replacement_func():
    return triton_bn_inference