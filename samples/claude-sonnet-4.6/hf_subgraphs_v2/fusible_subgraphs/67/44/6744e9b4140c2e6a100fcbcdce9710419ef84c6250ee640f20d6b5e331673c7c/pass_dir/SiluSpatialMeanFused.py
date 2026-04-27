import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    return (tmp_0, tmp_1)


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},  num_warps=1),
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _silu_spatial_mean_kernel(
    x_ptr,
    silu_ptr,
    mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program (pid) handles one (batch, channel) pair.
    - Reads HW elements, applies SiLU, writes them back.
    - Accumulates sum in float32, then divides by HW to get mean.
    - Stores mean to mean_ptr[pid].
    """
    pid = tl.program_id(0)
    base = pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for i in range(0, HW, BLOCK_HW):
        idx = tl.arange(0, BLOCK_HW)
        offsets = base + i + idx
        mask = (i + idx) < HW
        # Load; out-of-bounds padded with 0.0  (silu(0) == 0, safe for accumulator)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        silu_val = x_f32 * tl.sigmoid(x_f32)
        # Write SiLU output (auto-cast back to x_ptr dtype)
        tl.store(silu_ptr + offsets, silu_val, mask=mask)
        # Accumulate (masked padding is 0, so silu(0)=0 contributes nothing)
        acc += silu_val

    mean_val = tl.sum(acc) / HW
    # Store scalar mean (auto-cast to mean_ptr dtype)
    tl.store(mean_ptr + pid, mean_val)


@torch.fx.wrap
def silu_spatial_mean_fused(in_1):
    B, C, H, W = in_1.shape
    HW = H * W
    BC = B * C

    silu_out = torch.empty_like(in_1)
    # mean output matches tmp_1 = tmp_0.mean((2,3)) → shape [B, C]
    mean_out = torch.empty((B, C), dtype=in_1.dtype, device=in_1.device)

    _silu_spatial_mean_kernel[(BC,)](
        in_1, silu_out, mean_out,
        HW=HW,
    )

    return (silu_out, mean_out)


def replacement_func():
    return silu_spatial_mean_fused