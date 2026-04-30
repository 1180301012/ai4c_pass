import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_cat_slice_mean_kernel(
    in_ptr,
    out_mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused cat+slice+mean kernel.
    Since cat([in_0, in_1])[:, :C, :, :] == in_0, we simply read from in_ptr.
    Each program handles one (batch, channel) pair and reduces over H*W elements.
    in_ptr layout:  [B, C, H, W]  (contiguous)
    out_mean_ptr layout: [B, C, 1, 1] (contiguous)
    """
    pid = tl.program_id(0)   # pid = b * C + c
    base = pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for offset in range(0, HW, BLOCK_HW):
        hw_offsets = offset + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW
        x = tl.load(in_ptr + base + hw_offsets, mask=mask, other=0.0)
        acc += x.to(tl.float32)

    mean_val = tl.sum(acc) / HW
    tl.store(out_mean_ptr + pid, mean_val)


@torch.fx.wrap
def triton_fused_cat_slice_mean(x):
    """
    Replacement for: cat([x, y], dim=1)[:, :C, :, :].mean((2, 3), keepdim=True)
    Since the slice returns x unchanged, we just compute mean of x over H, W.
    Returns (x, mean_result) matching the original (tmp_1, tmp_2) outputs.
    """
    B, C, H, W = x.shape
    HW = H * W

    # tmp_1 is x itself (the slice is a no-op), but we still need to return it.
    # We allocate out_mean and launch the kernel.
    out_mean = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    grid = lambda meta: (B * C,)
    fused_cat_slice_mean_kernel[grid](x, out_mean, HW)

    return x, out_mean