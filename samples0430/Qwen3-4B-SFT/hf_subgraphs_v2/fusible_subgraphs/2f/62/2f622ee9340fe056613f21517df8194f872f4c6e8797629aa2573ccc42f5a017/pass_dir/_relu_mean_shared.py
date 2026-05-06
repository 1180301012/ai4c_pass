"""
Shared Triton kernel for fused ReLU + spatial mean over (H, W) dims.
Pattern matched: relu(x, inplace=True) + x.mean((2, 3), keepdim=True)
e.g.:
  tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
  tmp_3 = tmp_0.mean((2, 3), keepdim=True)
  return (tmp_0, tmp_3)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_relu_mean_kernel(
    x_ptr,
    out_relu_ptr,
    out_mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program handles one (batch, channel) pair.
    Applies ReLU and accumulates the spatial mean in a single pass.
    Grid size: BN * C
    """
    pid = tl.program_id(0)
    base = pid * HW

    # Accumulate ReLU sums; start with zeros to avoid NaN issues with masked loads
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Process HW elements in chunks of BLOCK_HW
    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW

        # Load native dtype (float32/bfloat16/float16)
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # Apply ReLU: max(0, x)
        relu_x = tl.maximum(x, 0.0)

        # ReLU output – write back in input dtype via explicit cast
        tl.store(out_relu_ptr + base + offsets, relu_x.to(x.dtype), mask=mask)

        # Accumulate for mean
        acc = acc + tl.where(mask, relu_x, 0.0)

    # Reduce within block and divide by HW
    total = tl.sum(acc, axis=0)
    mean_val = total / HW

    # Write mean to output – Triton auto-converts float32 → store dtype
    tl.store(out_mean_ptr + pid, mean_val)


def _launch_fused_relu_mean(in_1):
    """
    Wrapper that launches the fused ReLU + mean kernel.

    Args:
        in_1: Input tensor, shape [B, C, H, W], any floating dtype.

    Returns:
        Tuple of:
          - relu_out: Same shape/dtype as in_1  (ReLU output)
          - mean_out: Shape [B, C, 1, 1]        (spatial mean)
    """
    B, C, H, W = in_1.shape
    HW = H * W
    BN = B * C

    relu_out = torch.empty_like(in_1)
    # Allocate flat [BN*C, 1, 1] – same memory layout as [BN, C, 1, 1]
    mean_out_2d = torch.empty((BN, C, 1, 1), dtype=in_1.dtype, device=in_1.device)

    _fused_relu_mean_kernel[(BN * C,)](
        in_1,
        relu_out,
        mean_out_2d,
        HW,
    )

    return relu_out, mean_out_2d


@torch.fx.wrap
def dispatch_fused_relu_mean(in_1, route):
    """
    Shared dispatch wrapper used by all divisor-specific passes.
    The `route` string selects which variant to call (all do the same
    math – only the divisor changes which pass is applied).
    """
    # The divisor is encoded in the route string ("d8", "d16", "d32").
    # We support all three routes.  The Triton kernel is independent of
    # the divisor: it's not used in the fused kernel logic.
    if route == "d8":
        relu_out, mean_out = _launch_fused_relu_mean(in_1)
        return relu_out, mean_out
    elif route == "d16":
        relu_out, mean_out = _launch_fused_relu_mean(in_1)
        return relu_out, mean_out
    elif route == "d32":
        relu_out, mean_out = _launch_fused_relu_mean(in_1)
        return relu_out, mean_out
    else:
        relu_out, mean_out = _launch_fused_relu_mean(in_1)
        return relu_out, mean_out