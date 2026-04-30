import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _fused_silu_avgpool_kernel_ni(
    x_ptr,
    out_ptr,
    C,
    HW,
    stride_n,
    stride_c,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program handles one (n, c) pair.
    Computes: mean_{h,w}(silu(x[n, c, h, w]))
    Equivalent to silu -> adaptive_avg_pool2d(output_size=1) -> flatten(1).
    Dropout with training=False is a no-op, so it's absorbed.
    No inplace flag - handles the case where inplace is normalized away.
    """
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base_offset = n * stride_n + c * stride_c
    hw_offsets = tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    x_offsets = base_offset + hw_offsets
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

    # Upcast to float32 for numerically stable computation
    x_fp32 = x.to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_x = x_fp32 * tl.sigmoid(x_fp32)

    # Zero out padding positions
    silu_x = tl.where(mask, silu_x, 0.0)

    # Global average pooling: sum / HW
    total = tl.sum(silu_x, axis=0)
    result = total / HW

    # Store (Triton auto-converts float32 to output pointer dtype)
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def fused_silu_avgpool_flatten_ni(x):
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty((N, C), dtype=x.dtype, device=x.device)

    grid = (N * C,)
    _fused_silu_avgpool_kernel_ni[grid](
        x, out,
        C, HW,
        C * HW,   # stride_n
        HW,       # stride_c
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API - no inplace flag on silu
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_silu_avgpool_flatten_ni