import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu(inplace=True) → adaptive_avg_pool2d(1) → flatten(1,-1)
# ---------------------------------------------------------------------------

def pattern(tmp):
    tmp_5 = torch.nn.functional.relu(tmp, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return (tmp_7,)


def replacement_args(tmp):
    return (tmp,)


# ---------------------------------------------------------------------------
# Fused Triton kernel:
#   - One program per channel c  (grid = B*C)
#   - Loads all HW elements for that channel from `tmp`
#   - Applies relu element-wise, accumulates, divides by HW
#   - Stores the average into out[c]
# This fuses relu + adaptive_avg_pool2d + flatten into one kernel,
# eliminating two intermediate buffers and their associated memory traffic.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _relu_avgpool_flatten_kernel(
    tmp_ptr,            # [B, C, H, W] – input
    out_ptr,            # [B, C]        – output
    C,                  # number of channels
    HW,                 # H * W
    BLOCK_HW: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    c = tl.program_id(0)       # one program per (batch=0, channel=c)

    base = c * HW
    offs = tl.arange(0, BLOCK_HW)
    acc = 0.0

    for start in range(0, HW, BLOCK_HW):
        cur = offs + start
        mask = cur < HW
        val = tl.load(tmp_ptr + base + cur, mask=mask, other=0.0).to(tl.float32)
        # relu: max(val, 0)  (masked positions already 0 via other=0.0 above)
        val = tl.maximum(val, 0.0)
        val = tl.where(mask, val, 0.0)
        acc = acc + tl.sum(val, axis=0)

    avg = acc / HW
    tl.store(out_ptr + c, avg.to(OUT_DTYPE))


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------

_TRITON_DTYPE = {
    torch.float16:  tl.float16,
    torch.float32:  tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@torch.fx.wrap
def fused_relu_avgpool_flatten(tmp):
    B, C, H, W = tmp.shape
    HW = H * W
    out = torch.empty((B, C), dtype=tmp.dtype, device=tmp.device)

    _relu_avgpool_flatten_kernel[(B * C,)](
        tmp, out,
        C, HW,
        OUT_DTYPE=_TRITON_DTYPE[tmp.dtype],
    )
    return (out,)


def replacement_func():
    return fused_relu_avgpool_flatten