import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: global average pool over spatial dims of [N, C, H, W]
# One program per (n, c) pair; reduces H*W elements into a single mean.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW', 'DTYPE']
)
@triton.jit
def avgpool_kernel(
    in_ptr, out_ptr,
    HW,
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * HW
    acc  = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for i in range(0, HW, BLOCK_HW):
        offsets = i + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        val  = tl.load(in_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc  = acc + val

    total = tl.sum(acc, axis=0)
    avg   = total / HW

    if DTYPE == 'fp16':
        tl.store(out_ptr + pid, avg.to(tl.float16))
    elif DTYPE == 'bf16':
        tl.store(out_ptr + pid, avg.to(tl.bfloat16))
    else:
        tl.store(out_ptr + pid, avg)


# ---------------------------------------------------------------------------
# Diagnostic pattern: match ONLY adaptive_avg_pool2d(x, 1)
# This isolates which op-form is actually in the compiled graph.
# ---------------------------------------------------------------------------
def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, 1)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def triton_avgpool(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty(N, C, 1, 1, dtype=x.dtype, device=x.device)

    if x.dtype == torch.float16:
        dtype_str = 'fp16'
    elif x.dtype == torch.bfloat16:
        dtype_str = 'bf16'
    else:
        dtype_str = 'fp32'

    avgpool_kernel[(NC,)](
        x, out,
        HW,
        DTYPE=dtype_str,
    )
    return out


def replacement_func():
    return triton_avgpool