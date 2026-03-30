"""
Pass: FuseAdd2Mean
Pattern: tmp_0 = 0 + in_1; tmp_0 += in_0; tmp_2 = tmp_0.mean((2,3), keepdim=True); return (tmp_0, tmp_2)
Optimization: Fuse two-tensor add + spatial mean into a single kernel pass.
"""
import torch
import triton
import triton.language as tl


# ─── Triton kernel ────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _add2_mean_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    mean_ptr,
    HW,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """One CTA per (N,C) slice. Elementwise-adds two inputs and accumulates mean."""
    pid  = tl.program_id(0)
    base = pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW

        x0 = tl.load(in0_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(in1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        s  = x0 + x1

        if IS_FP16:
            tl.store(out_ptr + base + offs, s.to(tl.float16), mask=mask)
        elif IS_BF16:
            tl.store(out_ptr + base + offs, s.to(tl.bfloat16), mask=mask)
        else:
            tl.store(out_ptr + base + offs, s, mask=mask)

        acc += s

    total    = tl.sum(acc)
    mean_val = total / HW

    if IS_FP16:
        tl.store(mean_ptr + pid, mean_val.to(tl.float16))
    elif IS_BF16:
        tl.store(mean_ptr + pid, mean_val.to(tl.bfloat16))
    else:
        tl.store(mean_ptr + pid, mean_val)


# ─── Python wrapper ───────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_add2_mean(in_0, in_1):
    N, C, H, W = in_0.shape
    HW = H * W
    NC = N * C

    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()

    out       = torch.empty_like(in_0)
    mean_flat = torch.empty(NC, dtype=in_0.dtype, device=in_0.device)

    is_fp16 = in_0.dtype == torch.float16
    is_bf16 = in_0.dtype == torch.bfloat16

    _add2_mean_kernel[(NC,)](
        in_0, in_1, out, mean_flat,
        HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    mean_out = mean_flat.view(N, C, 1, 1)
    return (out, mean_out)


# ─── Pass interface ───────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add2_mean