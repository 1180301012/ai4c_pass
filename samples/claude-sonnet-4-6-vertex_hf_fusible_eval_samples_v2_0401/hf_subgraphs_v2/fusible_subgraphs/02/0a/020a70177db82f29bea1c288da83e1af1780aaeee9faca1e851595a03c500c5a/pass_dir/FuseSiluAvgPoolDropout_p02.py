"""
Optimized fused pass: AdaptiveAvgPool2d(1) + Flatten (pattern)

The matched 'in_0' maps to the silu_ output node in the model, so the
replacement receives the silu-already-applied tensor and needs to perform
global average pooling + flatten efficiently.

Hybrid approach:
 - Small workloads (B*C < threshold): PyTorch mean (lower launch overhead)
 - Large workloads (B*C >= threshold): Triton kernel (better throughput)
"""

import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(in_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 128},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def _avgpool_flatten_kernel(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW

    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    x = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(mask, x, 0.0), axis=0)

    for i in range(BLOCK_HW, HW, BLOCK_HW):
        offsets2 = i + tl.arange(0, BLOCK_HW)
        mask2 = offsets2 < HW
        x2 = tl.load(input_ptr + base + offsets2, mask=mask2, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask2, x2, 0.0), axis=0)

    result = acc / HW
    tl.store(output_ptr + pid, result)


# Threshold: below this, PyTorch mean is faster due to lower kernel overhead
_TRITON_THRESHOLD = 8192


@torch.fx.wrap
def _fused_silu_avgpool_flatten_p02(in_0):
    B, C, H, W = in_0.shape
    BC = B * C
    HW = H * W

    if BC < _TRITON_THRESHOLD:
        # Small workload: PyTorch mean avoids Triton kernel launch overhead
        # mean over [H, W] dims → shape [B, C]
        return in_0.mean(dim=[2, 3])

    # Large workload: Triton kernel for better throughput
    in_0_contig = in_0.contiguous()
    output_f32 = torch.empty(BC, dtype=torch.float32, device=in_0.device)

    _avgpool_flatten_kernel[(BC,)](
        in_0_contig,
        output_f32,
        HW,
    )

    return output_f32.to(in_0.dtype).view(B, C)


def replacement_func():
    return _fused_silu_avgpool_flatten_p02