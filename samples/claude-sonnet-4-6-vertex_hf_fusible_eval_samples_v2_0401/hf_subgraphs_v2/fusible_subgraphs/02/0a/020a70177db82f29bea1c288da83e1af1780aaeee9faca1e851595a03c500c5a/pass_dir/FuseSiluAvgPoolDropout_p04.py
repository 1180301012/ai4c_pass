"""
Fused pass: SiLU + AdaptiveAvgPool2d(1) + Flatten + Dropout(p=0.4, training=False)

The fused kernel computes: output[b, c] = mean(SiLU(input[b, c, :, :]))
for all (b, c) pairs in parallel.  Dropout at inference is a no-op.
"""

import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1);  tmp_0 = None
    tmp_2 = torch.flatten(tmp_1, 1);  tmp_1 = None
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.4, False, True);  tmp_2 = None
    return (tmp_3,)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def _silu_avgpool_kernel_p04(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for i in range(0, HW, BLOCK_HW):
        offsets = i + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        x = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        silu_x = x * tl.sigmoid(x)
        acc = acc + tl.where(mask, silu_x, tl.zeros([BLOCK_HW], dtype=tl.float32))

    result = tl.sum(acc) / HW
    tl.store(output_ptr + pid, result)


@torch.fx.wrap
def _fused_silu_avgpool_flatten_p04(in_0):
    B, C, H, W = in_0.shape
    HW = H * W

    in_0_contig = in_0.contiguous()
    output_f32 = torch.empty(B * C, dtype=torch.float32, device=in_0.device)

    _silu_avgpool_kernel_p04[(B * C,)](
        in_0_contig,
        output_f32,
        HW,
    )

    return (output_f32.to(in_0.dtype).view(B, C),)


def replacement_func():
    return _fused_silu_avgpool_flatten_p04