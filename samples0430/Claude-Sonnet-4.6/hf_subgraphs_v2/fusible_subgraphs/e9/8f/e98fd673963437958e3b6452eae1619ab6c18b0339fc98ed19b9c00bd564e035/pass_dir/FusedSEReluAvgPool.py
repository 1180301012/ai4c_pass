import operator
import torch
import triton
import triton.language as tl


# --- Pass 2: fuse relu(inplace) + adaptive_avg_pool2d + flatten ---
# This is the high-impact pass: avoids materialising the [C,H,W] relu output
# and does the reduction in the same kernel (halves memory bandwidth for that stage).

def pattern(in_0, in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_4 = operator.iadd(tmp_3, in_0)   # call_function iadd -> matches target
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_se_relu_avgpool_kernel(
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    # One program per channel
    c = tl.program_id(0)

    # in_2 has shape [1, 1, C] — element c is at contiguous offset c
    scale_raw = tl.load(in2_ptr + c).to(tl.float32)
    scale = tl.sigmoid(scale_raw)

    # in_0, in_1 have shape [1, C, H, W] — channel c starts at offset c * HW
    base = c * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    v0 = tl.load(in0_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

    # relu(in_1 * scale + in_0)
    val = v1 * scale + v0
    val = tl.maximum(val, 0.0)

    # Mask out-of-bounds, then sum -> average
    val = tl.where(mask, val, 0.0)
    acc = tl.sum(val)
    out_val = acc / HW

    # Store to out[0, c] — offset c in [1, C] output
    tl.store(out_ptr + c, out_val)


@torch.fx.wrap
def fused_se_relu_avgpool(in_0, in_1, in_2):
    B, C, H, W = in_0.shape
    HW = H * W

    # Output shape matches flatten(adaptive_avg_pool2d(x, 1), 1, -1) => [B, C]
    out = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)

    _fused_se_relu_avgpool_kernel[(C,)](
        in_0, in_1, in_2,
        out,
        C, HW,
    )

    return (out,)


def replacement_func():
    return fused_se_relu_avgpool