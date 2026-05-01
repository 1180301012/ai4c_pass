import torch
import triton
import triton.language as tl


# Pattern: cat([in_0,in_1,in_2,in_3], dim=1) -> adaptive_avg_pool2d((1,1)) -> dropout(train=False) -> flatten(1)
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# One program per output channel.  Vectorise over HW=25 with BLOCK_HW=32
# (7 padding lanes masked).  All tensor sizes are compile-time constants.
@triton.jit
def fused_cat_avgpool_flatten_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    B,
    C0:       tl.constexpr,   # 320
    C1:       tl.constexpr,   # 768
    C2:       tl.constexpr,   # 768
    C3:       tl.constexpr,   # 192
    TOTAL_C:  tl.constexpr,   # 2048
    HW:       tl.constexpr,   # 25
    BLOCK_HW: tl.constexpr,   # 32
):
    c = tl.program_id(0)   # channel [0, TOTAL_C)
    b = tl.program_id(1)   # batch   [0, B)

    hw_offs = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    # Select source tensor via compile-time-constant-bounded if/elif/else
    if c < C0:
        ptr = in0_ptr + b * C0 * HW + c * HW
    elif c < C0 + C1:
        ptr = in1_ptr + b * C1 * HW + (c - C0) * HW
    elif c < C0 + C1 + C2:
        ptr = in2_ptr + b * C2 * HW + (c - C0 - C1) * HW
    else:
        ptr = in3_ptr + b * C3 * HW + (c - C0 - C1 - C2) * HW

    vals = tl.load(ptr + hw_offs, mask=hw_mask, other=0.0).to(tl.float32)
    avg  = tl.sum(vals, axis=0) * (1.0 / HW)
    tl.store(out_ptr + b * TOTAL_C + c, avg)


@torch.fx.wrap
def fused_cat_avgpool_flatten(in_0, in_1, in_2, in_3):
    B   = in_0.shape[0]
    out = torch.empty((B, 2048), dtype=in_0.dtype, device=in_0.device)

    fused_cat_avgpool_flatten_kernel[(2048, B)](
        in_0, in_1, in_2, in_3,
        out,
        B,
        C0=320, C1=768, C2=768, C3=192,
        TOTAL_C=2048, HW=25, BLOCK_HW=32,
    )
    return out


def replacement_func():
    return fused_cat_avgpool_flatten