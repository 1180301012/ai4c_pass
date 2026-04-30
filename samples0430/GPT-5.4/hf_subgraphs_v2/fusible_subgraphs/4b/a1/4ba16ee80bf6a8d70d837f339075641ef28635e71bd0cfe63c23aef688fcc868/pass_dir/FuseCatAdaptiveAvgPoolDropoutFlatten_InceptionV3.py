import torch
import triton
import triton.language as tl


# Match the full subgraph exactly.
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_cat_pool_flatten_fixed_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)

    # Channels are exactly 2048 and BLOCK_C is fixed at 64, so no tail mask is needed.
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # branch 0: channels [0, 320)
    m0 = offs_c < 320
    c0 = offs_c
    base0 = in_0_ptr + c0 * 25

    # branch 1: channels [320, 1088)
    m1 = (offs_c >= 320) & (offs_c < 1088)
    c1 = offs_c - 320
    base1 = in_1_ptr + c1 * 25

    # branch 2: channels [1088, 1856)
    m2 = (offs_c >= 1088) & (offs_c < 1856)
    c2 = offs_c - 1088
    base2 = in_2_ptr + c2 * 25

    # branch 3: channels [1856, 2048)
    m3 = offs_c >= 1856
    c3 = offs_c - 1856
    base3 = in_3_ptr + c3 * 25

    for hw in range(25):
        acc += tl.load(base0 + hw, mask=m0, other=0.0).to(tl.float32)
        acc += tl.load(base1 + hw, mask=m1, other=0.0).to(tl.float32)
        acc += tl.load(base2 + hw, mask=m2, other=0.0).to(tl.float32)
        acc += tl.load(base3 + hw, mask=m3, other=0.0).to(tl.float32)

    tl.store(out_ptr + offs_c, acc * 0.04)


@torch.fx.wrap
def fused_cat_adaptive_avg_pool_dropout_flatten(in_0, in_1, in_2, in_3):
    out = torch.empty((1, 2048), device=in_0.device, dtype=in_0.dtype)
    fused_cat_pool_flatten_fixed_kernel[(32,)](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        BLOCK_C=64,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_cat_adaptive_avg_pool_dropout_flatten