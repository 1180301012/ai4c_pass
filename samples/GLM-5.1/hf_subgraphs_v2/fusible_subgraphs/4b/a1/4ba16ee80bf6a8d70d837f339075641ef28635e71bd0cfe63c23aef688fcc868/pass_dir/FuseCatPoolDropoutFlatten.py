import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_pool_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr,
    C0, C1, C2, C3,
    total_C, HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c_off = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_off < total_C

    m0 = c_off < C0
    m1 = (c_off >= C0) & (c_off < C0 + C1)
    m2 = (c_off >= C0 + C1) & (c_off < C0 + C1 + C2)
    m3 = c_off >= C0 + C1 + C2

    lc0 = c_off
    lc1 = c_off - C0
    lc2 = c_off - C0 - C1
    lc3 = c_off - C0 - C1 - C2

    base = tl.where(m0, in0_ptr + lc0 * HW,
           tl.where(m1, in1_ptr + lc1 * HW,
           tl.where(m2, in2_ptr + lc2 * HW,
                    in3_ptr + lc3 * HW)))

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for hw in range(HW):
        acc += tl.load(base + hw, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + c_off, acc / HW, mask=c_mask)


@torch.fx.wrap
def fused_cat_pool_dropout_flatten(in_0, in_1, in_2, in_3):
    B = in_0.shape[0]
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    total_C = C0 + C1 + C2 + C3
    H = in_0.shape[2]
    W = in_0.shape[3]
    HW = H * W

    out = torch.empty((B, total_C), dtype=in_0.dtype, device=in_0.device)

    BLOCK_C = 2048
    grid = 1

    fused_pool_kernel[(grid,)](
        in_0, in_1, in_2, in_3, out,
        C0, C1, C2, C3,
        total_C, HW=HW,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )

    return out


def replacement_func():
    return fused_cat_pool_dropout_flatten