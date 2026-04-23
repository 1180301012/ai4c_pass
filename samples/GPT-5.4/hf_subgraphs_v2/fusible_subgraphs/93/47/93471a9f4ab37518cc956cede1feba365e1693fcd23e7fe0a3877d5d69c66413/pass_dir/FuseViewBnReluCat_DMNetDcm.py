import torch
import triton
import triton.language as tl


# Match the full graph exactly so the matcher can bind the entire subgraph.
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    tmp_5 = conv2d.view(1, 512, 64, 64)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["TOTAL_ELEMS"],
)
@triton.jit

def cat_copy_prefix_kernel(
    in5_ptr,
    in6_ptr,
    in7_ptr,
    in8_ptr,
    out_ptr,
    C0,
    C1,
    C2,
    C3,
    H,
    W,
    in5_sc,
    in5_sh,
    in5_sw,
    in6_sc,
    in6_sh,
    in6_sw,
    in7_sc,
    in7_sh,
    in7_sw,
    in8_sc,
    in8_sh,
    in8_sw,
    out_sc,
    out_sh,
    out_sw,
    TOTAL_ELEMS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS

    hw = H * W
    out_c = offs // hw
    rem = offs - out_c * hw
    h = rem // W
    w = rem - h * W

    out_ptrs = out_ptr + out_c * out_sc + h * out_sh + w * out_sw

    r0 = mask & (out_c < C0)
    src0_ptrs = in5_ptr + out_c * in5_sc + h * in5_sh + w * in5_sw
    v0 = tl.load(src0_ptrs, mask=r0, other=0)
    tl.store(out_ptrs, v0, mask=r0)

    base1 = C0
    lim1 = C0 + C1
    r1 = mask & (out_c >= base1) & (out_c < lim1)
    c1 = out_c - base1
    src1_ptrs = in7_ptr + c1 * in7_sc + h * in7_sh + w * in7_sw
    v1 = tl.load(src1_ptrs, mask=r1, other=0)
    tl.store(out_ptrs, v1, mask=r1)

    base2 = lim1
    lim2 = lim1 + C2
    r2 = mask & (out_c >= base2) & (out_c < lim2)
    c2 = out_c - base2
    src2_ptrs = in8_ptr + c2 * in8_sc + h * in8_sh + w * in8_sw
    v2 = tl.load(src2_ptrs, mask=r2, other=0)
    tl.store(out_ptrs, v2, mask=r2)

    base3 = lim2
    lim3 = lim2 + C3
    r3 = mask & (out_c >= base3) & (out_c < lim3)
    c3 = out_c - base3
    src3_ptrs = in6_ptr + c3 * in6_sc + h * in6_sh + w * in6_sw
    v3 = tl.load(src3_ptrs, mask=r3, other=0)
    tl.store(out_ptrs, v3, mask=r3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=8, num_stages=2),
    ],
    key=["HW", "C"],
)
@triton.jit

def depthwise_conv_bn_relu_store_kernel(
    x_ptr,
    filter_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    bn_weight_ptr,
    out_ptr,
    C,
    H,
    W,
    X_H,
    X_W,
    OUT_BASE_C,
    x_sc,
    x_sh,
    x_sw,
    f_sc,
    f_sh,
    f_sw,
    out_sc,
    out_sh,
    out_sw,
    EPS,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    h = offs_hw // W
    w = offs_hw - h * W

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    bn_weight = tl.load(bn_weight_ptr + c).to(tl.float32)

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    x_c_base = x_ptr + c * x_sc
    f_c_base = filter_ptr + c * f_sc

    for kh in range(7):
        for kw in range(7):
            x_ptrs = x_c_base + (h + kh) * x_sh + (w + kw) * x_sw
            f_val = tl.load(f_c_base + kh * f_sh + kw * f_sw).to(tl.float32)
            x_val = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
            acc += x_val * f_val

    y = ((acc - mean) / tl.sqrt(var + EPS)) * bn_weight + bias
    y = tl.maximum(y, 0.0)

    out_ptrs = out_ptr + (OUT_BASE_C + c) * out_sc + h * out_sh + w * out_sw
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_dmnet_dcm_tail(
    in_0,
    in_1,
    in_2,
    in_3,
    in_4,
    in_5,
    in_6,
    in_7,
    in_8,
    in_9,
):
    h = in_6.shape[2]
    w = in_6.shape[3]
    x_h = in_9.shape[2]
    x_w = in_9.shape[3]

    c0 = in_5.shape[1]
    c1 = in_7.shape[1]
    c2 = in_8.shape[1]
    c3 = in_6.shape[1]
    c4 = in_9.shape[1]
    out_c = c0 + c1 + c2 + c3 + c4
    out_base_c = c0 + c1 + c2 + c3

    out = torch.empty((1, out_c, h, w), device=in_5.device, dtype=in_5.dtype)

    total_copy_elems = out_base_c * h * w
    copy_grid = lambda meta: (triton.cdiv(total_copy_elems, meta["BLOCK"]),)
    cat_copy_prefix_kernel[copy_grid](
        in_5,
        in_6,
        in_7,
        in_8,
        out,
        c0,
        c1,
        c2,
        c3,
        h,
        w,
        in_5.stride(1),
        in_5.stride(2),
        in_5.stride(3),
        in_6.stride(1),
        in_6.stride(2),
        in_6.stride(3),
        in_7.stride(1),
        in_7.stride(2),
        in_7.stride(3),
        in_8.stride(1),
        in_8.stride(2),
        in_8.stride(3),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        total_copy_elems,
    )

    hw = h * w
    conv_grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), c4)
    depthwise_conv_bn_relu_store_kernel[conv_grid](
        in_9,
        in_4,
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        c4,
        h,
        w,
        x_h,
        x_w,
        out_base_c,
        in_9.stride(1),
        in_9.stride(2),
        in_9.stride(3),
        in_4.stride(0),
        in_4.stride(2),
        in_4.stride(3),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        1e-05,
        hw,
    )
    return out


def replacement_func():
    return fused_dmnet_dcm_tail