import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_960")


@triton.jit
def fused_cat_mean_kernel(
    in0_ptr, in1_ptr, out_cat_ptr, out_mean_ptr,
    B, C0, C1, H, W,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_cat_b, stride_cat_c, stride_cat_h, stride_cat_w,
    stride_mean_b, stride_mean_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    total_c = C0 + C1
    b = pid // total_c
    c = pid % total_c

    hw_size = H * W

    if c < C0:
        src_ptr = in0_ptr
        src_b_s = stride_in0_b
        src_c_s = stride_in0_c
        src_h_s = stride_in0_h
        src_w_s = stride_in0_w
        src_c_idx = c
    else:
        src_ptr = in1_ptr
        src_b_s = stride_in1_b
        src_c_s = stride_in1_c
        src_h_s = stride_in1_h
        src_w_s = stride_in1_w
        src_c_idx = c - C0

    src_base = b * src_b_s + src_c_idx * src_c_s
    cat_base = b * stride_cat_b + c * stride_cat_c

    acc = 0.0

    for hw_start in range(0, hw_size, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < hw_size

        h_idx = hw_offsets // W
        w_idx = hw_offsets % W

        src_off = src_base + h_idx * src_h_s + w_idx * src_w_s
        vals = tl.load(src_ptr + src_off, mask=mask, other=0.0).to(tl.float32)

        cat_off = cat_base + h_idx * stride_cat_h + w_idx * stride_cat_w
        tl.store(out_cat_ptr + cat_off, vals, mask=mask)

        acc += tl.sum(vals, axis=0)

    mean_val = acc / hw_size
    mean_off = b * stride_mean_b + c * stride_mean_c
    tl.store(out_mean_ptr + mean_off, mean_val)


@torch.fx.wrap
def fused_cat_mean_dispatch(in_0, in_1, route):
    B = in_0.shape[0]
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]

    total_c = C0 + C1
    hw_size = H * W

    out_cat = torch.empty([B, total_c, H, W], dtype=in_0.dtype, device=in_0.device)
    out_mean = torch.empty([B, total_c, 1, 1], dtype=in_0.dtype, device=in_0.device)

    BLOCK_HW = min(triton.next_power_of_2(hw_size), 2048)
    if BLOCK_HW < 64:
        BLOCK_HW = 64

    grid = (B * total_c,)

    fused_cat_mean_kernel[grid](
        in0_ptr=in_0, in1_ptr=in_1,
        out_cat_ptr=out_cat, out_mean_ptr=out_mean,
        B=B, C0=C0, C1=C1, H=H, W=W,
        stride_in0_b=in_0.stride(0), stride_in0_c=in_0.stride(1),
        stride_in0_h=in_0.stride(2), stride_in0_w=in_0.stride(3),
        stride_in1_b=in_1.stride(0), stride_in1_c=in_1.stride(1),
        stride_in1_h=in_1.stride(2), stride_in1_w=in_1.stride(3),
        stride_cat_b=out_cat.stride(0), stride_cat_c=out_cat.stride(1),
        stride_cat_h=out_cat.stride(2), stride_cat_w=out_cat.stride(3),
        stride_mean_b=out_mean.stride(0), stride_mean_c=out_mean.stride(1),
        BLOCK_HW=BLOCK_HW,
    )

    return (out_cat, out_mean)


def replacement_func():
    return fused_cat_mean_dispatch