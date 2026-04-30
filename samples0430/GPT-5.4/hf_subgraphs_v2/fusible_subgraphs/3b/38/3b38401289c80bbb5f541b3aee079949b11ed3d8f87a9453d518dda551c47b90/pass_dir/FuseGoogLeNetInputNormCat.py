import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_input_norm_cat_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    hw,
    h_size,
    w_size,
    in0_c,
    BLOCK_W: tl.constexpr,
):
    nh = tl.program_id(0)
    n_idx = nh // h_size
    h_idx = nh % h_size

    offs_w = tl.arange(0, BLOCK_W)
    mask = offs_w < w_size
    row_off = h_idx * w_size + offs_w

    in1_off = n_idx * hw + row_off
    in0_base = n_idx * in0_c * hw + row_off
    out_base = n_idx * 3 * hw + row_off

    x0 = tl.load(in1_ptr + in1_off, mask=mask, other=0).to(tl.float32)
    x1 = tl.load(in0_ptr + in0_base + hw, mask=mask, other=0).to(tl.float32)
    x2 = tl.load(in0_ptr + in0_base + 2 * hw, mask=mask, other=0).to(tl.float32)

    tl.store(out_ptr + out_base, x0 * 0.458 + (-0.030000000000000027), mask=mask)
    tl.store(out_ptr + out_base + hw, x1 * 0.448 + (-0.08799999999999997), mask=mask)
    tl.store(out_ptr + out_base + 2 * hw, x2 * 0.45 + (-0.18799999999999994), mask=mask)


@torch.fx.wrap
def fused_input_norm_cat(in_0, in_1):
    n = in_1.shape[0]
    h = in_1.shape[2]
    w = in_1.shape[3]
    hw = h * w

    out = torch.empty((n, 3, h, w), device=in_0.device, dtype=in_0.dtype)

    if w <= 32:
        fused_input_norm_cat_kernel[(n * h,)](
            in_0,
            in_1,
            out,
            hw,
            h,
            w,
            in_0.shape[1],
            BLOCK_W=32,
            num_warps=1,
            num_stages=1,
        )
    else:
        fused_input_norm_cat_kernel[(n * h,)](
            in_0,
            in_1,
            out,
            hw,
            h,
            w,
            in_0.shape[1],
            BLOCK_W=256,
            num_warps=4,
            num_stages=2,
        )
    return out


def replacement_func():
    return fused_input_norm_cat