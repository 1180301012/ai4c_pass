import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


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
def _fused_channel_normalize_cat_kernel_256(
    in0_ptr,
    in1_ptr,
    out_ptr,
    in0_c,
):
    pid_n = tl.program_id(0)
    hw = 256
    offs = tl.arange(0, 256)

    batch_in0_base = pid_n * in0_c * hw
    batch_in1_base = pid_n * hw
    batch_out_base = pid_n * 3 * hw

    x0 = tl.load(in1_ptr + batch_in1_base + offs).to(tl.float32)
    x1 = tl.load(in0_ptr + batch_in0_base + hw + offs).to(tl.float32)
    x2 = tl.load(in0_ptr + batch_in0_base + 2 * hw + offs).to(tl.float32)

    y0 = x0 * 0.458 + (-0.030000000000000027)
    y1 = x1 * 0.448 + (-0.08799999999999997)
    y2 = x2 * 0.45 + (-0.18799999999999994)

    tl.store(out_ptr + batch_out_base + offs, y0)
    tl.store(out_ptr + batch_out_base + hw + offs, y1)
    tl.store(out_ptr + batch_out_base + 2 * hw + offs, y2)


@triton.jit
def _fused_channel_normalize_cat_kernel_1024(
    in0_ptr,
    in1_ptr,
    out_ptr,
    in0_c,
):
    pid_n = tl.program_id(0)
    hw = 1024
    offs = tl.arange(0, 1024)

    batch_in0_base = pid_n * in0_c * hw
    batch_in1_base = pid_n * hw
    batch_out_base = pid_n * 3 * hw

    x0 = tl.load(in1_ptr + batch_in1_base + offs).to(tl.float32)
    x1 = tl.load(in0_ptr + batch_in0_base + hw + offs).to(tl.float32)
    x2 = tl.load(in0_ptr + batch_in0_base + 2 * hw + offs).to(tl.float32)

    y0 = x0 * 0.458 + (-0.030000000000000027)
    y1 = x1 * 0.448 + (-0.08799999999999997)
    y2 = x2 * 0.45 + (-0.18799999999999994)

    tl.store(out_ptr + batch_out_base + offs, y0)
    tl.store(out_ptr + batch_out_base + hw + offs, y1)
    tl.store(out_ptr + batch_out_base + 2 * hw + offs, y2)


@triton.jit
def _fused_channel_normalize_cat_kernel_large(
    in0_ptr,
    in1_ptr,
    out_ptr,
    hw,
    in0_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < hw

    batch_in0_base = pid_n * in0_c * hw
    batch_in1_base = pid_n * hw
    batch_out_base = pid_n * 3 * hw

    x0 = tl.load(in1_ptr + batch_in1_base + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in0_ptr + batch_in0_base + hw + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in0_ptr + batch_in0_base + 2 * hw + offs, mask=mask, other=0.0).to(tl.float32)

    y0 = x0 * 0.458 + (-0.030000000000000027)
    y1 = x1 * 0.448 + (-0.08799999999999997)
    y2 = x2 * 0.45 + (-0.18799999999999994)

    tl.store(out_ptr + batch_out_base + offs, y0, mask=mask)
    tl.store(out_ptr + batch_out_base + hw + offs, y1, mask=mask)
    tl.store(out_ptr + batch_out_base + 2 * hw + offs, y2, mask=mask)


@torch.fx.wrap
def fused_channel_normalize_cat(in_0, in_1):
    in_0 = unwrap_tensor(in_0)
    in_1 = unwrap_tensor(in_1)

    n = in_1.shape[0]
    h = in_1.shape[2]
    w = in_1.shape[3]
    hw = h * w
    in0_c = in_0.shape[1]

    out = torch.empty((n, 3, h, w), device=in_1.device, dtype=in_1.dtype)

    if hw == 256:
        _fused_channel_normalize_cat_kernel_256[(n,)](
            in_0,
            in_1,
            out,
            in0_c,
        )
        return out

    if hw == 1024:
        _fused_channel_normalize_cat_kernel_1024[(n,)](
            in_0,
            in_1,
            out,
            in0_c,
        )
        return out

    _fused_channel_normalize_cat_kernel_large[(triton.cdiv(hw, 256), n)](
        in_0,
        in_1,
        out,
        hw,
        in0_c,
        BLOCK_SIZE=256,
    )
    return out


def replacement_func():
    return fused_channel_normalize_cat