import torch
import triton
import triton.language as tl


def pattern(conv_out, in_2):
    tmp_3 = torch.sigmoid(conv_out)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 512, "BLOCK_C": 8}, num_warps=8, num_stages=2),
    ],
    key=["HW", "C"],
)
@triton.jit
def fused_sigmoid_broadcast_mul_contiguous_kernel(
    conv_ptr,
    x_ptr,
    out_ptr,
    HW,
    C,
    x_stride_c,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_hw = offs_hw < HW
    mask_c = offs_c < C
    mask = mask_c[:, None] & mask_hw[None, :]

    gates = tl.load(conv_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    gates = tl.sigmoid(gates)

    x_offsets = offs_c[:, None] * x_stride_c + offs_hw[None, :]
    out_offsets = offs_c[:, None] * HW + offs_hw[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    y = x.to(tl.float32) * gates[:, None]
    tl.store(out_ptr + out_offsets, y, mask=mask)


@torch.fx.wrap
def fused_sigmoid_broadcast_mul_contiguous(conv_out, in_2):
    hw = in_2.shape[2] * in_2.shape[3]
    c = in_2.shape[1]
    out = torch.empty_like(in_2)

    grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), triton.cdiv(c, meta["BLOCK_C"]))
    fused_sigmoid_broadcast_mul_contiguous_kernel[grid](
        conv_ptr=conv_out,
        x_ptr=in_2,
        out_ptr=out,
        HW=hw,
        C=c,
        x_stride_c=in_2.stride(1),
    )
    return out


def replacement_func():
    return fused_sigmoid_broadcast_mul_contiguous