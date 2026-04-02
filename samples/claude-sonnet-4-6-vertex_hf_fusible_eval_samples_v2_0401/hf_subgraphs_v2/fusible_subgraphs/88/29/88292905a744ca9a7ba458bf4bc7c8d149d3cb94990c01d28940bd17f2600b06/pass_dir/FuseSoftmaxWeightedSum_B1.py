import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, 256)
    tmp_2 = tmp_1.view(1, 256, 1, 1)
    tmp_3 = tmp_2.view(1, 2, 128, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _fused_softmax_wsum_b1(
    in0_ptr,   # [B, 2, C, H, W]
    in1_ptr,   # [B, 2, 1, C]
    out_ptr,   # [B, C, H, W]
    B, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C
    c = pid_bc % C

    base_in1 = b * 2 * C + c
    x0 = tl.load(in1_ptr + base_in1).to(tl.float32)
    x1 = tl.load(in1_ptr + base_in1 + C).to(tl.float32)

    m  = tl.maximum(x0, x1)
    e0 = tl.exp(x0 - m)
    e1 = tl.exp(x1 - m)
    inv_s = 1.0 / (e0 + e1)
    w0 = e0 * inv_s
    w1 = e1 * inv_s

    hw_off = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = hw_off < HW

    base_in0_k0 = b * 2 * C * HW + c * HW
    base_in0_k1 = base_in0_k0 + C * HW

    v0 = tl.load(in0_ptr + base_in0_k0 + hw_off, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in0_ptr + base_in0_k1 + hw_off, mask=mask, other=0.0).to(tl.float32)

    result = w0 * v0 + w1 * v1
    tl.store(out_ptr + b * C * HW + c * HW + hw_off, result, mask=mask)


@torch.fx.wrap
def triton_softmax_weighted_sum_b1(in_0, in_1):
    B  = in_0.shape[0]
    C  = in_0.shape[2]
    H  = in_0.shape[3]
    W  = in_0.shape[4]
    HW = H * W

    out = torch.empty(B, C, H, W, dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 1024
    grid = (B * C, triton.cdiv(HW, BLOCK_SIZE))
    _fused_softmax_wsum_b1[grid](in_0, in_1, out, B, C, HW, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_softmax_weighted_sum_b1