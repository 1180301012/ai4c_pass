import torch
import triton
import triton.language as tl


def pattern(in_0, weights):
    # weights: [B, 2, C, 1, 1] — pre-computed attention weights (softmax already applied)
    # in_0:    [B, 2, C, H, W]
    tmp_4 = weights * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, weights):
    return (in_0, weights)


@triton.jit
def _fused_wsum_kernel(
    in0_ptr,      # [B, 2, C, H, W]
    weights_ptr,  # [B, 2, C, 1, 1]  stride (2C, C, 1, 1, 1)
    out_ptr,      # [B, C, H, W]
    B, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C
    c = pid_bc % C

    # weights[b, k, c, 0, 0]  stride=(2C, C, 1, 1, 1)
    w0 = tl.load(weights_ptr + b * 2 * C + c).to(tl.float32)
    w1 = tl.load(weights_ptr + b * 2 * C + C + c).to(tl.float32)

    hw_off = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = hw_off < HW

    # in_0[b, k, c, hw]  stride=(2*C*HW, C*HW, HW, W, 1)
    base0 = b * 2 * C * HW + c * HW
    base1 = base0 + C * HW

    v0 = tl.load(in0_ptr + base0 + hw_off, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in0_ptr + base1 + hw_off, mask=mask, other=0.0).to(tl.float32)

    result = w0 * v0 + w1 * v1
    tl.store(out_ptr + b * C * HW + c * HW + hw_off, result, mask=mask)


@torch.fx.wrap
def triton_weighted_sum(in_0, weights):
    B  = in_0.shape[0]
    C  = in_0.shape[2]
    H  = in_0.shape[3]
    W  = in_0.shape[4]
    HW = H * W

    # For small tensors (B*HW <= threshold), fall back to PyTorch ops
    # which are faster due to lower kernel-launch overhead
    if B * HW <= 8192:
        return (weights * in_0).sum(dim=1).contiguous()

    out = torch.empty(B, C, H, W, dtype=in_0.dtype, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = (B * C, triton.cdiv(HW, BLOCK_SIZE))
    _fused_wsum_kernel[grid](in_0, weights, out, B, C, HW, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_weighted_sum