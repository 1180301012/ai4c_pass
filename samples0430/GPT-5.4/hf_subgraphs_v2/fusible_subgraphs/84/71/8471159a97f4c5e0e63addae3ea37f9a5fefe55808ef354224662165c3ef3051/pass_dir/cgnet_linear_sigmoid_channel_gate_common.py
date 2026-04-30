import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 2048}, num_warps=8, num_stages=2),
    ],
    key=["hw"],
)
@triton.jit
def _cgnet_linear_sigmoid_channel_gate_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    feat_ptr,
    out_ptr,
    hw,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bc = tl.program_id(1)

    n = pid_bc // 64
    c = pid_bc % 64

    k_offsets = tl.arange(0, 8)
    x_vals = tl.load(x_ptr + n * 8 + k_offsets).to(tl.float32)
    w_vals = tl.load(weight_ptr + c * 8 + k_offsets).to(tl.float32)
    bias_val = tl.load(bias_ptr + c).to(tl.float32)

    gate = tl.sum(x_vals * w_vals, axis=0) + bias_val
    gate = 1.0 / (1.0 + tl.exp(-gate))

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < hw
    base = (n * 64 + c) * hw
    feat_vals = tl.load(feat_ptr + base + hw_offsets, mask=mask, other=0.0)
    out_vals = feat_vals * gate
    tl.store(out_ptr + base + hw_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_linear_sigmoid_channel_gate(bias, weight, x, feat):
    out = torch.empty_like(feat)
    hw = feat.shape[2] * feat.shape[3]
    grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), feat.shape[0] * 64)
    _cgnet_linear_sigmoid_channel_gate_kernel[grid](
        bias,
        weight,
        x,
        feat,
        out,
        hw,
    )
    return out