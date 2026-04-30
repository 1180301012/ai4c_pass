import torch
import triton
import triton.language as tl


@triton.jit
def _fused_se_kernel(
    in_3_ptr, weight_ptr, bias_ptr, in_2_ptr, out_ptr,
    C_in, C_out, HW,
    add_const, div_const,
    BLOCK_HW: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c_out = tl.program_id(1)
    b = tl.program_id(2)

    # Compute dot product: conv1x1 on 1x1 spatial
    cin_offsets = tl.arange(0, BLOCK_CIN)
    cin_mask = cin_offsets < C_in

    x = tl.load(in_3_ptr + b * C_in + cin_offsets, mask=cin_mask, other=0.0)
    w = tl.load(weight_ptr + c_out * C_in + cin_offsets, mask=cin_mask, other=0.0)

    acc = tl.sum(x.to(tl.float32) * w.to(tl.float32))

    bias_val = tl.load(bias_ptr + c_out)
    scale = acc + bias_val.to(tl.float32)

    # Hard sigmoid activation
    scale = (scale + add_const) / div_const
    scale = tl.maximum(scale, 0.0)
    scale = tl.minimum(scale, 1.0)

    # Broadcast multiply
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW

    base_offset = (b * C_out + c_out) * HW
    in2_vals = tl.load(in_2_ptr + base_offset + hw_offsets, mask=hw_mask, other=0.0)

    out_vals = in2_vals * scale.to(in2_vals.dtype)
    tl.store(out_ptr + base_offset + hw_offsets, out_vals, mask=hw_mask)


@torch.fx.wrap
def fused_se_block(in_0, in_1, in_2, in_3, add_const, div_const):
    B = in_3.shape[0]
    C_in = in_3.shape[1]
    C_out = in_1.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    HW = H * W

    out = torch.empty_like(in_2)

    BLOCK_CIN = triton.next_power_of_2(C_in)

    BLOCK_HW = 1024
    grid_hw = (HW + BLOCK_HW - 1) // BLOCK_HW

    _fused_se_kernel[(grid_hw, C_out, B)](
        in_3, in_1, in_0, in_2, out,
        C_in, C_out, HW,
        add_const, div_const,
        BLOCK_HW=BLOCK_HW,
        BLOCK_CIN=BLOCK_CIN,
        num_warps=4,
    )

    return out