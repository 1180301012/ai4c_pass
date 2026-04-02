import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lsm_kernel_b128_diag(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    hw_block = tl.program_id(1)
    b_idx = bc_idx // C
    c_idx = bc_idx % C
    k_range = tl.arange(0, 8)
    w = tl.load(in1_ptr + c_idx * 8 + k_range).to(tl.float32)
    x = tl.load(in2_ptr + b_idx * 8 + k_range).to(tl.float32)
    bias = tl.load(in0_ptr + c_idx).to(tl.float32)
    scale = tl.sigmoid(tl.sum(w * x, axis=0) + bias)
    hw_start = hw_block * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW
    base = bc_idx * HW
    feat = tl.load(in3_ptr + base + hw_offsets, mask=mask, other=0.0)
    out = feat * scale.to(feat.dtype)
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def _wrapper_lsm_b128_diag(in_0, in_1, in_2, in_3):
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    in_2 = in_2.contiguous()
    in_3 = in_3.contiguous()
    B, C, H, W = in_3.shape
    HW = H * W
    BC = B * C
    out = torch.empty_like(in_3)
    grid = (BC, triton.cdiv(HW, 512))
    _fused_lsm_kernel_b128_diag[grid](in_0, in_1, in_2, in_3, out, C, HW, BLOCK_HW=512)
    return out


def pattern(in_0, in_1, in_2, in_3):
    # DIAGNOSTIC: same view(1,64,1,1) as pass 1 to test if loading works
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _wrapper_lsm_b128_diag