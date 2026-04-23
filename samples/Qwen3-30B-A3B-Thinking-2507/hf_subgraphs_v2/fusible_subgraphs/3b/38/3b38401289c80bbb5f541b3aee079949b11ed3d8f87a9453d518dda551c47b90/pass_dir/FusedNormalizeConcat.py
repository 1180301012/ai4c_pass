import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp1 = in_1 * 0.458
    tmp2 = tmp1 + -0.030000000000000027
    tmp3 = in_0[:, 1]
    tmp4 = torch.unsqueeze(tmp3, 1)
    tmp5 = tmp4 * 0.448
    tmp6 = tmp5 + -0.08799999999999997
    tmp7 = in_0[:, 2]
    tmp8 = torch.unsqueeze(tmp7, 1)
    tmp9 = tmp8 * 0.45
    tmp10 = tmp9 + -0.18799999999999994
    tmp11 = torch.cat((tmp2, tmp6, tmp10), 1)
    return tmp11

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    batch,
    channels,
    h,
    w,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch * h * w

    batch_idx = offsets // (h * w)
    h_idx = (offsets % (h * w)) // w
    w_idx = offsets % w

    in1_offset = batch_idx * (1 * h * w) + h_idx * w + w_idx
    in1_val = tl.load(in1_ptr + in1_offset, mask=mask, other=0.0)

    in0_offset1 = batch_idx * (channels * h * w) + 1 * (h * w) + h_idx * w + w_idx
    in0_val1 = tl.load(in0_ptr + in0_offset1, mask=mask, other=0.0)

    in0_offset2 = batch_idx * (channels * h * w) + 2 * (h * w) + h_idx * w + w_idx
    in0_val2 = tl.load(in0_ptr + in0_offset2, mask=mask, other=0.0)

    out0 = in1_val * 0.458 - 0.030000000000000027
    out1 = in0_val1 * 0.448 - 0.08799999999999997
    out2 = in0_val2 * 0.45 - 0.18799999999999994

    out_offset0 = batch_idx * (3 * h * w) + 0 * (h * w) + h_idx * w + w_idx
    out_offset1 = batch_idx * (3 * h * w) + 1 * (h * w) + h_idx * w + w_idx
    out_offset2 = batch_idx * (3 * h * w) + 2 * (h * w) + h_idx * w + w_idx

    tl.store(out_ptr + out_offset0, out0, mask=mask)
    tl.store(out_ptr + out_offset1, out1, mask=mask)
    tl.store(out_ptr + out_offset2, out2, mask=mask)

@torch.fx.wrap
def fused_normalize_concat(in_0, in_1):
    batch, channels, h, w = in_0.shape
    out = torch.empty((batch, 3, h, w), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 1024
    num_blocks = (batch * h * w + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_kernel[(num_blocks,)](
        in_0,
        in_1,
        out,
        batch,
        channels,
        h,
        w,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def replacement_func():
    return fused_normalize_concat