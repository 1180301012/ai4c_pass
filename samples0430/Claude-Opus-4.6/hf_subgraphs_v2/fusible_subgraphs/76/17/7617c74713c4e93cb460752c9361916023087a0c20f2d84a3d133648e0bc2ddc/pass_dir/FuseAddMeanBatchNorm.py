import torch
import triton
import triton.language as tl


def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_4, in_5):
    return (in_4, in_5)


@triton.jit
def fused_add_mean_kernel(
    in_4_ptr, in_5_ptr,
    out_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C

    base_offset = batch_idx * C * HW + channel_idx * HW
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    # Load both input tensors and add them
    x = tl.load(in_4_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(in_5_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute sum and mean
    sum_val = tl.sum(x + y, axis=0)
    mean_val = sum_val / HW

    # Store result
    out_idx = batch_idx * C + channel_idx
    tl.store(out_ptr + out_idx, mean_val)


@torch.fx.wrap
def fused_add_mean(in_4, in_5):
    B = in_4.shape[0]
    C = in_4.shape[1]
    H = in_4.shape[2]
    W = in_4.shape[3]
    HW = H * W

    out = torch.empty((B, C), dtype=in_4.dtype, device=in_4.device)

    grid = (B * C,)
    fused_add_mean_kernel[grid](
        in_4, in_5,
        out,
        C, HW,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_add_mean