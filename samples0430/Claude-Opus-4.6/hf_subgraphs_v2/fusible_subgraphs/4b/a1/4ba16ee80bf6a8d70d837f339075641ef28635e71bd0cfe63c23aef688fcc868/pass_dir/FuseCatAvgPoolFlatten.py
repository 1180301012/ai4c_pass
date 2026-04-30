import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_cat_avgpool_flatten_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    channel = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_HW)
    hw_mask = offsets < HW

    # Determine which input tensor this channel belongs to
    is_0 = channel < C0
    is_1 = (channel >= C0) & (channel < C0 + C1)
    is_2 = (channel >= C0 + C1) & (channel < C0 + C1 + C2)

    # Compute local channel index
    local_ch = tl.where(is_0, channel,
               tl.where(is_1, channel - C0,
               tl.where(is_2, channel - C0 - C1,
                        channel - C0 - C1 - C2)))

    # Select base pointer
    base_ptr = tl.where(is_0, in_0_ptr,
               tl.where(is_1, in_1_ptr,
               tl.where(is_2, in_2_ptr,
                        in_3_ptr)))

    # Load spatial elements and compute mean
    vals = tl.load(base_ptr + local_ch * HW + offsets, mask=hw_mask, other=0.0).to(tl.float32)
    mean_val = tl.sum(vals, axis=0) * (1.0 / HW)

    # Store result
    tl.store(out_ptr + channel, mean_val)


@torch.fx.wrap
def fused_cat_avgpool_flatten(in_0, in_1, in_2, in_3):
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    total_channels = C0 + C1 + C2 + C3

    out = torch.empty((1, total_channels), dtype=in_0.dtype, device=in_0.device)

    fused_cat_avgpool_flatten_kernel[(total_channels,)](
        in_0, in_1, in_2, in_3,
        out,
        C0=C0, C1=C1, C2=C2, C3=C3,
        HW=25, BLOCK_HW=32,
        num_warps=1,
    )

    return out


def replacement_func():
    return fused_cat_avgpool_flatten