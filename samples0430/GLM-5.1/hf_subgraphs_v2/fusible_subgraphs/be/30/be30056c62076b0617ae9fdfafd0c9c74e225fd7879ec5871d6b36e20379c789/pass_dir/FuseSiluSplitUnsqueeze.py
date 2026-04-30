import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[None, None, slice(None, None, None)]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def silu_split_kernel(
    in_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    total_elements,
    H: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < total_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    silu_val = x * tl.sigmoid(x)

    w = offsets % W
    bh = offsets // W

    mask0 = (w < S0) & mask
    mask1 = (w >= S0) & (w < S0 + S1) & mask
    mask2 = (w >= S0 + S1) & mask

    out0_off = bh * S0 + w
    out1_off = bh * S1 + (w - S0)
    out2_off = bh * S2 + (w - S0 - S1)

    tl.store(out0_ptr + out0_off, silu_val, mask=mask0)
    tl.store(out1_ptr + out1_off, silu_val, mask=mask1)
    tl.store(out2_ptr + out2_off, silu_val, mask=mask2)


@torch.fx.wrap
def fused_silu_split_unsqueeze(in_0, in_1):
    B = in_1.shape[0]
    H_val = in_1.shape[1]
    W_val = in_1.shape[2]

    total_elements = in_1.numel()

    # Allocate output tensors directly in their final shapes
    # out0: [B, H, 512] - first split
    # out1: [B, H, 512] - second split
    # out2: [B, H, 1, 128] - third split, already unsqueezed
    out0 = torch.empty((B, H_val, 512), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((B, H_val, 512), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((B, H_val, 1, 128), dtype=in_1.dtype, device=in_1.device)

    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    silu_split_kernel[(num_programs,)](
        in_ptr=in_1,
        out0_ptr=out0,
        out1_ptr=out1,
        out2_ptr=out2,
        total_elements=total_elements,
        H=H_val,
        W=W_val,
        S0=512,
        S1=512,
        S2=128,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # in_0 view: [2, 128] -> [1, 1, 2, 128]
    # This is a metadata-only operation, handled via as_tensor + reshape
    in_0_view = torch.as_tensor(in_0).reshape(1, 1, in_0.shape[0], in_0.shape[1])

    # Return in same order as pattern: (tmp_7, tmp_3, tmp_6, tmp_4)
    return (in_0_view, out0, out2, out1)


def replacement_func():
    return fused_silu_split_unsqueeze