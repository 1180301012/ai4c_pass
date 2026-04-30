import torch
import triton
import triton.language as tl


def pattern(tmp_1, tmp_2, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return tmp_3


def replacement_args(tmp_1, tmp_2, in_2, in_3):
    return (tmp_1, tmp_2, in_2, in_3)


@triton.jit
def fused_cat_stack_kernel(
    tmp_1_ptr, tmp_2_ptr, in_2_ptr, in_3_ptr, out_ptr,
    slice_size,
    C_half_HW,  # C_half * H * W
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Output is [3, B, C, H, W] contiguous, total = 3 * slice_size
    # slice_size = B * C * H * W where C = 2 * C_half
    # For s=0: out[0:slice_size] = tmp_1 (contiguous copy)
    # For s=1: out[slice_size:2*slice_size] = tmp_2 (contiguous copy)
    # For s=2: out[2*slice_size:3*slice_size] = cat(in_2, in_3, dim=1)

    # Slice 0: direct copy from tmp_1
    mask_s0 = mask & (offsets < slice_size)
    val_s0 = tl.load(tmp_1_ptr + offsets, mask=mask_s0, other=0.0)

    # Slice 1: direct copy from tmp_2
    off_s1 = offsets - slice_size
    mask_s1 = mask & (offsets >= slice_size) & (offsets < 2 * slice_size)
    val_s1 = tl.load(tmp_2_ptr + off_s1, mask=mask_s1, other=0.0)

    # Slice 2: cat(in_2, in_3) along channel dim
    # Layout: [B, C, H, W] where C = 2*C_half
    # Element at local offset `off_s2` maps to (b, c, h, w)
    # in_batch = off_s2 % (2*C_half_HW), batch = off_s2 // (2*C_half_HW)
    # if in_batch < C_half_HW: read from in_2 at batch*C_half_HW + in_batch
    # else: read from in_3 at batch*C_half_HW + (in_batch - C_half_HW)
    off_s2 = offsets - 2 * slice_size
    mask_s2 = mask & (offsets >= 2 * slice_size)
    
    two_C_half_HW = 2 * C_half_HW
    in_batch = off_s2 % two_C_half_HW
    batch_base = off_s2 - in_batch  # = batch * 2*C_half_HW
    # batch_base // 2 gives batch * C_half_HW
    is_lo = in_batch < C_half_HW
    
    addr_in2 = (batch_base // 2) + in_batch
    addr_in3 = (batch_base // 2) + (in_batch - C_half_HW)
    
    mask_s2_lo = mask_s2 & is_lo
    mask_s2_hi = mask_s2 & ~is_lo
    val_s2_lo = tl.load(in_2_ptr + addr_in2, mask=mask_s2_lo, other=0.0)
    val_s2_hi = tl.load(in_3_ptr + addr_in3, mask=mask_s2_hi, other=0.0)

    val = val_s0 + val_s1 + val_s2_lo + val_s2_hi
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_cat_stack(tmp_1, tmp_2, in_2, in_3):
    B = tmp_1.shape[0]
    C = tmp_1.shape[1]
    C_half = in_2.shape[1]
    H = tmp_1.shape[2]
    W = tmp_1.shape[3]

    out = torch.empty(3, B, C, H, W, dtype=tmp_1.dtype, device=tmp_1.device)
    
    slice_size = B * C * H * W
    C_half_HW = C_half * H * W
    n_elements = 3 * slice_size
    
    BLOCK_SIZE = 4096
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_cat_stack_kernel[grid](
        tmp_1, tmp_2, in_2, in_3, out,
        slice_size,
        C_half_HW,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_cat_stack