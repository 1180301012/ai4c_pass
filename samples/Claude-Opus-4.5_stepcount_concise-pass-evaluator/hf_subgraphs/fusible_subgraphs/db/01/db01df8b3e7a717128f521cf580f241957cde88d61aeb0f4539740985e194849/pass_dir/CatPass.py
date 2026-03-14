import torch
import triton
import triton.language as tl


@triton.jit
def cat5_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, in4_ptr,
    out_ptr,
    n0, n1, n2, n3, n4,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = n0 + n1 + n2 + n3 + n4
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    # Determine which tensor this belongs to and load accordingly
    # Use conditional loads to avoid branching
    
    # Check if in tensor 0
    in0_mask = offsets < n0
    val = tl.load(in0_ptr + offsets, mask=in0_mask & mask, other=0.0)
    
    # Check if in tensor 1
    off1 = offsets - n0
    in1_mask = (offsets >= n0) & (offsets < n0 + n1)
    val = tl.where(in1_mask, tl.load(in1_ptr + off1, mask=in1_mask & mask, other=0.0), val)
    
    # Check if in tensor 2
    off2 = offsets - n0 - n1
    in2_mask = (offsets >= n0 + n1) & (offsets < n0 + n1 + n2)
    val = tl.where(in2_mask, tl.load(in2_ptr + off2, mask=in2_mask & mask, other=0.0), val)
    
    # Check if in tensor 3
    off3 = offsets - n0 - n1 - n2
    in3_mask = (offsets >= n0 + n1 + n2) & (offsets < n0 + n1 + n2 + n3)
    val = tl.where(in3_mask, tl.load(in3_ptr + off3, mask=in3_mask & mask, other=0.0), val)
    
    # Check if in tensor 4
    off4 = offsets - n0 - n1 - n2 - n3
    in4_mask = offsets >= n0 + n1 + n2 + n3
    val = tl.where(in4_mask, tl.load(in4_ptr + off4, mask=in4_mask & mask, other=0.0), val)
    
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_cat(in_5, in_7, in_8, in_6, tmp_7):
    N, C0, H, W = in_5.shape
    C1 = in_7.shape[1]
    C2 = in_8.shape[1]
    C3 = in_6.shape[1]
    C4 = tmp_7.shape[1]
    
    HW = H * W
    n0 = C0 * HW
    n1 = C1 * HW
    n2 = C2 * HW
    n3 = C3 * HW
    n4 = C4 * HW
    total = n0 + n1 + n2 + n3 + n4
    
    total_channels = C0 + C1 + C2 + C3 + C4
    out = torch.empty((N, total_channels, H, W), dtype=in_5.dtype, device=in_5.device)
    
    BLOCK_SIZE = 2048
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cat5_kernel[(num_blocks,)](
        in0_ptr=in_5,
        in1_ptr=in_7,
        in2_ptr=in_8,
        in3_ptr=in_6,
        in4_ptr=tmp_7,
        out_ptr=out,
        n0=n0, n1=n1, n2=n2, n3=n3, n4=n4,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    return out


def pattern(in_5, in_7, in_8, in_6, tmp_7):
    result = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return result


def replacement_args(in_5, in_7, in_8, in_6, tmp_7):
    return (in_5, in_7, in_8, in_6, tmp_7)


def replacement_func():
    return triton_cat