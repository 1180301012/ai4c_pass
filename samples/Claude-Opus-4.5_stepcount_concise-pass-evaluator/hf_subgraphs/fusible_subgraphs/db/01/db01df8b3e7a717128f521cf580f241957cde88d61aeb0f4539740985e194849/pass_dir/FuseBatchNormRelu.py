import torch
import triton
import triton.language as tl


@triton.jit
def cat_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, in4_ptr,
    out_ptr,
    C0, C1, C2, C3, C4, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total_elements = (C0 + C1 + C2 + C3 + C4) * HW
    mask = offsets < total_elements
    
    # Calculate which input tensor and position within it
    # Total channels = C0 + C1 + C2 + C3 + C4
    # For each element in output, determine source
    hw_idx = offsets % HW
    c_idx = offsets // HW
    
    # Determine which tensor to read from based on channel index
    # in_5: channels [0, C0)
    # in_7: channels [C0, C0+C1)
    # in_8: channels [C0+C1, C0+C1+C2)
    # in_6: channels [C0+C1+C2, C0+C1+C2+C3)
    # tmp_7: channels [C0+C1+C2+C3, C0+C1+C2+C3+C4)
    
    # Default value
    val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load from appropriate tensor
    # in_5 (C0 channels)
    c0_mask = c_idx < C0
    in0_idx = c_idx * HW + hw_idx
    val = tl.where(c0_mask & mask, tl.load(in0_ptr + in0_idx, mask=c0_mask & mask, other=0.0), val)
    
    # in_7 (C1 channels)
    c1_start = C0
    c1_mask = (c_idx >= c1_start) & (c_idx < c1_start + C1)
    in1_idx = (c_idx - c1_start) * HW + hw_idx
    val = tl.where(c1_mask & mask, tl.load(in1_ptr + in1_idx, mask=c1_mask & mask, other=0.0), val)
    
    # in_8 (C2 channels)
    c2_start = C0 + C1
    c2_mask = (c_idx >= c2_start) & (c_idx < c2_start + C2)
    in2_idx = (c_idx - c2_start) * HW + hw_idx
    val = tl.where(c2_mask & mask, tl.load(in2_ptr + in2_idx, mask=c2_mask & mask, other=0.0), val)
    
    # in_6 (C3 channels)
    c3_start = C0 + C1 + C2
    c3_mask = (c_idx >= c3_start) & (c_idx < c3_start + C3)
    in3_idx = (c_idx - c3_start) * HW + hw_idx
    val = tl.where(c3_mask & mask, tl.load(in3_ptr + in3_idx, mask=c3_mask & mask, other=0.0), val)
    
    # tmp_7 (C4 channels)
    c4_start = C0 + C1 + C2 + C3
    c4_mask = (c_idx >= c4_start) & (c_idx < c4_start + C4)
    in4_idx = (c_idx - c4_start) * HW + hw_idx
    val = tl.where(c4_mask & mask, tl.load(in4_ptr + in4_idx, mask=c4_mask & mask, other=0.0), val)
    
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_cat(in_5, in_7, in_8, in_6, tmp_7):
    # Get shapes
    N, C0, H, W = in_5.shape
    C1 = in_7.shape[1]
    C2 = in_8.shape[1]
    C3 = in_6.shape[1]
    C4 = tmp_7.shape[1]
    
    HW = H * W
    total_channels = C0 + C1 + C2 + C3 + C4
    total_elements = total_channels * HW
    
    out = torch.empty((N, total_channels, H, W), dtype=in_5.dtype, device=in_5.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cat_kernel[(num_programs,)](
        in0_ptr=in_5,
        in1_ptr=in_7,
        in2_ptr=in_8,
        in3_ptr=in_6,
        in4_ptr=tmp_7,
        out_ptr=out,
        C0=C0, C1=C1, C2=C2, C3=C3, C4=C4, HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_5, in_7, in_8, in_6, tmp_7):
    result = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return result


def replacement_args(in_5, in_7, in_8, in_6, tmp_7):
    return (in_5, in_7, in_8, in_6, tmp_7)


def replacement_func():
    return triton_cat