import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Fuse add + split for Graph 1 (197 -> 1 + 196)
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    return tmp_2, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_split_kernel_14(
    in_0_ptr, in_1_ptr,
    out_0_ptr, out_1_ptr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for add + split (14x14 case: 196 = 14*14)
    """
    pid = tl.program_id(0)
    
    # out_0: [1, 1, 384] - first element of split
    # Only program 0 computes out_0
    if pid == 0:
        offsets_c = tl.arange(0, C)
        in_0_val = tl.load(in_0_ptr + offsets_c)
        in_1_val = tl.load(in_1_ptr + offsets_c)
        out_0_val = in_0_val + in_1_val
        tl.store(out_0_ptr + offsets_c, out_0_val)
    
    # out_1: [1, 196, 384] - second part of split
    num_elements_out1 = 196 * C  # 75264
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements_out1
    
    # Compute indices for out_1
    # out_1 is [196, 384], flattened as [196 * 384]
    # We need to compute: output[196*384] = input[1:197, :]
    dim1 = offsets // C  # 0 to 195
    c = offsets % C  # 0 to 383
    
    # Input index: we start from index 1 (skip first element)
    in_idx = dim1 + 1
    
    in_0_offset = in_idx * C + c
    in_1_offset = in_idx * C + c
    
    in_0_vals = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    out_1_vals = in_0_vals + in_1_vals
    
    tl.store(out_1_ptr + offsets, out_1_vals, mask=mask)


@torch.fx.wrap
def fused_add_split_14(in_0, in_1):
    """
    Fused add + split for 14x14 case
    Input: [1, 197, 384], [1, 197, 384]
    Output: [1, 1, 384], [1, 196, 384]
    """
    B, N, C = in_0.shape  # [1, 197, 384]
    
    # Output tensors
    out_0 = torch.empty((1, 1, C), dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((1, 196, C), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten for easier addressing
    in_0_flat = in_0.view(-1)
    in_1_flat = in_1.view(-1)
    out_0_flat = out_0.view(-1)
    out_1_flat = out_1.view(-1)
    
    BLOCK_SIZE = 1024
    num_elements_out1 = 196 * C  # 75264
    num_programs = (num_elements_out1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_split_kernel_14[(num_programs,)](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        out_0_ptr=out_0_flat,
        out_1_ptr=out_1_flat,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_0, out_1


def replacement_func():
    return fused_add_split_14