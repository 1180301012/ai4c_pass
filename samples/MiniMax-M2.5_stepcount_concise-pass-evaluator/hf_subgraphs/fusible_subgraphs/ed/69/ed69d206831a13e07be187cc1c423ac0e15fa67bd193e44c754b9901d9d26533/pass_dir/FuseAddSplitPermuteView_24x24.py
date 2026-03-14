import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the computation pattern for Graph 2:
    1. Add two tensors
    2. Split along dimension 1
    3. Permute and view the second part
    
    Graph 2: split [1, 576], view to 24x24
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 576], 1)
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 24, 24)
    return (tmp_2, tmp_5)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_split_permute_view_24_kernel(
    in_0_ptr, in_1_ptr,
    out_0_ptr, out_1_ptr,
    C: tl.constexpr, split_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for Graph 2 that performs:
    1. Element-wise addition
    2. Split extraction (slice)
    3. Permute and view to 24x24
    
    Input shapes: [1, 577, 384] each
    Output shapes: ([1, 1, 384], [1, 384, 24, 24])
    """
    pid = tl.program_id(0)
    
    # Compute out_0: the first element from split (position 0)
    # out_0 = in_0[0, 0, :] + in_1[0, 0, :]
    offsets_c = tl.arange(0, C)
    
    # Only program 0 computes out_0
    if pid == 0:
        in_0_val = tl.load(in_0_ptr + offsets_c)
        in_1_val = tl.load(in_1_ptr + offsets_c)
        out_0_val = in_0_val + in_1_val
        tl.store(out_0_ptr + offsets_c, out_0_val)
    
    # Compute out_1: from elements 1:577 (indices 1 to 576)
    # Shape transformation: [1, 576, 384] -> permute -> [1, 384, 576] -> view -> [1, 384, 24, 24]
    num_elements_out1 = C * 24 * 24  # 384 * 576 = 221184
    
    # Each program processes a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements_out1
    
    # Convert linear offset to (c, i, j) for view
    # out_1 shape: [C=384, 24, 24], flat index = c * 576 + i * 24 + j
    c = offsets // 576
    rem = offsets % 576
    i = rem // 24
    j = rem % 24
    
    # Map to input indices: after permute, input is [1, 384, 576]
    # Input index mapping: output[c, i, j] corresponds to input[0, c, i*24 + j + 1]
    # (skip +1 because we skip the first split element)
    in_idx = c * (split_size - 1) + i * 24 + j + 1  # split_size = 577, so split_size - 1 = 576
    
    # Load from in_0 and in_1 at the computed indices
    # in_0 and in_1 are contiguous [1, 577, 384], so linear index = dim1 * 384 + c
    in_0_offset = in_idx * C + c
    in_1_offset = in_idx * C + c
    
    in_0_vals = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    out_1_vals = in_0_vals + in_1_vals
    
    # Store to out_1
    out_1_offset = c * 576 + i * 24 + j
    tl.store(out_1_ptr + out_1_offset, out_1_vals, mask=mask)


@torch.fx.wrap
def fused_add_split_permute_view_24(in_0, in_1):
    """
    Fused kernel for Graph 2:
    - Element-wise addition
    - Split extraction  
    - Permute and view reshape
    
    Input shapes: [1, 577, 384] each
    Output shapes: ([1, 1, 384], [1, 384, 24, 24])
    """
    B, N, C = in_0.shape  # [1, 577, 384]
    split_size = N  # 577
    
    # Output tensors
    out_0 = torch.empty((1, 1, C), dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((1, C, 24, 24), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten for easier addressing
    in_0_flat = in_0.view(-1)
    in_1_flat = in_1.view(-1)
    out_0_flat = out_0.view(-1)
    out_1_flat = out_1.view(-1)
    
    BLOCK_SIZE = 1024
    num_elements_out1 = C * 24 * 24  # 221184
    num_programs = (num_elements_out1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_split_permute_view_24_kernel[(num_programs,)](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        out_0_ptr=out_0_flat,
        out_1_ptr=out_1_flat,
        C=C, split_size=split_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_0, out_1


def replacement_func():
    return fused_add_split_permute_view_24