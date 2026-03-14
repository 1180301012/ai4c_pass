import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    1. Add two tensors
    2. Split along dimension 1
    3. Permute and view the second part
    
    Graph 1: split [1, 196], view to 14x14
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 14, 14)
    return (tmp_2, tmp_5)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_split_permute_view_kernel(
    in_0_ptr, in_1_ptr,
    out_0_ptr, out_1_ptr,
    B: tl.constexpr, C: tl.constexpr, split_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that performs:
    1. Element-wise addition
    2. Split extraction (slice)
    3. Permute and view
    """
    # Process batch elements
    pid = tl.program_id(0)
    
    # Output shapes
    # out_0: [1, 1, 384] - single position from split
    # out_1: [1, 384, 14, 14] - reshaped from [1, 196, 384] -> [1, 384, 14, 14]
    
    # For out_0: we need the first element along dim 1 (index 0 of split [1, 196])
    # The input is [B=1, 197, C=384], output out_0 is [1, 1, 384]
    # out_1 is [1, 384, 14, 14] = [1, C, 14, 14] where 14*14 = 196
    
    # Load input tensors
    # in_0 and in_1 are [1, 197, 384]
    
    # Compute out_0: the first element from split (position 0)
    # out_0 = in_0[0, 0, :] + in_1[0, 0, :]
    offsets_c = tl.arange(0, C)
    
    # Load in_0[0, 0, :] and in_1[0, 0, :]
    in_0_val = tl.load(in_0_ptr + offsets_c)  # Shape [C]
    in_1_val = tl.load(in_1_ptr + offsets_c)  # Shape [C]
    out_0_val = in_0_val + in_1_val
    tl.store(out_0_ptr + offsets_c, out_0_val)
    
    # Compute out_1: from elements 1:197 (indices 1 to 196)
    # Shape transformation: [1, 196, 384] -> permute -> [1, 384, 196] -> view -> [1, 384, 14, 14]
    # Original indices: we skip index 0 (the first split) and take indices 1-196
    # After permute (0,2,1): [1, 384, 196]
    # After view: [1, 384, 14, 14]
    
    # Total elements in out_1: 384 * 14 * 14 = 75264
    # We process BLOCK_SIZE elements at a time
    
    num_elements_out1 = C * 14 * 14  # 384 * 196 = 75264
    
    # Each program processes a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements_out1
    
    # Convert linear offset to (c, i, j) for view
    # out_1 shape: [C=384, 14, 14], flat index = c * 196 + i * 14 + j
    c = offsets // 196
    rem = offsets % 196
    i = rem // 14
    j = rem % 14
    
    # Map to input indices: after permute, input is [1, 384, 196]
    # Input index mapping: output[c, i, j] corresponds to input[0, c, i*14 + j + 1]
    # (skip +1 because we skip the first split element)
    in_idx = c * (split_size - 1) + i * 14 + j + 1  # split_size = 197, so split_size - 1 = 196
    
    # Load from in_0 and in_1 at the computed indices
    # in_0 and in_1 are contiguous [B, 197, C], so linear index = b * 197 * C + dim1 * C + c
    # b = 0 always since B = 1
    in_0_linear = in_idx * C + c  # This is wrong, let me fix
    
    # Actually, for in_0[0, in_idx, c]:
    # linear = 0 * 197 * 384 + in_idx * 384 + c = in_idx * 384 + c
    in_0_offset = in_idx * C + c
    in_1_offset = in_idx * C + c
    
    in_0_vals = tl.load(in_0_ptr + in_0_offset, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    out_1_vals = in_0_vals + in_1_vals
    
    # Store to out_1
    # out_1 is [1, 384, 14, 14], flat offset = c * 196 + i * 14 + j
    out_1_offset = c * 196 + i * 14 + j
    tl.store(out_1_ptr + out_1_offset, out_1_vals, mask=mask)


@torch.fx.wrap
def fused_add_split_permute_view(in_0, in_1):
    """
    Fused kernel for:
    - Element-wise addition
    - Split extraction  
    - Permute and view reshape
    
    Input shapes: [1, 197, 384] each
    Output shapes: ([1, 1, 384], [1, 384, 14, 14])
    """
    B, N, C = in_0.shape  # [1, 197, 384]
    split_size = N  # 197
    second_part_size = N - 1  # 196
    
    # Output tensors
    out_0 = torch.empty((1, 1, C), dtype=in_0.dtype, device=in_0.device)
    out_1 = torch.empty((1, C, 14, 14), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten for easier addressing
    in_0_flat = in_0.view(-1)
    in_1_flat = in_1.view(-1)
    out_0_flat = out_0.view(-1)
    out_1_flat = out_1.view(-1)
    
    BLOCK_SIZE = 1024
    num_elements_out1 = C * 14 * 14  # 75264
    num_programs = (num_elements_out1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_split_permute_view_kernel[(num_programs,)](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        out_0_ptr=out_0_flat,
        out_1_ptr=out_1_flat,
        B=B, C=C, split_size=split_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_0, out_1


def replacement_func():
    return fused_add_split_permute_view