import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Fuse permute + view for Graph 2 (576 -> 24x24)
    """
    tmp_0 = in_0.permute(0, 2, 1)
    tmp_1 = tmp_0.view(1, 384, 24, 24)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_permute_view_kernel_24(
    in_ptr, out_ptr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for permute + view (24x24 case)
    Input: [1, 576, 384]
    Output: [1, 384, 24, 24]
    """
    pid = tl.program_id(0)
    
    num_elements = C * 24 * 24  # 221184
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Convert linear offset to (c, i, j) for view
    # out shape: [C=384, 24, 24], flat index = c * 576 + i * 24 + j
    c = offsets // 576
    rem = offsets % 576
    i = rem // 24
    j = rem % 24
    
    # Map to input indices: after permute, input is [1, 384, 576]
    # Input index mapping: output[c, i, j] corresponds to input[0, c, i*24 + j]
    in_dim1 = i * 24 + j
    in_offset = in_dim1 * C + c
    
    vals = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Store to out
    out_offset = c * 576 + i * 24 + j
    tl.store(out_ptr + out_offset, vals, mask=mask)


@torch.fx.wrap
def fused_permute_view_24(in_0):
    """
    Fused permute + view for 24x24 case
    Input: [1, 576, 384]
    Output: [1, 384, 24, 24]
    """
    B, N, C = in_0.shape  # [1, 576, 384]
    
    # Output tensor
    out = torch.empty((1, C, 24, 24), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten for easier addressing
    in_flat = in_0.view(-1)
    out_flat = out.view(-1)
    
    BLOCK_SIZE = 1024
    num_elements = C * 24 * 24  # 221184
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_permute_view_kernel_24[(num_programs,)](
        in_ptr=in_flat,
        out_ptr=out_flat,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return fused_permute_view_24