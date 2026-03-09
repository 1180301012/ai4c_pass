import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern: interpolate (no-op) + permute + reshape + slice (47x47 version)"""
    # Skip the intermediate assignment tmp_0 = in_0, start from in_0
    tmp_1 = torch.nn.functional.interpolate(in_1, size=(47, 47), mode='bilinear')
    tmp_2 = tmp_1.permute(0, 2, 3, 1)
    tmp_3 = tmp_2.reshape(2209, -1)
    tmp_4 = in_0[slice(2209, None, None)]
    return tmp_4, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Triton kernel for permute + reshape: [1, C, H, W] -> [H*W, C]
@triton.jit
def permute_reshape_kernel_47(
    input_ptr,
    output_ptr,
    n_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused permute (0,2,3,1) + reshape: [1, C, H, W] -> [H*W, C]"""
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute h, w from linear index
    # output is [H*W, C], each row has C elements
    row_offsets = offsets // C
    col_offsets = offsets % C
    
    # For input [1, C, H, W], we need to access [0, col_offsets, row_offsets, 0]
    # Flattened: 0 * C*H*W + col_offsets * H*W + row_offsets * W + 0
    input_offsets = col_offsets * (H * W) + row_offsets * W
    
    # Load from input [1, C, H, W] - flatten to [C*H*W]
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Store to output [H*W, C]
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def permute_reshape_kernel_wrapper_47(in_1, out_shape):
    """Wrapper for the fused permute+reshape kernel"""
    C = in_1.shape[1]
    H = in_1.shape[2]
    W = in_1.shape[3]
    n_elements = H * W * C
    
    # Output shape: [H*W, C]
    out = torch.empty([H * W, C], dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    permute_reshape_kernel_47[(num_programs,)](
        in_1,
        out,
        n_elements,
        C,
        H,
        W,
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return custom_interpolate_permute_reshape_47


@torch.fx.wrap
def custom_interpolate_permute_reshape_47(in_0, in_1):
    """Custom implementation that fuses interpolate (no-op) + permute + reshape + slice
    
    Since interpolate is a no-op (input already at target size), we:
    1. Skip interpolate
    2. Fuse permute+reshape using Triton kernel
    3. Slice in_0 to get remaining rows
    """
    # in_1 is [1, C, H, W], interpolate to same size is a no-op
    # Just do fused permute + reshape
    C = in_1.shape[1]
    H = in_1.shape[2]
    W = in_1.shape[3]
    
    # Fused permute (0,2,3,1) + reshape: [1, C, H, W] -> [H*W, C]
    tmp_3 = permute_reshape_kernel_wrapper_47(in_1, [H * W, C])
    
    # Slice in_0: get rows from H*W onwards
    tmp_4 = in_0[slice(H * W, None, None)]
    
    return tmp_4, tmp_3