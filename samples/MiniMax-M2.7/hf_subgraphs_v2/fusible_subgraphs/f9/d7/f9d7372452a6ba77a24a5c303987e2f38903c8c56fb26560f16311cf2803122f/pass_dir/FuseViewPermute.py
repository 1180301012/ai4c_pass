import torch
import triton
import triton.language as tl

@triton.jit
def view_permute_kernel(
    in_ptr,
    out_ptr,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    dim3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for view(1, 32, -1) + permute(0, 2, 1).
    
    Input shape: [dim0, dim1, dim2, dim3] = [1, 32, 64, 48]
    Output shape: [dim0, dim2*dim3, dim1] = [1, 3072, 32]
    
    The kernel performs:
    - view: reshape to [1, 32, 3072] 
    - permute: transpose to [1, 3072, 32]
    
    We iterate over output positions and compute the correct input index.
    Output position [b, i, j] corresponds to input position [b, j, i // dim1, i % dim1]
    where dim1=32.
    """
    pid = tl.program_id(0)
    
    # Each thread handles BLOCK_SIZE consecutive output elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (dim0 * dim2 * dim3 * dim1)
    
    # Compute output coordinates: [b, i, j]
    # Total output size = dim0 * (dim2*dim3) * dim1
    total_out = dim2 * dim3 * dim1
    b = offsets // total_out
    remaining = offsets % total_out
    i = remaining // dim1
    j = remaining % dim1
    
    # Compute input coordinates for view(1, 32, 64, 48) -> view(1, 32, 3072) -> permute(0, 2, 1)
    # Output [b, i, j] -> Input [b, j, i // dim3, i % dim3]
    in_b = b
    in_j = j
    in_i_div = i // dim3
    in_i_mod = i % dim3
    
    # Input flat index for shape [1, 32, 64, 48]
    in_flat_idx = (in_b * dim1 * dim2 * dim3 + 
                   in_j * dim2 * dim3 + 
                   in_i_div * dim3 + 
                   in_i_mod)
    
    # Load and store
    val = tl.load(in_ptr + in_flat_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_view_permute(in_1):
    """
    Wrapper function that launches the fused view+permute kernel.
    
    Args:
        in_1: Input tensor [1, 32, 64, 48]
    
    Returns:
        Transposed view: [1, 3072, 32]
    """
    # Get dimensions
    dim0, dim1, dim2, dim3 = in_1.shape  # [1, 32, 64, 48]
    
    # Compute output size: [1, 3072, 32]
    out_size = dim0 * dim2 * dim3 * dim1
    
    # Output shape
    out_shape = (dim0, dim2 * dim3, dim1)
    
    # Allocate output
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    out_flat = out.flatten()
    
    BLOCK_SIZE = 1024
    num_programs = (out_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    view_permute_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out_flat,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    Returns tmp_4
    """
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_view_permute