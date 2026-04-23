import torch
import triton
import triton.language as tl

# Pattern matching: view + expand fusion for broadcasting
def pattern(in_0):
    """
    Match view(1, 2, 1, 8, 8) followed by expand(1, 2, 64, 8, 8).
    in_0 shape: [1, 2, 8, 8]
    Output shape: [1, 2, 64, 8, 8]
    """
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def triton_broadcast_view_expand_kernel(
    in_ptr,
    out_ptr,
    n_elements_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_out
    
    # Output shape: [1, 2, 64, 8, 8] -> indices: (d0, d1, d2, d3, d4)
    # Total elements = 1 * 2 * 64 * 8 * 8 = 8192
    
    # Calculate 5D indices from offset
    d4 = offsets % 8
    rem = offsets // 8
    d3 = rem % 8
    rem = rem // 8
    d2 = rem % 64
    rem = rem // 64
    d1 = rem % 2
    d0 = rem // 2
    
    # For input [1, 2, 8, 8] view to [1, 2, 1, 8, 8], we index with (d0, d1, 0, d3, d4)
    # The broadcasted value is at input indices (d0, d1, d3, d4) where d2 is always 0
    # Input strides: [128, 64, 8, 1]
    in_offset = d0 * 128 + d1 * 64 + d3 * 8 + d4
    
    # Load from input and store to output
    x = tl.load(in_ptr + in_offset)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def triton_broadcast_view_expand(in_0):
    """
    Optimized view + expand: [1, 2, 8, 8] -> [1, 2, 1, 8, 8] -> [1, 2, 64, 8, 8]
    Direct broadcast from [1, 2, 8, 8] to [1, 2, 64, 8, 8] avoiding intermediate tensor.
    """
    assert in_0.shape == (1, 2, 8, 8), f"Expected shape [1, 2, 8, 8], got {in_0.shape}"
    
    # Output shape: [1, 2, 64, 8, 8]
    N_out = 1 * 2 * 64 * 8 * 8  # 8192
    BLOCK_SIZE = 512
    
    num_programs = (N_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with expanded shape
    out = torch.empty((1, 2, 64, 8, 8), dtype=in_0.dtype, device=in_0.device)
    
    triton_broadcast_view_expand_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements_out=N_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_broadcast_view_expand