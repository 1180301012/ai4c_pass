import torch
import triton
import triton.language as tl

@triton.jit
def triton_broadcast_kernel(
    in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    """
    Fused view + expand broadcast kernel.
    Input:  [1, 2, 8, 8]  (in_0)
    After view: [1, 2, 1, 8, 8]
    After expand: [1, 2, 64, 8, 8]
    
    The view reshapes to [1, 2, 1, 8, 8] in contiguous memory.
    The expand broadcasts dimension 2 from 1 to 64.
    Output shape: [1, 2, 64, 8, 8] = 1024 elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For output element at position d in flattened [1, 2, 64, 8, 8]:
    # Output indices: b, c, b2, h, w where dim 2 (b2) broadcasts from 0
    # Input position is (b, c, 0, h, w) -> offset in view is d % 512 + d % 8
    
    # Original computation:
    # b = d // 1024, c = (d // 512) % 2, b2 = (d // 64) % 64, h = (d // 8) % 8, w = d % 8
    # For broadcast: b2 always maps to 0
    # in_view_offset = b*128 + c*64 + 0*8 + h*8 + w = (d // 128) * 128 + (d // 8) % 64 * 8 + d % 8
    
    in_offsets = (offsets // 128) * 128 + (offsets // 8) % 64 * 8 + offsets % 8
    
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def triton_broadcast_wrapper(in_0):
    """
    Fused view + expand: [1, 2, 8, 8] -> [1, 2, 64, 8, 8]
    """
    N = 1 * 2 * 64 * 8 * 8  # 1024 elements
    BLOCK_SIZE = 1024
    num_programs = 1  # Single program can handle all 1024 elements
    
    out = torch.empty((1, 2, 64, 8, 8), dtype=in_0.dtype, device=in_0.device)
    
    triton_broadcast_kernel[(num_programs,)](
        in_0, out, N, BLOCK_SIZE
    )
    
    return out

def pattern(in_0):
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return triton_broadcast_wrapper