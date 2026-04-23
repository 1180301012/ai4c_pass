import torch
import triton
import triton.language as tl

# Pattern matching: combined sum/div and view/expand operations
def pattern(in_0, in_1):
    """
    Match the full subgraph:
    - Normalization branch: in_1.sum(dim=2, keepdim=True) then divide
    - View/Expand branch: in_0.view(1, 2, 1, 8, 8).expand(1, 2, 64, 8, 8)
    Returns: (expanded_tensor, normalized_tensor)
    """
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return (tmp_3, tmp_1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    # Input pointers
    in_0_ptr,  # [1, 2, 8, 8]
    in_1_ptr,  # [1, 2, 8, 8]
    # Output pointers
    out_0_ptr,  # [1, 2, 64, 8, 8]
    out_1_ptr,  # [1, 2, 8, 8]
    # Metadata
    n_in_elements,
    n_out_0_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Process in chunks
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # ========== Normalization Branch (out_1) ==========
    # in_1 shape: [1, 2, 8, 8] -> 128 elements
    # sum(dim=2) -> sum over 8 elements for each (d0, d1, d3)
    # Then divide each element by its group's sum
    
    if block_start + BLOCK_SIZE <= n_in_elements:
        mask = offsets < n_in_elements
        # Calculate indices for input [1, 2, 8, 8]
        d3 = offsets % 8
        rem = offsets // 8
        d2 = rem % 8
        rem = rem // 8
        d1 = rem % 2
        d0 = rem // 2
        
        # Compute sum over dim 2 (8 elements per group)
        x0 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 0 * 8 + d3)
        x1 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 1 * 8 + d3)
        x2 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 2 * 8 + d3)
        x3 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 3 * 8 + d3)
        x4 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 4 * 8 + d3)
        x5 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 5 * 8 + d3)
        x6 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 6 * 8 + d3)
        x7 = tl.load(in_1_ptr + d0 * 128 + d1 * 64 + 7 * 8 + d3)
        sum_val = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
        
        x = tl.load(in_1_ptr + offsets)
        out = x / (sum_val + 1e-8)
        tl.store(out_1_ptr + offsets, out, mask=mask)
    
    # ========== View/Expand Branch (out_0) ==========
    # in_0 shape: [1, 2, 8, 8]
    # view(1, 2, 1, 8, 8) -> expand(1, 2, 64, 8, 8)
    # out_0 shape: [1, 2, 64, 8, 8] -> 8192 elements
    
    out_0_offset = block_start
    out_0_mask = (out_0_offset + tl.arange(0, BLOCK_SIZE)) < n_out_0_elements
    
    if out_0_offset < n_out_0_elements:
        out_0_offsets = out_0_offset + tl.arange(0, BLOCK_SIZE)
        out_0_mask = out_0_offsets < n_out_0_elements
        
        # Calculate 5D indices for output [1, 2, 64, 8, 8]
        d4 = out_0_offsets % 8
        rem = out_0_offsets // 8
        d3 = rem % 8
        rem = rem // 8
        d2 = rem % 64
        rem = rem // 64
        d1 = rem % 2
        d0 = rem // 2
        
        # For input [1, 2, 8, 8] view to [1, 2, 1, 8, 8], index is (d0, d1, 0, d3, d4)
        # In linear: d0*128 + d1*64 + d3*8 + d4
        in_0_offset = d0 * 128 + d1 * 64 + d3 * 8 + d4
        
        x = tl.load(in_0_ptr + in_0_offset)
        tl.store(out_0_ptr + out_0_offsets, x, mask=out_0_mask)

@torch.fx.wrap
def fused_normalize_broadcast(in_0, in_1):
    """
    Fused kernel for both:
    1. Normalization: in_1.sum(dim=2, keepdim=True) / in_1
    2. View/Expand: in_0.view(1, 2, 1, 8, 8).expand(1, 2, 64, 8, 8)
    """
    assert in_0.shape == (1, 2, 8, 8), f"Expected in_0 shape [1, 2, 8, 8], got {in_0.shape}"
    assert in_1.shape == (1, 2, 8, 8), f"Expected in_1 shape [1, 2, 8, 8], got {in_1.shape}"
    
    # Normalization output: [1, 2, 8, 8] = 128 elements
    out_1 = torch.empty_like(in_1)
    n_in = 128
    
    # Broadcast output: [1, 2, 64, 8, 8] = 8192 elements
    out_0 = torch.empty((1, 2, 64, 8, 8), dtype=in_0.dtype, device=in_0.device)
    n_out = 8192
    
    # Use enough programs to cover both outputs
    BLOCK_SIZE = 128
    n_programs = max((n_in + BLOCK_SIZE - 1) // BLOCK_SIZE, 
                     (n_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    fused_kernel[(n_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        n_in_elements=n_in,
        n_out_0_elements=n_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_0, out_1)

def replacement_func():
    return fused_normalize_broadcast