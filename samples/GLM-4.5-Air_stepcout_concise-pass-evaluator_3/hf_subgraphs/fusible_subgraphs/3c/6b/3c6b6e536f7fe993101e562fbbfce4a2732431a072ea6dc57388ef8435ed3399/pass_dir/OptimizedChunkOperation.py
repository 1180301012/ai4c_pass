import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return tmp_9, tmp_10

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def chunk_kernel(
    in_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # For chunking along last dimension with 2 chunks,
    # each element goes to either out1 or out2 based on position
    # We need to determine which chunk each element belongs to
    total_dims = len(x.shape) if hasattr(x, 'shape') else 1
    last_dim_offset = tl.arange(0, BLOCK_SIZE) % (BLOCK_SIZE // 2)
    
    # Create masks for each chunk
    chunk_size = n_elements // 2
    offset1 = offsets < chunk_size
    offset2 = offsets >= chunk_size
    
    if tl.any(offset1):
        # Elements in first chunk (first half)
        chunk1_offsets = offsets[offset1] - offset1[mask].cumsum() * offset1[mask]
        tl.store(out1_ptr + chunk1_offsets, x[offset1], mask=offset1[mask])
    
    if tl.any(offset2):
        # Elements in second chunk (second half)  
        chunk2_offsets = offsets[offset2] - offset2[mask].cumsum() * offset2[mask] - chunk_size
        tl.store(out2_ptr + chunk2_offsets, x[offset2], mask=offset2[mask])

@torch.fx.wrap
def optimized_chunk(in_3):
    # Get input shape
    original_shape = in_3.shape
    in_elements = in_3.numel()
    
    # Create output tensors (each chunk has half the elements in last dimension)
    chunk_shape = list(original_shape[:-1]) + [original_shape[-1] // 2]
    
    out1 = torch.empty(chunk_shape, dtype=in_3.dtype, device=in_3.device)
    out2 = torch.empty(chunk_shape, dtype=in_3.dtype, device=in_3.device)
    
    BLOCK_SIZE = 1024
    num_programs = (in_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    chunk_kernel[(num_programs,)](
        in_ptr=in_3,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=in_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return optimized_chunk