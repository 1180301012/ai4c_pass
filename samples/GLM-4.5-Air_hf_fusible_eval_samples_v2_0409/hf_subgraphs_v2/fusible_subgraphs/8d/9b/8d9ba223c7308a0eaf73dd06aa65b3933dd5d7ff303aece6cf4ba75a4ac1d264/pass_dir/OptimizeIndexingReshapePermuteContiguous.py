import torch
import triton
import triton.language as tl

@triton.jit
def simple_indexing_kernel(
    input_ptr,           # Pointer to linear tensor after view(-1, head_dim)
    index_ptr,           # Pointer to index tensor (in_0.view(-1))
    out_ptr,             # Output pointer
    numel,               # Number of indexing operations
    head_dim,            # Head dimension for stride calculation
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel for indexed gather operation that preserves head_dim dimension
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # We need to compute both the index position and the head dimension position
    index_pos = offsets // head_dim  # Which index we're accessing
    head_pos = offsets % head_dim     # Which element within the head vector
    
    mask = index_pos < numel
    
    # Load indices
    indices = tl.load(index_ptr + index_pos, mask=mask, other=0).to(tl.int32)
    
    # Calculate gather offset: indices * head_dim + head_pos
    gather_offset = indices * head_dim + head_pos
    
    # Load input data and perform gather
    input_val = tl.load(input_ptr + gather_offset, mask=(gather_offset < (numel * head_dim)) & mask, other=0.0)
    
    # Store result
    tl.store(out_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_indexing_reshape_permute_contiguous(linear, index, head_dim):
    linear_numel = linear.numel()
    index_numel = index.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (index_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(index_numel * head_dim, dtype=linear.dtype, device=linear.device)
    
    simple_indexing_kernel[(num_programs,)](
        input_ptr=linear,
        index_ptr=index,
        out_ptr=out,
        numel=index_numel,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(linear, index, head_dim):
    # Pattern matching: linear.view(-1, head_dim)[index.view(-1)]
    # This is the core indexed gather operation that we want to optimize
    tmp_3 = linear.view(-1, head_dim)
    tmp_4 = index.view(-1)
    tmp_5 = tmp_3[tmp_4]
    return tmp_5

def replacement_args(linear, index, head_dim):
    return (linear, index, head_dim)

def replacement_func():
    return optimized_indexing_reshape_permute_contiguous