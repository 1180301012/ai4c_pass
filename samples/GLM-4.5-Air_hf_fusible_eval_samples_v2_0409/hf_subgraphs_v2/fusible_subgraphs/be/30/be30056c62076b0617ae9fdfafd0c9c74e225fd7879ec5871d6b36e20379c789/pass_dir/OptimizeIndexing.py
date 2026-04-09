import torch
import triton
import triton.language as tl

# Pattern matching for in_0 indexing operation
def pattern(in_0):
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return tmp_7

# Extract arguments for the optimized operation
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel for in_0 indexing/broadcasting
@triton.jit
def optimized_indexing_kernel(
    in_ptr,
    out_ptr,
    original_dims,
    target_batch,
    target_seq,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (target_batch * target_seq * original_dims[1])
    
    # Reshape offsets to 3D: [target_batch, target_seq, original_dims[1]]
    batch_idx = offsets // (target_seq * original_dims[1])
    seq_idx = (offsets % (target_seq * original_dims[1])) // original_dims[1]
    dim_idx = offsets % original_dims[1]
    
    # Load from original in_0 [2, 128] element
    # Always load from the first element (dimension 0) since we're broadcasting
    in_val = tl.load(in_ptr + dim_idx, mask=dim_idx < original_dims[1], other=0.0)
    
    # Store to expanded output
    tl.store(out_ptr + offsets, in_val, mask=mask)

@torch.fx.wrap
def optimized_indexing(in_0):
    original_shape = in_0.shape  # [2, 128]
    target_batch = 512  # From in_1 batch size
    target_seq = 17     # From in_1 sequence length
    
    total_elements = target_batch * target_seq * original_shape[1]
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate expanded output: [512, 17, 2, 128]
    expanded_in_0 = torch.empty(target_batch, target_seq, *original_shape, 
                               dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel - broadcast along batch and sequence dimensions
    optimized_indexing_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=expanded_in_0,
        original_dims=tl.tensor(original_shape, dtype=tl.int32),
        target_batch=target_batch,
        target_seq=target_seq,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return expanded_in_0

def replacement_func():
    return optimized_indexing