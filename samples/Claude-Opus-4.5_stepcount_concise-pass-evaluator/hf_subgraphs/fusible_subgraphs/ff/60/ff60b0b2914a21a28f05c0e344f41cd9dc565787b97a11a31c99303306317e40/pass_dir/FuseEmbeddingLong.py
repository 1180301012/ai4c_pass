import torch
import triton
import triton.language as tl

# Pattern matching function - match just the embedding operation
def pattern(in_1, in_2):
    """
    Match the embedding lookup pattern
    in_1: input_ids (int64)
    in_2: embedding weight (bfloat16)
    """
    tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    return tmp_3

# Argument extraction function
def replacement_args(in_1, in_2):
    return (in_1, in_2)

# Triton kernel for embedding lookup - single row per program, large block
@triton.jit
def embedding_kernel_fast(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_indices,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast embedding lookup kernel
    Each program handles one index (one row of the output)
    Uses large BLOCK_SIZE to process entire row in single load/store
    """
    # Get the row index this program handles
    row_idx = tl.program_id(0)
    
    if row_idx >= num_indices:
        return
    
    # Load the embedding index for this position
    idx = tl.load(indices_ptr + row_idx)
    
    # Calculate base pointers
    weight_row_ptr = weight_ptr + idx * embedding_dim
    output_row_ptr = output_ptr + row_idx * embedding_dim
    
    # Process entire row in single load/store operation
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < embedding_dim
    values = tl.load(weight_row_ptr + offsets, mask=mask)
    tl.store(output_row_ptr + offsets, values, mask=mask)

# Wrapper function for the optimized embedding
@torch.fx.wrap
def optimized_embedding(in_1, in_2):
    """
    Optimized embedding lookup
    in_1: input_ids (int64)
    in_2: embedding weight (bfloat16)
    """
    # Get shapes
    batch_shape = in_1.shape
    num_indices = in_1.numel()
    embedding_dim = in_2.shape[1]
    
    # Flatten indices for easier processing
    flat_indices = in_1.reshape(-1).contiguous()
    
    # Allocate output tensor
    output = torch.empty(num_indices, embedding_dim, dtype=in_2.dtype, device=in_2.device)
    
    # Use BLOCK_SIZE=2048 to cover embedding_dim=1536 in single load
    BLOCK_SIZE = 2048
    grid = (num_indices,)
    
    embedding_kernel_fast[grid](
        flat_indices,
        in_2,
        output,
        num_indices,
        embedding_dim,
        BLOCK_SIZE,
    )
    
    # Reshape output to match expected shape
    output_shape = batch_shape + (embedding_dim,)
    result = output.view(output_shape)
    
    return result

# Replacement function - returns the wrapper function
def replacement_func():
    return optimized_embedding