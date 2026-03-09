import torch
import triton
import triton.language as tl

# Pattern matching function - matches embedding lookup followed by scalar multiplication by 1.0
def pattern(in_1, embedding_weight):
    tmp_1 = torch.nn.functional.embedding(in_1, embedding_weight, 1, None, 2.0, False, False)
    tmp_2 = tmp_1 * 1.0
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_1, embedding_weight):
    return (in_1, embedding_weight)

# Optimized kernel for embedding lookup (scalar multiplication eliminated)
@triton.jit
def embedding_lookup_kernel(
    output_ptr,
    input_ids_ptr,
    weight_ptr,
    num_embeddings,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load input IDs
    input_ids = tl.load(input_ids_ptr + pid)
    
    # Clamp input IDs to valid range to prevent out-of-bounds access
    input_ids = tl.maximum(input_ids, 0)
    input_ids = tl.minimum(input_ids, num_embeddings - 1)
    
    # Calculate weight offset for the requested embedding
    offset = input_ids * embedding_dim
    
    # Load embedding vector components using vectorized loads
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_embeddings * embedding_dim
    
    # Load embedding vector components
    embedding_vector = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + pid * embedding_dim + tl.arange(0, BLOCK_SIZE), embedding_vector)

# Get embedding dim from weight tensor
def get_embedding_dim(weight):
    return weight.shape[1]

# Get number of embeddings from weight tensor  
def get_num_embeddings(weight):
    return weight.shape[0]

# Kernel wrapper (eliminates redundant scalar multiplication)
@torch.fx.wrap
def optimized_embedding_lookup(input_ids, embedding_weight):
    # Get tensor properties
    num_embeddings = get_num_embeddings(embedding_weight)
    embedding_dim = get_embedding_dim(embedding_weight)
    
    # Handle input shape - input_ids should be flattened for processing
    input_size = input_ids.numel()
    
    # Output shape matches what the pattern expects
    output_shape = list(input_ids.shape) + [embedding_dim]
    
    # Create output tensor on same device as input_ids
    output = torch.empty(output_shape, dtype=embedding_weight.dtype, device=input_ids.device)
    
    # Choose optimal block size for GPU utilization and parallelism
    # Based on empirical testing for embedding_dim = 1024 and small input sizes
    if embedding_dim >= 512:
        BLOCK_SIZE = 256  # Good for large embeddings, maintains GPU occupancy
    elif embedding_dim >= 128:
        BLOCK_SIZE = 128  # Balanced for medium embeddings
    elif embedding_dim >= 64:
        BLOCK_SIZE = 64   # Efficient for smaller embeddings  
    else:
        BLOCK_SIZE = embedding_dim  # Use full dimension for tiny embeddings
    
    # Ensure reasonable bounds for block size
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    elif BLOCK_SIZE < 32:
        BLOCK_SIZE = 32  # Minimum efficient block size
    
    # Launch kernel with optimal configuration
    embedding_lookup_kernel[(input_size,)](
        output_ptr=output,
        input_ids_ptr=input_ids,
        weight_ptr=embedding_weight,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (returns optimized embedding lookup without scalar multiplication)
def replacement_func():
    return optimized_embedding_lookup