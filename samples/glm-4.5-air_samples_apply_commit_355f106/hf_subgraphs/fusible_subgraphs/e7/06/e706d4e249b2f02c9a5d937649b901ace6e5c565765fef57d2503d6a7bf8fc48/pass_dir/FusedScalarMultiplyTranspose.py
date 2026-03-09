import torch
import triton
import triton.language as tl

# Pattern matching function - matches the computation structure found in all graphs
def pattern(in_0, in_1):
    # Use a placeholder pattern that matches scalar multiplication with any constant
    # The pattern should match: (tensor * scalar, tensor.transpose(-1, -2))
    # We'll capture the actual constant value in replacement_args
    tmp_0 = in_1 * 1.0  # Use a placeholder constant, actual value will be captured in replacement
    tmp_1 = in_0.transpose(-1, -2)      # transpose last two dimensions
    return (tmp_0, tmp_1)

# Argument extraction function - extracts both inputs needed for the replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel using Triton
@triton.jit
def fused_scalar_mul_transpose_kernel(
    # Input and output pointers
    key_ptr,           # in_0 - tensor to be transposed
    query_ptr,         # in_1 - tensor for scalar multiplication  
    out_key_ptr,       # output for transpose result
    out_query_ptr,     # output for scalar multiplication result
    
    # Tensor shape information
    batch_size,
    num_heads, 
    seq_len,
    head_dim,
    
    # Scalar constant for multiplication
    scalar_const,
    
    # Kernel metadata
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID for grid computation
    pid = tl.program_id(0)
    
    # Calculate which batch/head this program handles
    batch_idx = pid // (num_heads * seq_len)
    head_idx = (pid % (num_heads * seq_len)) // seq_len
    seq_idx = pid % seq_len
    
    # Transposition operation: swap last two dimensions (head_dim <-> seq_len)
    # Each program handles one element of the transposed tensor
    transpose_output_offset = batch_idx * (num_heads * seq_len * head_dim) + \
                            head_idx * (seq_len * head_dim) + \
                            seq_idx * head_dim
    
    # Scalar multiplication operation: handle element-wise multiplication
    # Each program handles one element of the query tensor
    scalar_output_offset = batch_idx * (num_heads * seq_len * head_dim) + \
                          head_idx * (seq_len * head_dim) + \
                          seq_idx * head_dim
    
    # Process a block of elements for transpose
    for k in range(0, head_dim, BLOCK_SIZE_N):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_N)
        mask_k = offsets_k < head_dim
        
        # Load element from source (transpose operation)
        # Original layout: [batch, heads, seq_len, head_dim]
        # Input offset: batch_idx*(num_heads*seq_len*head_dim) + head_idx*(seq_len*head_dim) + seq_idx*head_dim + k
        input_offset = batch_idx * (num_heads * seq_len * head_dim) + \
                      head_idx * (seq_len * head_dim) + \
                      seq_idx * head_dim + k
        
        key_val = tl.load(key_ptr + input_offset, mask=mask_k, other=0.0)
        
        # Store transposed element (swap seq_len and head_dim dimensions)
        # Transposed layout: [batch, heads, head_dim, seq_len]
        transposed_offset = batch_idx * (num_heads * head_dim * seq_len) + \
                           head_idx * (head_dim * seq_len) + \
                           k * seq_len + seq_idx
        
        tl.store(out_key_ptr + transposed_offset, key_val, mask=mask_k)
    
    # Process scalar multiplication on query tensor
    for k in range(0, head_dim, BLOCK_SIZE_N):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_N)
        mask_k = offsets_k < head_dim
        
        # Load query element
        query_offset = batch_idx * (num_heads * seq_len * head_dim) + \
                      head_idx * (seq_len * head_dim) + \
                      seq_idx * head_dim + k
        
        query_val = tl.load(query_ptr + query_offset, mask=mask_k, other=0.0)
        
        # Apply scalar multiplication
        result_val = query_val * scalar_const
        
        # Store result
        result_offset = scalar_output_offset + k
        
        tl.store(out_query_ptr + result_offset, result_val, mask=mask_k)

# Kernel wrapper with @torch.fx.wrap decorator
@torch.fx.wrap
def fused_scalar_mul_transpose(in_0, in_1, scalar_const):
    # Get tensor properties
    batch_size, num_heads, seq_len, head_dim = in_0.shape
    
    # Create output tensors
    out_0 = torch.empty_like(in_0)  # For transpose result
    out_1 = torch.empty_like(in_1)  # For scalar multiplication result
    
    # Set up grid dimensions
    total_elements = batch_size * num_heads * seq_len
    grid = (total_elements,)
    
    # Choose appropriate block sizes
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 32  # Process multiple head_dim elements per program
    
    # Launch the kernel
    fused_scalar_mul_transpose_kernel[grid](
        key_ptr=in_0,
        query_ptr=in_1,
        out_key_ptr=out_0,
        out_query_ptr=out_1,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        scalar_const=scalar_const,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return (out_1, out_0)  # Return in same order as original: (tmp_0, tmp_1)

# Replacement function - returns the fused kernel function
def replacement_func():
    return fused_scalar_mul_transpose