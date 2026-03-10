import torch
import triton
import triton.language as tl

# Pattern matching for Branch 2: Flatten -> Transpose
def pattern(input_tensor):
    """
    Pattern: Flatten(2) -> Transpose(1, 2)
    """
    # Flatten starting from dimension 2
    tmp_6 = input_tensor.flatten(2)
    
    # Transpose dimensions 1 and 2
    tmp_7 = tmp_6.transpose(1, 2)
    
    return tmp_7

def replacement_args(input_tensor):
    """Extract arguments for the optimized kernel"""
    return (input_tensor,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr, output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for flatten + transpose operation"""
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len * feature_dim
    
    if pid * BLOCK_SIZE >= total_elements:
        return
    
    # Calculate coordinates in input: (B, S, F) -> output: (B, F, S)
    idx = pid * BLOCK_SIZE
    b = idx // (seq_len * feature_dim)
    s = (idx % (seq_len * feature_dim)) // feature_dim
    f = idx % feature_dim
    
    # Skip if out of bounds
    if b >= batch_size:
        return
    if s >= seq_len:
        return
    if f >= feature_dim:
        return
    
    # Load input value
    input_val = tl.load(input_ptr + (b * seq_len * feature_dim + s * feature_dim + f))
    
    # Calculate output coordinates: transpose S <-> F
    output_idx = (b * feature_dim * seq_len + f * seq_len + s)
    
    # Store output value
    tl.store(output_ptr + output_idx, input_val)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    """Wrapper function for the optimized flatten + transpose kernel"""
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1] 
    feature_dim = input_tensor.shape[2] * input_tensor.shape[3]
    
    # Create output tensor with shape (B, F, S)
    output = torch.empty((batch_size, feature_dim, seq_len), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch optimized kernel
    BLOCK_SIZE = 1024
    total_elements = batch_size * seq_len * feature_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    flatten_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_dim=feature_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized kernel function"""
    return optimized_flatten_transpose