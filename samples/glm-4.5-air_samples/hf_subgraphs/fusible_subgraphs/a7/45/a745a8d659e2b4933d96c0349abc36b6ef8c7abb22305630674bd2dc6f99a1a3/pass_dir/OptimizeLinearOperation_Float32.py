import torch
import triton
import triton.language as tl

@triton.jit
def optimized_linear_kernel(
    x_ptr,           # Input tensor: [batch, seq_len, in_features]
    weight_ptr,      # Weight matrix: [out_features, in_features] 
    out_ptr,         # Output tensor: [batch, seq_len, out_features]
    batch,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one output position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate output dimensions and boundaries
    m_offset = batch_idx * seq_len + seq_idx
    n_offset = tl.program_id(2) * BLOCK_SIZE_N
    
    m_end = batch * seq_len
    n_end = out_features
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Process input features in blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, in_features)
        
        # Load input data block
        x_ptr_local = x_ptr + (batch_idx * seq_len * in_features + seq_idx * in_features + k)
        x_block = tl.load(x_ptr_local + tl.arange(0, min(BLOCK_SIZE_K, k_end - k)),
                         mask=None)
        
        # Load weight data block  
        weight_ptr_local = weight_ptr + (n_offset * in_features + k)
        weight_block = tl.load(weight_ptr_local + tl.arange(0, BLOCK_SIZE_N * min(BLOCK_SIZE_K, k_end - k)),
                              stride=in_features, mask=None)
        weight_block = tl.reshape(weight_block, [BLOCK_SIZE_N, min(BLOCK_SIZE_K, k_end - k)])
        
        # Matrix multiplication
        acc += tl.dot(x_block, weight_block, acc_type=tl.float32)
    
    # Store result
    out_ptr_local = out_ptr + (m_offset * out_features + n_offset)
    tl.store(out_ptr_local + tl.arange(0, min(BLOCK_SIZE_N, n_end - n_offset)), acc)

@torch.fx.wrap
def triton_optimized_linear(x, weight):
    """Optimized linear operation using Triton"""
    batch, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    
    # Create output tensor
    out = torch.empty([batch, seq_len, out_features], dtype=torch.float32, device=x.device)
    
    # Block sizes optimized for typical matrix multiplication patterns
    BLOCK_SIZE_M = 1  # Process one batch*seq position at a time
    BLOCK_SIZE_N = 256  # Output features per block
    BLOCK_SIZE_K = 64   # Input features per block
    
    # Calculate grid dimensions
    num_batch_seq = batch * seq_len
    num_out_blocks = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    optimized_linear_kernel[(num_batch_seq, num_out_blocks)](
        x, weight, out,
        batch, seq_len, in_features, out_features,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

# Pattern: linear operation
def pattern(weight_tensor, input_tensor):
    return torch.nn.functional.linear(input_tensor, weight_tensor, None)

# Extract arguments for replacement
def replacement_args(weight_tensor, input_tensor):
    return (input_tensor, weight_tensor)

# Return optimized function
def replacement_func():
    return triton_optimized_linear