import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    This pattern matches the LayerNorm computation in the model.
    The pattern includes:
    1. Element-wise addition between in_3 and in_2
    2. Type conversion to float32
    3. LayerNorm normalization (mean, variance, sqrt, division)
    4. Type conversion back to original dtype
    5. Scaling with in_1 (weight)
    6. Adding bias (in_0)
    """
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15

def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the optimized kernel.
    We need all four input tensors plus the epsilon value.
    """
    return (in_0, in_1, in_2, in_3)

@triton.jit
def layernorm_kernel(
    bias_ptr,    # in_0 - bias tensor [768]
    weight_ptr,  # in_1 - weight tensor [768] 
    x_ptr,       # in_2 - first input tensor [batch, seq_len, 768]
    residual_ptr,# in_3 - residual tensor [batch, seq_len, 768]
    output_ptr,  # output tensor [batch, seq_len, 768]
    batch_size,  # batch dimension
    seq_len,     # sequence length
    hidden_size, # hidden size (768)
    epsilon,     # epsilon for numerical stability
    BLOCK_SIZE_M: tl.constexpr, # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr, # Block size for sequence dimension  
    BLOCK_SIZE_K: tl.constexpr  # Block size for hidden dimension (must be compile-time constant)
):
    """
    LayerNorm kernel with fixed-size vector operations to avoid compilation errors.
    """
    pid_m = tl.program_id(0)  # batch dimension  
    pid_n = tl.program_id(1)  # sequence dimension
    
    # Only process valid ranges
    if pid_m >= batch_size or pid_n >= seq_len:
        return
    
    # Base offset for this (batch, seq) position
    offset = (pid_m * seq_len + pid_n) * hidden_size
    
    # Initialize reduction accumulators
    sum_val = 0.0
    
    # First, compute the mean (single pass)
    indices = tl.arange(0, BLOCK_SIZE_K)
    
    for k_offset in range(0, hidden_size, BLOCK_SIZE_K):
        # Create mask for valid indices within this block
        mask = (indices < (hidden_size - k_offset)) & (k_offset < hidden_size)
        
        if k_offset < hidden_size:
            # Load input tensors for this block
            x_vals = tl.load(x_ptr + offset + k_offset + indices, mask=mask, other=0.0).to(tl.float32)
            residual_vals = tl.load(residual_ptr + offset + k_offset + indices, mask=mask, other=0.0).to(tl.float32)
            
            # Add residual and accumulate sum
            sum_vals = x_vals + residual_vals
            sum_val += tl.sum(sum_vals)
    
    # Compute mean
    mean = sum_val / hidden_size
    
    # Second, compute variance using the mean (more numerically stable)
    var_sum = 0.0
    
    for k_offset in range(0, hidden_size, BLOCK_SIZE_K):
        # Create mask for valid indices within this block
        mask = (indices < (hidden_size - k_offset)) & (k_offset < hidden_size)
        
        if k_offset < hidden_size:
            # Load input tensors for this block
            x_vals = tl.load(x_ptr + offset + k_offset + indices, mask=mask, other=0.0).to(tl.float32)
            residual_vals = tl.load(residual_ptr + offset + k_offset + indices, mask=mask, other=0.0).to(tl.float32)
            
            # Add residual and compute squared differences from mean
            sum_vals = x_vals + residual_vals
            diff = sum_vals - mean
            var_sum += tl.sum(diff * diff)
    
    var = var_sum / hidden_size
    
    # Load bias and weight (process them in chunks too)
    # We'll assume BLOCK_SIZE_K is small enough to handle with a single masked load
    bias_block = tl.load(bias_ptr + indices, mask=indices < hidden_size, other=0.0)
    weight_block = tl.load(weight_ptr + indices, mask=indices < hidden_size, other=0.0)
    
    # Apply normalization and scale/bias
    for k_offset in range(0, hidden_size, BLOCK_SIZE_K):
        # Create mask for valid indices within this block
        mask = (indices < (hidden_size - k_offset)) & (k_offset < hidden_size)
        
        if k_offset < hidden_size:
            # Load input tensors for this block
            x_vals = tl.load(x_ptr + offset + k_offset + indices, mask=mask, other=0.0).to(tl.float32)
            residual_vals = tl.load(residual_ptr + offset + k_offset + indices, mask=mask, other=0.0).to(tl.float32)
            
            # Compute normalized output
            # Note: Since we're processing in blocks, we need to broadcast mean/var to each block
            normalized = (x_vals + residual_vals - mean) / tl.sqrt(var + epsilon)
            output = normalized * weight_block + bias_block
            
            # Store result
            tl.store(output_ptr + offset + k_offset + indices, output, mask=mask)

@torch.fx.wrap
def optimized_layernorm(in_0, in_1, in_2, in_3):
    """
    Optimized LayerNorm function that computes the entire operation in a single kernel.
    """
    # Get tensor shapes and compute grid dimensions
    batch_size, seq_len, hidden_size = in_2.shape
    
    # Set block sizes
    BLOCK_SIZE_M = 1  # Process one batch at a time for better memory locality
    BLOCK_SIZE_N = 64  # Process multiple sequence elements simultaneously
    BLOCK_SIZE_K = 256  # Process multiple hidden features simultaneously (compile-time constant)
    
    # Calculate grid dimensions (2D grid for batch and seq)
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    output = torch.empty_like(in_2, dtype=torch.float32)
    
    # Launch kernel
    layernorm_kernel[(grid_m, grid_n)](
        in_0, in_1, in_2, in_3, output,
        batch_size, seq_len, hidden_size,
        1e-07,  # epsilon
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    """
    Returns the optimized kernel function.
    """
    return optimized_layernorm