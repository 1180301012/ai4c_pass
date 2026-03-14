import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    seq_len,
    hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element in the batch-dimension
    batch_idx = tl.program_id(0)
    
    if batch_idx >= batch:
        return
    
    # Each thread processes one column in the hidden dimension
    ch_idx = tl.program_id(1)
    
    if ch_idx >= hidden_dim:
        return
    
    # For large sequence lengths, compute mean in a more efficient way
    # by dividing into blocks and combining results
    
    # Initialize accumulators
    total_mean = 0.0
    total_m2 = 0.0
    
    # Block size for mean/variance computation
    block_size = min(BLOCK_SIZE, seq_len)
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Compute mean using parallel reduction
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, seq_len)
        
        # Compute mean for this block
        block_mean = 0.0
        for i in range(start_idx, end_idx):
            x_val = tl.load(x_ptr + batch_idx * seq_len * hidden_dim + 
                           i * hidden_dim + ch_idx)
            block_mean += x_val
        
        block_mean = block_mean / (end_idx - start_idx)
        
        # Combine with overall mean (Welford's algorithm)
        if block_idx == 0:
            total_mean = block_mean
        else:
            delta = block_mean - total_mean
            total_mean += delta * (end_idx - start_idx) / seq_len
    
    # Compute variance using parallel reduction  
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, seq_len)
        
        # Compute M2 for this block
        for i in range(start_idx, end_idx):
            x_val = tl.load(x_ptr + batch_idx * seq_len * hidden_dim + 
                           i * hidden_dim + ch_idx)
            diff = x_val - total_mean
            total_m2 += diff * diff
    
    variance = total_m2 / seq_len + eps
    std = tl.sqrt(variance)
    
    # Apply normalization efficiently
    weight = tl.load(weight_ptr + ch_idx)
    bias = tl.load(bias_ptr + ch_idx)
    
    # Process output in blocks
    for block_idx in range(0, seq_len, BLOCK_SIZE):
        start_idx = block_idx
        end_idx = min(block_idx + BLOCK_SIZE, seq_len)
        
        for i in range(start_idx, end_idx):
            x_val = tl.load(x_ptr + batch_idx * seq_len * hidden_dim + 
                           i * hidden_dim + ch_idx)
            norm_val = (x_val - total_mean) / std
            result = norm_val * weight + bias
            tl.store(out_ptr + batch_idx * seq_len * hidden_dim + 
                    i * hidden_dim + ch_idx, result)

def pattern(tmp_7, tmp_1, tmp_0):
    """
    Pattern matches: LayerNorm operation
    tmp_7: input to layer norm
    tmp_1: weight for layer norm  
    tmp_0: bias for layer norm
    """
    # Try with explicit 128 channel size first
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (128,), tmp_1, tmp_0, 1e-05)
    return tmp_8

def replacement_args(tmp_7, tmp_1, tmp_0):
    """
    Extracts arguments for the replacement function
    """
    return (tmp_7, tmp_1, tmp_0)

@torch.fx.wrap  
def optimized_layer_norm(tmp_7, tmp_1, tmp_0):
    """
    Optimized LayerNorm using Triton kernel
    """
    batch, seq_len, hidden_dim = tmp_7.shape
    
    # Create output tensor
    output = torch.empty_like(tmp_7)
    
    # Move tensors to device
    tmp_7_d = tmp_7.to(tmp_1.device)
    tmp_1_d = tmp_1.to(tmp_7_d.device)
    tmp_0_d = tmp_0.to(tmp_7_d.device)
    
    # Set grid dimensions
    # One program per batch element * hidden dimension
    grid = (batch, hidden_dim)
    
    # Large block size for better efficiency
    BLOCK_SIZE = 1024
    
    # Launch kernel
    layer_norm_kernel[grid](
        tmp_7_d,
        tmp_1_d,
        tmp_0_d,
        output,
        batch,
        seq_len,
        hidden_dim,
        1e-05,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """
    Returns the optimized function
    """
    return optimized_layer_norm