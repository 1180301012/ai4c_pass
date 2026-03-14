import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Simple mean operation pattern
    return input_tensor.mean(dim=-2, keepdim=True)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_mean_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles features for one batch
    pid = tl.program_id(0)
    batch_idx = pid // (feature_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    feature_idx = (pid // BLOCK_SIZE) % feature_size
    local_feature_idx = (pid % BLOCK_SIZE)
    
    # Check bounds
    if batch_idx >= batch_size or feature_idx >= feature_size:
        return
    
    # Initialize sum for this batch and feature
    sum_val = 0.0
    count = 0
    
    # Process sequence dimension (dim=-2, which is 1) with vectorization
    for seq_idx in range(0, seq_len, BLOCK_SIZE):
        seq_end = min(seq_idx + BLOCK_SIZE, seq_len)
        seq_mask = seq_idx + tl.arange(0, seq_end - seq_idx) < seq_len
        
        # Load input values
        input_offset = batch_idx * seq_len * feature_size + feature_idx * seq_len + seq_idx
        input_vals = tl.load(input_ptr + input_offset, mask=seq_mask, other=0.0)
        
        # Accumulate sum
        sum_val += tl.sum(input_vals)
        count += tl.sum(seq_mask)
    
    # Compute mean
    if count > 0:
        mean_val = sum_val / count
    else:
        mean_val = 0.0
    
    # Store result
    output_offset = batch_idx * feature_size + feature_idx
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap  
def optimized_mean_keepdim(input_tensor):
    # Get tensor dimensions
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]  # This is dim=-2
    feature_size = input_tensor.shape[2]
    
    # Determine grid configuration
    BLOCK_SIZE = 256  # Adjust for optimal performance
    num_programs = (batch_size * feature_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor [batch_size, 1, feature_size]
    output = torch.empty((batch_size, 1, feature_size), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten output for easier kernel access, then reshape back
    output_flat = output.view(batch_size * feature_size)
    
    # Launch kernel
    optimized_mean_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_flat,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_size=feature_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_mean_keepdim