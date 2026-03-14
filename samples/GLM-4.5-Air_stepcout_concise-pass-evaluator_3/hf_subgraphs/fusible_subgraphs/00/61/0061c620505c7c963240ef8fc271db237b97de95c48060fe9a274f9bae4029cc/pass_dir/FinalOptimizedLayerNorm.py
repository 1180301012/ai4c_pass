import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def optimized_layer_norm_transpose_kernel(
    x_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_features,
    seq_len,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1) 
    pid_feature = tl.program_id(2)
    
    # Compute memory offsets for this program
    batch_offset = pid_batch * seq_len * num_features
    seq_offset = pid_seq * num_features
    feature_offset = pid_feature * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid feature indices
    mask = feature_offset < num_features
    
    # Only proceed if this batch/seq position is valid
    if pid_batch >= batch_size or pid_seq >= seq_len:
        return
    
    # Efficient mean computation for this (batch, seq) position
    mean = 0.0
    for feature_idx in range(num_features):
        offset = batch_offset + seq_offset + feature_idx
        x_val = tl.load(x_ptr + offset)
        mean += x_val
    mean = mean / num_features
    
    # Efficient variance computation for this (batch, seq) position
    variance = 0.0
    for feature_idx in range(num_features):
        offset = batch_offset + seq_offset + feature_idx
        x_val = tl.load(x_ptr + offset)
        variance += (x_val - mean) * (x_val - mean)
    variance = tl.sqrt(variance / num_features + 1e-05)
    
    # Load weights and biases for the features in this block
    weights = tl.load(weight_ptr + feature_offset, mask=mask, other=0.0)
    biases = tl.load(bias_ptr + feature_offset, mask=mask, other=0.0)
    
    # Load input values for the features in this block
    input_offsets = batch_offset + seq_offset + feature_offset
    x_vals = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply layer normalization with improved numerical stability
    # Using: (x - mean) / sqrt(variance + eps) * weight + bias
    normalized = (x_vals - mean) / variance * weights + biases
    
    # Compute output offsets (transposed: [batch, features, seq])
    output_offsets = batch_offset + feature_offset * seq_len + pid_seq
    
    # Store the transposed result with proper masking
    tl.store(out_ptr + output_offsets, normalized, mask=mask)

@torch.fx.wrap
def optimized_layer_norm_transpose(x, weight, bias):
    batch_size, seq_len, num_features = x.shape
    
    # OPTIMIZATION: Use larger block size to reduce number of programs
    # For better performance, especially on larger batches
    BLOCK_SIZE = 1024  # Increased from 256 for better performance
    
    # Compute grid size - structure: (batch, seq, features/block_size)
    grid_batch = batch_size
    grid_seq = seq_len  
    grid_features = (num_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (transposed: [batch, features, seq])
    out = torch.empty((batch_size, num_features, seq_len), dtype=x.dtype, device=x.device)
    
    # Launch kernel with optimized parameters
    optimized_layer_norm_transpose_kernel[(grid_batch, grid_seq, grid_features)](
        x,
        weight,
        bias,
        out,
        num_features,
        seq_len,
        batch_size,
        BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm_transpose