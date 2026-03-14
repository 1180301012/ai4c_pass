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
def fused_layer_norm_transpose_kernel(
    x_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_features,
    seq_len,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
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
    
    # Check if there are any valid features in this block
    if tl.sum(mask) == 0:
        return
    
    # Compute mean for this (batch, seq) position over ALL features
    mean_sum = 0.0
    for feature_idx in range(num_features):
        offset = batch_offset + seq_offset + feature_idx
        x_val = tl.load(x_ptr + offset)
        mean_sum += x_val
    
    mean = mean_sum / num_features
    
    # Compute variance for this (batch, seq) position over ALL features
    variance_sum = 0.0
    for feature_idx in range(num_features):
        offset = batch_offset + seq_offset + feature_idx
        x_val = tl.load(x_ptr + offset)
        variance_sum += (x_val - mean) * (x_val - mean)
    
    variance = tl.sqrt(variance_sum / num_features + eps)
    
    # Load weights and biases for the features in this block
    weights = tl.load(weight_ptr + feature_offset, mask=mask, other=0.0)
    biases = tl.load(bias_ptr + feature_offset, mask=mask, other=0.0)
    
    # Load input values for the features in this block
    input_offsets = batch_offset + seq_offset + feature_offset
    x_vals = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply layer normalization
    normalized = (x_vals - mean) / variance * weights + biases
    
    # Compute output offsets (transposed: [batch, features, seq])
    output_offsets = batch_offset + feature_offset * seq_len + pid_seq
    
    # Store the transposed result
    tl.store(out_ptr + output_offsets, normalized, mask=mask)

@torch.fx.wrap
def fused_layer_norm_transpose(x, weight, bias):
    batch_size, seq_len, num_features = x.shape
    
    # Choose block size for feature dimension
    BLOCK_SIZE = 1024  # Feature dimension block size
    
    # Compute grid size
    # Each program handles one (batch, seq) pair and processes BLOCK_SIZE features
    grid_batch = batch_size
    grid_seq = seq_len
    grid_features = (num_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (transposed: [batch, features, seq])
    out = torch.empty((batch_size, num_features, seq_len), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_layer_norm_transpose_kernel[(grid_batch, grid_seq, grid_features)](
        x,
        weight,
        bias,
        out,
        num_features,
        seq_len,
        batch_size,
        BLOCK_SIZE,
        1e-05,
    )
    
    return out

def replacement_func():
    return fused_layer_norm_transpose