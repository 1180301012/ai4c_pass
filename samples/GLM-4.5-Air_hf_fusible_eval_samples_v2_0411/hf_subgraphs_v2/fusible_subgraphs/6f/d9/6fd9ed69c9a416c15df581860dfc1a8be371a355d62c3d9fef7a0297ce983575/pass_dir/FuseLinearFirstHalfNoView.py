import torch
import triton
import triton.language as tl

def pattern(input_feat, weight, bias):
    """Match linear + first half slice pattern (no view)"""
    linear = torch.nn.functional.linear(input_feat, weight, bias)
    sliced = linear[..., :256]  # First 256 columns with ellipsis
    return linear, sliced  # Return intermediates for observability

def replacement_args(input_feat, weight, bias):
    return (input_feat, weight, bias)

@triton.jit
def linear_first_half_no_view_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, hidden_size, in_features, 
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Custom Triton kernel for linear transformation and first half slice without view"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load input features (3D tensors: batch, seq, features)
    input_ptrs = input_ptr + (offsets[:, None, None] * seq_len * in_features + 
                              tl.arange(seq_len)[None, :, None] * in_features + 
                              tl.arange(0, in_features)[None, None, :])
    input_vals = tl.load(input_ptrs, mask=mask[:, None, None] & 
                        (tl.arange(seq_len)[None, :, None] < seq_len) & 
                        (tl.arange(0, in_features)[None, None, :] < in_features), other=0.0)
    
    # Reshape for efficient computation: (batch*seq, features)
    batch_seq = batch_size * seq_len
    input_flat = input_vals.reshape((batch_seq, in_features))
    
    # Load weights (only first 256 columns)
    weight_ptrs = weight_ptr + tl.arange(0, 256)[:, None] * in_features + tl.arange(0, in_features)[None, :]
    weight_vals = tl.load(weight_ptrs, mask=(tl.arange(0, 256)[:, None] < 256) & (tl.arange(0, in_features)[None, :] < in_features))
    
    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_SIZE, seq_len, 256), dtype=tl.float32)
    for k in range(0, in_features, 32):
        input_block = tl.load(input_ptrs + k, mask=mask[:, None, None] & 
                            (tl.arange(seq_len)[None, :, None] < seq_len) & 
                            (tl.arange(k, k+32)[None, None, :] < in_features), other=0.0)
        input_flat_block = input_block.reshape((BLOCK_SIZE * seq_len, 32))
        weight_block = tl.load(weight_ptrs + k[None, :], mask=(tl.arange(0, 256)[:, None] < 256) & (tl.arange(k, k+32)[None, :] < in_features))
        acc_flat = acc.reshape((BLOCK_SIZE * seq_len, 256))
        acc_flat += tl.dot(input_flat_block, weight_block, acc_type=tl.float32)
    
    # Add bias
    bias_ptrs = bias_ptr + tl.arange(0, 256)
    bias_vals = tl.load(bias_ptrs, mask=tl.arange(0, 256) < 256)
    bias_reshaped = bias_vals.reshape((1, 1, 256))
    acc += bias_reshaped
    
    # Store only first 256 columns
    out_ptrs = out_ptr + (offsets[:, None, None] * seq_len * 256 + 
                          tl.arange(seq_len)[None, :, None] * 256 + 
                          tl.arange(0, 256)[None, None, :])
    tl.store(out_ptrs, acc, mask=mask[:, None, None])
    
    return acc

@torch.fx.wrap
def fused_linear_first_half_no_view(input_feat, weight, bias):
    # Handle 3D input: (batch, seq, features)
    batch_size = input_feat.shape[0]
    seq_len = input_feat.shape[1]
    in_features = input_feat.shape[2]
    
    input_reshaped = input_feat.reshape(-1, in_features)
    total_batch = input_reshaped.shape[0]
        
    BLOCK_SIZE = 256
    num_programs = (total_batch + BLOCK_SIZE - 1) // BLOCK_SIZE
        
    # Output tensor for first half
    out_first_half = torch.empty((batch_size, seq_len, 256), dtype=input_feat.dtype, device=input_feat.device)
    
    # Launch kernel
    linear_first_half_no_view_kernel[(num_programs,)](
        input_reshaped,
        weight,
        bias,
        out_first_half,
        batch_size,
        256,  # hidden_size
        in_features,
        seq_len,  # additional dimension
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_first_half, out_first_half

def replacement_func():
    return fused_linear_first_half_no_view