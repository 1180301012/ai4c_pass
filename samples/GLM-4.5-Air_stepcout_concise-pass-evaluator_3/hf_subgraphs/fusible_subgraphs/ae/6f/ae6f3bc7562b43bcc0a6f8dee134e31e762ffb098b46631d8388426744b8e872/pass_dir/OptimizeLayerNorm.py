import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # x: tmp_10, [4, 512, 1280] 
    # weight: tmp_3, [1280]
    # bias: tmp_2, [1280]
    
    tmp_11 = torch.nn.functional.layer_norm(x, (1280,), weight, bias, 1e-12)
    return tmp_11

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def optimized_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_sequences,
    seq_len,
    embed_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Total number of sequence positions
    total_positions = num_sequences * seq_len
    
    # Program identifier
    pid = tl.program_id(0)
    
    # Memory offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_positions
    
    # Load weight and bias (they are per-embedding dimension)
    weight = tl.load(weight_ptr + tl.arange(0, embed_dim), mask=tl.arange(0, embed_dim) < embed_dim, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, embed_dim), mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
    
    # Process each sequence position in the block
    for i in range(BLOCK_SIZE):
        if offsets[i] < total_positions:
            seq_idx = offsets[i] // seq_len
            pos_idx = offsets[i] % seq_len
            
            # Load input values for this sequence position
            x_offset = (seq_idx * seq_len + pos_idx) * embed_dim + tl.arange(0, embed_dim)
            x_vals = tl.load(x_ptr + x_offset, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
            
            # Compute mean and variance
            mean = tl.sum(x_vals) / embed_dim
            mean_sq = tl.sum(x_vals * x_vals) / embed_dim
            var = mean_sq - mean * mean
            
            # Add epsilon for numerical stability
            var = var + eps
            std = tl.sqrt(var)
            
            # Normalize: (x - mean) / std  
            x_norm = (x_vals - mean) / std
            
            # Apply scale and shift
            out_vals = x_norm * weight + bias
            
            # Store results
            tl.store(out_ptr + x_offset, out_vals, mask=tl.arange(0, embed_dim) < embed_dim)

@torch.fx.wrap  
def optimized_layer_norm(x, weight, bias):
    num_sequences, seq_len, embed_dim = x.shape
    
    # Use a conservative block size that works with power-of-2 constraints
    BLOCK_SIZE = 256  # Number of sequence positions per program
    
    # Calculate grid size
    total_positions = num_sequences * seq_len
    grid_size = (total_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel with 1D grid
    optimized_layernorm_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        num_sequences=num_sequences,
        seq_len=seq_len,
        embed_dim=embed_dim,
        eps=1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm