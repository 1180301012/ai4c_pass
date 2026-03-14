import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching the computation:
    1. Addition: in_2 += in_3
    2. Dropout: torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    3. Layer Normalization: torch.nn.functional.layer_norm(tmp_3, (feature_dim,), weight, bias, 1e-12)
    
    Returns both dropout output (tmp_3) and layer norm output (tmp_4)
    """
    # Match in-place addition
    in_2 += in_3
    tmp_2 = in_2
    # Match dropout with dropout_rate=0.1, training=False, inplace=False
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    # Match layer normalization with the given parameters
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (in_1.shape[0],), in_1, in_0, 1e-12)
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_dropout_layernorm_kernel(
    # Input tensors
    embeddings_ptr, position_embeddings_ptr,
    ln_weight_ptr, ln_bias_ptr,
    # Output tensors  
    dropout_output_ptr, layernorm_output_ptr,
    # Tensor metadata
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feature_dim: tl.constexpr,
    dropout_p: tl.constexpr,
    eps: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel for: Add + Dropout (scaling) + Layer Normalization
    
    Dropout with training=False is just multiplicative scaling, so we can fuse it with layer norm.
    """
    # Compute program id for 2D grid (batch, sequence)
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute base pointer offsets
    embeddings_offset = m * seq_len * feature_dim + n * feature_dim
    pos_embeddings_offset = embeddings_offset  # Same shape
    
    # Create offsets for loading
    offsets = embeddings_offset + tl.arange(0, BLOCK_N)
    mask = offsets < (batch_size * seq_len * feature_dim)
    
    # Load embeddings and position embeddings
    embeddings = tl.load(embeddings_ptr + offsets, mask=mask, other=0.0)
    pos_embeddings = tl.load(position_embeddings_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Addition (embeddings += position_embeddings)
    added = embeddings + pos_embeddings
    
    # Step 2: Dropout is just scaling by (1 - dropout_p) when training=False
    dropout_scale = 1.0 - dropout_p
    dropout_output = added * dropout_scale
    
    # Store dropout output
    tl.store(dropout_output_ptr + offsets, dropout_output, mask=mask)
    
    # Step 3: Layer Normalization
    # Load layer norm parameters
    ln_weight = tl.load(ln_weight_ptr + tl.arange(0, min(BLOCK_N, feature_dim)), mask=tl.arange(0, min(BLOCK_N, feature_dim)) < feature_dim)
    ln_bias = tl.load(ln_bias_ptr + tl.arange(0, min(BLOCK_N, feature_dim)), mask=tl.arange(0, min(BLOCK_N, feature_dim)) < feature_dim)
    
    # Compute mean and variance for layer normalization
    # For efficiency, compute mean across feature dimension for this (batch, seq) position
    if BLOCK_N >= feature_dim:
        # Handle full feature dimension in one block
        mean = tl.sum(dropout_output) / float(feature_dim)
        var = tl.sum((dropout_output - mean) * (dropout_output - mean)) / float(feature_dim)
    else:
        # Need to handle partial blocks - simplified for now
        mean = tl.sum(dropout_output) / float(min(BLOCK_N, feature_dim))
        var = tl.sum((dropout_output - mean) * (dropout_output - mean)) / float(min(BLOCK_N, feature_dim))
    
    # Apply layer normalization formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    layernorm_output = (dropout_output - mean) * inv_std * ln_weight + ln_bias
    
    # Store layer norm output
    tl.store(layernorm_output_ptr + offsets, layernorm_output, mask=mask)

@torch.fx.wrap
def fused_add_dropout_layernorm(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Args:
        in_0: Layer norm bias tensor [feature_dim]
        in_1: Layer norm weight tensor [feature_dim]  
        in_2: Embeddings tensor [batch_size, seq_len, feature_dim]
        in_3: Position embeddings tensor [batch_size, seq_len, feature_dim]
    """
    batch_size, seq_len, feature_dim = in_2.shape
    
    # Create output tensors
    dropout_output = torch.empty_like(in_2)
    layernorm_output = torch.empty_like(in_2)
    
    # Set up kernel grid
    grid = lambda meta: (
        (batch_size, seq_len),  # 2D grid for batch and sequence
        meta['BLOCK_M'], meta['BLOCK_N']  # Block sizes from metadata
    )
    
    # Launch kernel with autotuning configuration
    BLOCK_M = 1  # Process one batch at a time
    BLOCK_N = min(1024, feature_dim)  # Adjust based on feature dimension
    
    # Handle different tensor shapes by adjusting block sizes
    if feature_dim <= 64:
        BLOCK_N = 64
    elif feature_dim <= 256:
        BLOCK_N = 128
    elif feature_dim <= 512:
        BLOCK_N = 256
    else:
        BLOCK_N = 512
    
    fused_add_dropout_layernorm_kernel[grid](
        in_2, in_3,  # embeddings and position_embeddings
        in_1, in_0,  # ln_weight and ln_bias
        dropout_output, layernorm_output,
        batch_size, seq_len, feature_dim,  # tensor metadata
        0.1, 1e-12,  # dropout_p=0.1, eps=1e-12
        BLOCK_M, BLOCK_N  # block sizes
    )
    
    return dropout_output, layernorm_output

def replacement_func():
    """Returns the fused kernel function"""
    return fused_add_dropout_layernorm